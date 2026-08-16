import cv2
import numpy as np
import time
import ctypes
import threading

from collections import deque
from pathlib import Path
from insightface.app import FaceAnalysis

from tray import TrayApp
from logs import start_log_window, log, stop_log_window


# ============================================================
# CONFIG
# ============================================================

EMBEDDING_FILE = Path('data/yash_embeddings.npy')

MATCH_THRESHOLD = 0.38
VERIFY_INTERVAL = 0.7
DETECTION_INTERVAL = 3.0
LOCK_TIMEOUT = 8.0
ENABLE_LOCK = True

# Temporal voting
HISTORY_SIZE = 5
REQUIRED_MATCHES = 3


# ============================================================
# STATE
# ============================================================

history = deque(maxlen=HISTORY_SIZE)

last_detection = 0.0
last_verify = 0.0
last_verified = time.monotonic()

paused = False
pause_until = None
locked_once = False
running = True

# Tracking
tracker = None
tracking = False
track_box = None

# Cached recognition
last_embedding = None

# Used to safely coordinate tray callbacks
state_lock = threading.Lock()


# ============================================================
# LOGGING
# ============================================================

start_log_window()


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

if EMBEDDING_FILE.exists():

    SAVED_EMBEDDINGS = np.load(
        EMBEDDING_FILE
    ).astype(np.float32)

    SAVED_NORMS = np.linalg.norm(
        SAVED_EMBEDDINGS,
        axis=1
    )

    log(
        f'Loaded {len(SAVED_EMBEDDINGS)} embeddings'
    )

else:

    SAVED_EMBEDDINGS = None
    SAVED_NORMS = None

    log('WARNING: No face embeddings found')


# ============================================================
# INITIALIZE ARCFACE
# ============================================================

log('Initializing ArcFace...')

app = FaceAnalysis(
    name='buffalo_l',
    allowed_modules=['detection', 'recognition']
)

app.prepare(
    ctx_id=0,
    det_size=(320, 320)
)

log('ArcFace initialized')


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)

cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)

if not cap.isOpened():

    log('ERROR: Could not open webcam')
    raise SystemExit(1)


log('Webcam opened')
log('ArcLock running in headless mode')


# ============================================================
# RESET TRACKING
# ============================================================

def reset_tracking():

    global tracker
    global tracking
    global track_box
    global last_embedding

    tracker = None
    tracking = False
    track_box = None
    last_embedding = None


# ============================================================
# PAUSE / RESUME
# ============================================================

def pause_monitoring():

    global paused
    global pause_until

    with state_lock:

        paused = True
        pause_until = None

        history.clear()

        reset_tracking()

    log('Monitoring paused')


def resume_monitoring():

    global paused
    global pause_until
    global locked_once
    global last_verified
    global last_detection
    global last_verify

    with state_lock:

        paused = False
        pause_until = None

        history.clear()

        reset_tracking()

        locked_once = False

        now = time.monotonic()

        last_verified = now
        last_detection = 0.0
        last_verify = 0.0

    log('Monitoring resumed')


def pause_for(seconds):

    global paused
    global pause_until

    with state_lock:

        paused = True

        history.clear()

        reset_tracking()

        if seconds is None:

            pause_until = None

            log(
                'Monitoring paused until restart'
            )

        else:

            pause_until = (
                time.monotonic() + seconds
            )

            log(
                f'Monitoring paused for '
                f'{seconds // 3600} hour(s)'
            )


def quit_app():

    global running

    with state_lock:

        running = False

    log('ArcLock stopping...')


# ============================================================
# SYSTEM TRAY
# ============================================================

tray = TrayApp(
    pause_callback=pause_monitoring,
    resume_callback=resume_monitoring,
    pause_for_callback=pause_for,
    quit_callback=quit_app,
)

tray_thread = threading.Thread(
    target=tray.run,
    daemon=True
)

tray_thread.start()


# ============================================================
# MAIN ENGINE
# ============================================================

try:

    while running:

        # ----------------------------------------------------
        # CHECK PAUSE TIMER
        # ----------------------------------------------------

        with state_lock:

            currently_paused = paused
            current_pause_until = pause_until

        if (
            currently_paused
            and current_pause_until is not None
            and time.monotonic() >= current_pause_until
        ):

            resume_monitoring()

            continue


        # ----------------------------------------------------
        # PAUSED
        # ----------------------------------------------------

        if currently_paused:

            time.sleep(0.1)
            continue


        # ----------------------------------------------------
        # READ FRAME
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:

            reset_tracking()

            log('WARNING: Webcam frame capture failed')

            time.sleep(0.1)
            continue


        now = time.monotonic()


        # ----------------------------------------------------
        # FULL RETINAFACE DETECTION
        # ----------------------------------------------------

        if (
            not tracking
            or
            now - last_detection >= DETECTION_INTERVAL
        ):

            last_detection = now

            try:

                faces = app.get(frame)

            except Exception as e:

                log(
                    f'Detection error: {e}'
                )

                reset_tracking()

                time.sleep(0.1)
                continue


            # ------------------------------------------------
            # EXACTLY ONE FACE
            # ------------------------------------------------

            if len(faces) == 1:

                face = faces[0]

                x1, y1, x2, y2 = (
                    face.bbox.astype(int)
                )

                track_box = (
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1
                )


                # --------------------------------------------
                # CACHE EMBEDDING
                # --------------------------------------------

                last_embedding = (
                    face.embedding.astype(
                        np.float32
                    )
                )


                # --------------------------------------------
                # START KCF TRACKER
                # --------------------------------------------

                try:

                    if hasattr(
                        cv2,
                        'TrackerKCF_create'
                    ):

                        tracker = (
                            cv2.TrackerKCF_create()
                        )

                    else:

                        tracker = (
                            cv2.legacy
                            .TrackerKCF_create()
                        )


                    tracker.init(
                        frame,
                        track_box
                    )

                    tracking = True

                except Exception as e:

                    log(
                        f'Tracker initialization '
                        f'failed: {e}'
                    )

                    reset_tracking()


            # ------------------------------------------------
            # ZERO OR MULTIPLE FACES
            # ------------------------------------------------

            else:

                reset_tracking()


        # ----------------------------------------------------
        # TRACK BETWEEN DETECTIONS
        # ----------------------------------------------------

        elif tracking:

            try:

                ok, box = tracker.update(
                    frame
                )


                if ok:

                    x, y, w, h = map(
                        int,
                        box
                    )

                    track_box = (
                        x,
                        y,
                        w,
                        h
                    )

                else:

                    reset_tracking()


            except Exception as e:

                log(
                    f'Tracker error: {e}'
                )

                reset_tracking()


        # ----------------------------------------------------
        # VERIFY USING CACHED EMBEDDING
        # ----------------------------------------------------

        if (
            not paused
            and tracking
            and last_embedding is not None
            and SAVED_EMBEDDINGS is not None
            and now - last_verify >= VERIFY_INTERVAL
        ):

            last_verify = now

            current = last_embedding

            current_norm = np.linalg.norm(
                current
            )


            if current_norm > 0:

                dots = (
                    SAVED_EMBEDDINGS
                    @ current
                )


                dists = (
                    1
                    -
                    dots
                    /
                    (
                        SAVED_NORMS
                        *
                        current_norm
                    )
                )


                best = float(
                    np.min(dists)
                )


                is_match = (
                    best < MATCH_THRESHOLD
                )


                history.append(
                    is_match
                )


                verified_count = sum(
                    history
                )


                # --------------------------------------------
                # VERIFIED
                # --------------------------------------------

                if (
                    verified_count
                    >= REQUIRED_MATCHES
                ):

                    last_verified = now

                    locked_once = False

                    log(
                        f'VERIFIED {best:.3f}'
                    )


                # --------------------------------------------
                # UNCERTAIN
                # --------------------------------------------

                elif verified_count >= 1:

                    log(
                        f'UNCERTAIN {best:.3f}'
                    )


                # --------------------------------------------
                # NOT VERIFIED
                # --------------------------------------------

                else:

                    log(
                        f'NOT VERIFIED {best:.3f}'
                    )


        # ----------------------------------------------------
        # NO FACE
        # ----------------------------------------------------

        if not tracking:

            history.append(False)


        # ----------------------------------------------------
        # LOCK TIMER
        # ----------------------------------------------------

        elapsed = (
            now - last_verified
        )


        if (
            not paused
            and elapsed >= LOCK_TIMEOUT
            and ENABLE_LOCK
            and not locked_once
        ):

            locked_once = True

            log('LOCKING')

            try:

                ctypes.windll.user32.LockWorkStation()

            except Exception as e:

                log(
                    f'Lock failed: {e}'
                )


            # ------------------------------------------------
            # RESET AFTER LOCK
            # ------------------------------------------------

            history.clear()

            reset_tracking()

            last_verified = (
                time.monotonic()
            )

            last_detection = 0.0
            last_verify = 0.0

            log(
                'Waiting for user after unlock'
            )


        # ----------------------------------------------------
        # SMALL SLEEP
        # ----------------------------------------------------

        time.sleep(0.01)


finally:

    # ========================================================
    # CLEANUP
    # ========================================================

    running = False

    try:

        cap.release()

    except Exception:
        pass


    try:

        tray.icon.stop()

    except Exception:
        pass


    log('ArcLock stopped')

    stop_log_window()
    import tkinter as tk
from tkinter.scrolledtext import ScrolledText

from datetime import datetime

import queue
import threading


# ============================================================
# STATE
# ============================================================

_command_queue = queue.Queue()

_log_history = []

_started = False

_start_lock = threading.Lock()

_tk_thread = None


# ============================================================
# TKINTER THREAD
# ============================================================

def _tk_worker():

    root = tk.Tk()

    root.title("ArcLock Logs")

    root.geometry(
        "700x400"
    )

    root.minsize(
        500,
        250
    )


    # --------------------------------------------------------
    # LOG TEXT AREA
    # --------------------------------------------------------

    text = ScrolledText(
        root,
        state="disabled",
        font=("Consolas", 10)
    )

    text.pack(
        fill="both",
        expand=True
    )


    # --------------------------------------------------------
    # CLOSE = HIDE
    # --------------------------------------------------------

    root.protocol(
        "WM_DELETE_WINDOW",
        root.withdraw
    )


    # Start hidden
    root.withdraw()


    # --------------------------------------------------------
    # INSERT LOG
    # --------------------------------------------------------

    def insert_log(line):

        text.configure(
            state="normal"
        )

        text.insert(
            "end",
            line
        )

        text.see(
            "end"
        )

        text.configure(
            state="disabled"
        )


    # --------------------------------------------------------
    # LOAD EXISTING LOGS
    # --------------------------------------------------------

    for line in _log_history:

        insert_log(line)


    # --------------------------------------------------------
    # PROCESS QUEUE
    # --------------------------------------------------------

    def process_commands():

        try:

            while True:

                command = (
                    _command_queue.get_nowait()
                )


                # --------------------------------------------
                # SHOW WINDOW
                # --------------------------------------------

                if command == "show":

                    root.deiconify()

                    root.lift()

                    root.attributes(
                        "-topmost",
                        True
                    )

                    root.after(
                        100,
                        lambda: root.attributes(
                            "-topmost",
                            False
                        )
                    )

                    root.focus_force()


                # --------------------------------------------
                # HIDE WINDOW
                # --------------------------------------------

                elif command == "hide":

                    root.withdraw()


                # --------------------------------------------
                # QUIT
                # --------------------------------------------

                elif command == "quit":

                    root.destroy()

                    return


                # --------------------------------------------
                # LOG MESSAGE
                # --------------------------------------------

                elif (
                    isinstance(command, tuple)
                    and command[0] == "log"
                ):

                    insert_log(
                        command[1]
                    )


        except queue.Empty:

            pass


        # Check again in 100 ms

        root.after(
            100,
            process_commands
        )


    # --------------------------------------------------------
    # START QUEUE PROCESSING
    # --------------------------------------------------------

    process_commands()


    # --------------------------------------------------------
    # TKINTER MAIN LOOP
    # --------------------------------------------------------

    root.mainloop()


# ============================================================
# START LOG WINDOW
# ============================================================

def start_log_window():

    global _started
    global _tk_thread

    with _start_lock:

        if _started:

            return

        _started = True

        _tk_thread = threading.Thread(
            target=_tk_worker,
            name="ArcLock-LogWindow",
            daemon=True
        )

        _tk_thread.start()


# ============================================================
# SHOW LOG WINDOW
# ============================================================

def show_logs(
    icon=None,
    item=None
):

    if not _started:

        start_log_window()


    _command_queue.put(
        "show"
    )


# ============================================================
# WRITE LOG
# ============================================================

def log(message):

    timestamp = datetime.now().strftime(
        "%H:%M:%S"
    )

    line = (
        f"[{timestamp}] "
        f"{message}\n"
    )


    # Keep an in-memory copy so logs that happen before
    # the window is opened are still displayed.

    _log_history.append(
        line
    )


    # Send the message to the Tkinter thread.

    if _started:

        _command_queue.put(
            (
                "log",
                line
            )
        )


# ============================================================
# STOP LOG WINDOW
# ============================================================

def stop_log_window():

    if not _started:

        return


    _command_queue.put(
        "quit"
    )