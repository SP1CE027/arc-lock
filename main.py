import cv2
import numpy as np
import time
import ctypes
from collections import deque
from pathlib import Path
from insightface.app import FaceAnalysis


# ============================================================
# CONFIG
# ============================================================

EMBEDDING_FILE = Path('data/yash_embeddings.npy')

MATCH_THRESHOLD = 0.38
VERIFY_INTERVAL = 0.8
DETECTION_INTERVAL = 3.0
LOCK_TIMEOUT = 7.0
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
locked_once = False

# Tracking
tracker = None
tracking = False
track_box = None

# Cached recognition
last_embedding = None


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

    print(
        f'Loaded {len(SAVED_EMBEDDINGS)} embeddings'
    )

else:

    SAVED_EMBEDDINGS = None
    SAVED_NORMS = None

    print('WARNING: No face embeddings found')


# ============================================================
# INITIALIZE ARCFACE
# ============================================================

app = FaceAnalysis(
    name='buffalo_l',
    allowed_modules=['detection', 'recognition']
)

app.prepare(
    ctx_id=0,
    det_size=(320, 320)
)

print('ArcFace initialized')


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

    print('Could not open webcam')
    raise SystemExit(1)


print('Webcam opened')
print('ArcLock running in headless mode')


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # READ FRAME
    # --------------------------------------------------------

    ret, frame = cap.read()

    if not ret:

        time.sleep(0.1)
        continue


    now = time.monotonic()


    # --------------------------------------------------------
    # FULL RETINAFACE DETECTION
    # --------------------------------------------------------

    if (
        not tracking
        or
        now - last_detection >= DETECTION_INTERVAL
    ):

        last_detection = now

        faces = app.get(frame)


        if len(faces) == 1:

            face = faces[0]

            x1, y1, x2, y2 = face.bbox.astype(int)

            track_box = (
                x1,
                y1,
                x2 - x1,
                y2 - y1
            )


            # Cache embedding from this detection

            last_embedding = (
                face.embedding.astype(np.float32)
            )


            # Start tracker

            if hasattr(
                cv2,
                'TrackerKCF_create'
            ):

                tracker = cv2.TrackerKCF_create()

            else:

                tracker = cv2.legacy.TrackerKCF_create()


            tracker.init(
                frame,
                track_box
            )

            tracking = True


        else:

            tracking = False
            track_box = None
            last_embedding = None


    # --------------------------------------------------------
    # TRACK BETWEEN DETECTIONS
    # --------------------------------------------------------

    elif tracking:

        ok, box = tracker.update(frame)


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

            tracking = False
            track_box = None
            last_embedding = None


    # --------------------------------------------------------
    # VERIFY USING CACHED EMBEDDING
    # --------------------------------------------------------

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


        dots = (
            SAVED_EMBEDDINGS @ current
        )


        dists = (
            1
            -
            dots / (
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


        # ----------------------------------------------------
        # VERIFIED
        # ----------------------------------------------------

        if verified_count >= REQUIRED_MATCHES:

            last_verified = now

            locked_once = False

            print(
                f'VERIFIED {best:.3f}'
            )


        # ----------------------------------------------------
        # UNCERTAIN
        # ----------------------------------------------------

        elif verified_count >= 1:

            print(
                f'UNCERTAIN {best:.3f}'
            )


        # ----------------------------------------------------
        # NOT VERIFIED
        # ----------------------------------------------------

        else:

            print(
                f'NOT VERIFIED {best:.3f}'
            )


    # --------------------------------------------------------
    # NO FACE
    # --------------------------------------------------------

    if not tracking:

        history.append(False)

        print('NO FACE')


    # --------------------------------------------------------
    # LOCK TIMER
    # --------------------------------------------------------

    elapsed = (
        now - last_verified
    )


    if paused:

        # Keep the monitoring state alive,
        # but do not lock while paused.

        pass


    elif elapsed >= LOCK_TIMEOUT:

        if (
            ENABLE_LOCK
            and not locked_once
        ):

            locked_once = True

            print('LOCKING')

            try:

                ctypes.windll.user32.LockWorkStation()

            except Exception as e:

                print(
                    f'Lock failed: {e}'
                )


            # Reset state after Windows
            # returns control to the user.

            history.clear()

            tracking = False
            track_box = None
            last_embedding = None

            last_verified = (
                time.monotonic()
            )

            print(
                'Waiting for user after unlock'
            )


    # --------------------------------------------------------
    # SMALL SLEEP
    # --------------------------------------------------------

    time.sleep(0.01)


# ============================================================
# CLEANUP
# ============================================================

cap.release()