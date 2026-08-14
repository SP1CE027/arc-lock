import cv2
import numpy as np
import time
import ctypes
from collections import deque
from pathlib import Path
from insightface.app import FaceAnalysis

EMBEDDING_FILE = Path('data/yash_embeddings.npy')

MATCH_THRESHOLD = 0.38
VERIFY_INTERVAL = 0.5
DETECTION_INTERVAL = 3.0
LOCK_TIMEOUT = 7.0
ENABLE_LOCK = True

# Temporal voting
HISTORY_SIZE = 5
REQUIRED_MATCHES = 3

history = deque(maxlen=HISTORY_SIZE)

# Load embeddings once
if EMBEDDING_FILE.exists():
    SAVED_EMBEDDINGS = np.load(EMBEDDING_FILE).astype(np.float32)
    SAVED_NORMS = np.linalg.norm(SAVED_EMBEDDINGS, axis=1)
else:
    SAVED_EMBEDDINGS = None
    SAVED_NORMS = None

# Initialize ArcFace (detection + recognition only)
app = FaceAnalysis(
    name='buffalo_l',
    allowed_modules=['detection', 'recognition']
)
app.prepare(ctx_id=0, det_size=(320, 320))

# Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print('Could not open webcam')
    exit()

print('Press P to pause/resume')
print('Press Q to quit')

status_text = 'Starting'
status_color = (0, 255, 0)

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

while True:

    ret, frame = cap.read()

    if not ret:
        time.sleep(0.1)
        continue

    now = time.monotonic()

    # -------- FULL RETINAFACE DETECTION (rare) --------
    if (not tracking) or (now - last_detection >= DETECTION_INTERVAL):

        last_detection = now

        faces = app.get(frame)

        if len(faces) == 1:

            face = faces[0]

            x1, y1, x2, y2 = face.bbox.astype(int)

            track_box = (x1, y1, x2 - x1, y2 - y1)

            # Cache embedding from this detection
            last_embedding = face.embedding.astype(np.float32)

            # Start tracker
            if hasattr(cv2, 'TrackerKCF_create'):
                tracker = cv2.TrackerKCF_create()
            else:
                tracker = cv2.legacy.TrackerKCF_create()

            tracker.init(frame, track_box)

            tracking = True

        else:

            tracking = False
            track_box = None
            last_embedding = None

    # -------- TRACK BETWEEN DETECTIONS --------
    elif tracking:

        ok, box = tracker.update(frame)

        if ok:

            x, y, w, h = map(int, box)
            track_box = (x, y, w, h)

        else:

            tracking = False
            track_box = None
            last_embedding = None

    # -------- VERIFY USING CACHED EMBEDDING --------
    if (
        not paused
        and tracking
        and last_embedding is not None
        and SAVED_EMBEDDINGS is not None
        and now - last_verify >= VERIFY_INTERVAL
    ):

        last_verify = now

        current = last_embedding
        current_norm = np.linalg.norm(current)

        dots = SAVED_EMBEDDINGS @ current
        dists = 1 - dots / (SAVED_NORMS * current_norm)

        best = float(np.min(dists))

        is_match = best < MATCH_THRESHOLD

        history.append(is_match)

        verified_count = sum(history)

        if verified_count >= REQUIRED_MATCHES:

            last_verified = now
            locked_once = False

            status_text = f'VERIFIED {best:.3f}'
            status_color = (0, 255, 0)

        elif verified_count >= 1:

            status_text = f'UNCERTAIN {best:.3f}'
            status_color = (0, 255, 255)

        else:

            status_text = f'NOT VERIFIED {best:.3f}'
            status_color = (0, 0, 255)

    # -------- NO FACE --------
    if not tracking:

        history.append(False)

        status_text = 'NO FACE'
        status_color = (0, 0, 255)

    # -------- LOCK TIMER --------
    elapsed = now - last_verified
    remaining = max(0.0, LOCK_TIMEOUT - elapsed)

    if paused:

        lock_text = 'PAUSED'
        lock_color = (255, 255, 0)

    elif elapsed >= LOCK_TIMEOUT:

        lock_text = 'LOCKING...'
        lock_color = (0, 0, 255)

        if ENABLE_LOCK and not locked_once:
            locked_once = True
            ctypes.windll.user32.LockWorkStation()

    else:

        lock_text = f'LOCK IN {remaining:.1f}s'

        if remaining <= 2:
            lock_color = (0, 0, 255)
        elif remaining <= 5:
            lock_color = (0, 165, 255)
        else:
            lock_color = (0, 255, 255)

    # -------- DRAW UI --------
    if track_box is not None:

        x, y, w, h = track_box

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    cv2.putText(frame, status_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

    cv2.putText(frame, lock_text, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, lock_color, 2)

    cv2.putText(frame, f'Votes: {sum(history)}/{HISTORY_SIZE}',
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(frame, 'P: Pause  Q: Quit',
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow('ArcFace V1.8', frame)

    key = cv2.waitKey(1) & 0xFF

    # -------- PAUSE --------
    if key == ord('p'):

        paused = not paused

        if paused:

            status_text = 'PAUSED'
            status_color = (255, 255, 0)

            print('Monitoring paused')

        else:

            history.clear()
            last_verified = time.monotonic()
            locked_once = False

            status_text = 'RESUMED'
            status_color = (0, 255, 0)

            print('Monitoring resumed')

    # -------- QUIT --------
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()