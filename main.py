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
LOCK_TIMEOUT = 15.0      # Use 15s for testing, change back to 7.0 later
ENABLE_LOCK = True       # Set to False if you want dry-run mode

# Temporal voting
HISTORY_SIZE = 5
REQUIRED_MATCHES = 3

history = deque(maxlen=HISTORY_SIZE)

# Initialize ArcFace
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Could not open webcam')
    exit()

print('Press E to enroll (20 samples)')
print('Press P to pause/resume')
print('Press Q to quit')

status_text = 'Ready'
status_color = (0, 255, 0)

last_verify = 0.0
last_verified = time.monotonic()

paused = False
locked_once = False


def cosine_distance(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return 1 - np.dot(a, b)


while True:

    ret, frame = cap.read()

    # Camera recovery
    if not ret:
        print('Camera lost, attempting recovery...')
        cap.release()
        time.sleep(1)
        cap = cv2.VideoCapture(0)
        continue

    faces = app.get(frame)

    # Draw face boxes
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    now = time.monotonic()

    # ---------------- AUTO VERIFY ----------------
    if not paused and now - last_verify >= VERIFY_INTERVAL:

        last_verify = now

        if EMBEDDING_FILE.exists() and len(faces) == 1:

            saved = np.load(EMBEDDING_FILE)
            current = faces[0].embedding

            dists = [cosine_distance(e, current) for e in saved]

            best = min(dists)
            avg_top5 = np.mean(sorted(dists)[:5])

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

            print(
                f'{status_text} | '
                f'Votes: {verified_count}/{HISTORY_SIZE} | '
                f'AvgTop5: {avg_top5:.3f}'
            )

        else:
            history.append(False)
            status_text = 'NO FACE'
            status_color = (0, 0, 255)

    # ---------------- ABSENCE TIMER ----------------
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

    # ---------------- DRAW UI ----------------
    cv2.putText(
        frame,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        status_color,
        2
    )

    cv2.putText(
        frame,
        lock_text,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        lock_color,
        2
    )

    cv2.putText(
        frame,
        f'Votes: {sum(history)}/{HISTORY_SIZE}',
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        'P: Pause  E: Enroll  Q: Quit',
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1
    )

    cv2.imshow('ArcFace V1.5', frame)

    key = cv2.waitKey(1) & 0xFF

    # ---------------- ENROLL ----------------
    if key == ord('e'):

        embeddings = []

        print('Capturing 20 samples...')
        status_text = 'Capturing samples...'
        status_color = (0, 255, 255)

        while len(embeddings) < 20:

            ret, frame = cap.read()

            if not ret:
                continue

            faces = app.get(frame)

            if len(faces) == 1:

                face = faces[0]

                x1, y1, x2, y2 = face.bbox.astype(int)
                face_area = (x2 - x1) * (y2 - y1)

                if face_area > 12000:

                    embeddings.append(face.embedding)

                    print(f'Sample {len(embeddings)}/20')

                    status_text = f'Sample {len(embeddings)}/20'

            preview = frame.copy()

            cv2.putText(
                preview,
                status_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

            cv2.imshow('ArcFace V1.5', preview)

            cv2.waitKey(50)

        np.save(EMBEDDING_FILE, np.array(embeddings))

        history.clear()
        last_verified = time.monotonic()
        locked_once = False

        status_text = 'Enrollment complete'
        status_color = (0, 255, 0)

        print('Enrollment complete')

    # ---------------- PAUSE ----------------
    elif key == ord('p'):

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

    # ---------------- QUIT ----------------
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()