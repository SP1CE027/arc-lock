import cv2
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis

EMBEDDING_FILE = Path('data/yash_embeddings.npy')
MATCH_THRESHOLD = 0.38  # lower = stricter

# Initialize ArcFace
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Could not open webcam')
    exit()

print('Press E to enroll (20 samples)')
print('Press V to verify')
print('Press Q to quit')

status_text = 'Ready'


def cosine_distance(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return 1 - np.dot(a, b)


while True:
    ret, frame = cap.read()

    if not ret:
        break

    faces = app.get(frame)

    # Draw face boxes
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw status text
    cv2.putText(
        frame,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow('ArcFace V1.2', frame)

    key = cv2.waitKey(1) & 0xFF

    # ---------------- ENROLL ----------------
    if key == ord('e'):

        embeddings = []

        print('Capturing 20 samples...')
        status_text = 'Capturing samples...'

        while len(embeddings) < 20:

            ret, frame = cap.read()
            if not ret:
                continue

            faces = app.get(frame)

            if len(faces) == 1:

                face = faces[0]

                # Reject tiny faces
                x1, y1, x2, y2 = face.bbox.astype(int)
                face_area = (x2 - x1) * (y2 - y1)

                if face_area > 12000:
                    embeddings.append(face.embedding)
                    print(f'Sample {len(embeddings)}/20')
                    status_text = f'Sample {len(embeddings)}/20'

            # Show capture preview
            preview = frame.copy()
            cv2.putText(
                preview,
                status_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow('ArcFace V1.2', preview)
            cv2.waitKey(50)

        np.save(EMBEDDING_FILE, np.array(embeddings))

        status_text = 'Enrollment complete'
        print('Enrollment complete')

    # ---------------- VERIFY ----------------
    elif key == ord('v'):

        if not EMBEDDING_FILE.exists():

            status_text = 'No enrollment found'
            print(status_text)

        elif len(faces) != 1:

            status_text = 'Need exactly 1 face'
            print(status_text)

        else:

            saved = np.load(EMBEDDING_FILE)  # (20, 512)
            current = faces[0].embedding

            dists = [cosine_distance(e, current) for e in saved]

            best = min(dists)
            avg_top5 = np.mean(sorted(dists)[:5])

            if best < MATCH_THRESHOLD:
                status_text = f'VERIFIED {best:.3f}'
            else:
                status_text = f'NOT VERIFIED {best:.3f}'

            print(f'{status_text} | AvgTop5: {avg_top5:.3f}')

    # ---------------- QUIT ----------------
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()