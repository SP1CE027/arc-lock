import cv2
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis

app = FaceAnalysis(
    name='buffalo_l',
    allowed_modules=['detection', 'recognition']
)

app.prepare(ctx_id=0, det_size=(320, 320))

cap = cv2.VideoCapture(0)

embeddings = []

print('Look at the camera. Press SPACE to capture, Q to finish.')

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow('Enroll', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        faces = app.get(frame)

        if len(faces) == 1:
            embeddings.append(faces[0].embedding.astype(np.float32))
            print(f'Captured {len(embeddings)}')
        else:
            print('Need exactly one face')

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

Path('data').mkdir(exist_ok=True)

np.save('data/yash_embeddings.npy', np.array(embeddings, dtype=np.float32))

print(f'Saved {len(embeddings)} embeddings')