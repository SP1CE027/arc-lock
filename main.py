import cv2
from insightface.app import FaceAnalysis

# Initialize ArcFace
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Could not open webcam')
    exit()

print('Press Q to quit')

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect faces
    faces = app.get(frame)

    # Draw boxes
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(
        frame,
        f'Faces: {len(faces)}',
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow('ArcFace V1', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()