import cv2
import os

# Haar Cascade model
cascade_path = os.path.join(
    "assets",
    "models",
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_path)

camera = cv2.VideoCapture(0)

# Patient IDs
next_patient_id = 1
patient_ids = {}

while True:

    ret, frame = camera.read()

    if not ret:
        print("Camera not detected!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    # Detect faces
    for face_index, (x, y, w, h) in enumerate(faces):

        # Temporary identity based on face position
        center_x = x + w // 2
        center_y = y + h // 2

        if face_index not in patient_ids:
            patient_ids[face_index] = next_patient_id
            next_patient_id += 1

        patient_id = patient_ids[face_index]

        # Draw face box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Display patient ID
        cv2.putText(
            frame,
            f"Patient {patient_id}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        f"Patients Detected: {len(faces)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.imshow(
        "MedGuardian AI - Identity Tracking",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows() 