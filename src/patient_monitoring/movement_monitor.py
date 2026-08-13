import cv2
import os
import math

cascade_path = os.path.join(
    "assets",
    "models",
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_path)

camera = cv2.VideoCapture(0)

previous_position = None

# Movement threshold
MOVEMENT_THRESHOLD = 80

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

    status = "NO PATIENT"

    if len(faces) > 0:

        x, y, w, h = faces[0]

        center_x = x + w // 2
        center_y = y + h // 2

        current_position = (center_x, center_y)

        movement = 0

        if previous_position is not None:

            movement = math.sqrt(
                (center_x - previous_position[0]) ** 2 +
                (center_y - previous_position[1]) ** 2
            )
            print("Movement:", int(movement))  
        if movement > MOVEMENT_THRESHOLD:
            status = "ALERT: SUDDEN MOVEMENT"
        else:
            status = "NORMAL MOVEMENT"

        previous_position = current_position

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Patient 1",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Movement: {int(movement)}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    else:

        previous_position = None
        status = "NO PATIENT"

    cv2.putText(
        frame,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.imshow(
        "MedGuardian AI - Movement Monitoring",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()  