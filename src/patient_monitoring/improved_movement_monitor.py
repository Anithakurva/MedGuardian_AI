import cv2
import os
import math
import time

cascade_path = os.path.join(
    "assets",
    "models",
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_path)

camera = cv2.VideoCapture(0)

previous_position = None

# How much movement is considered significant
MOVEMENT_THRESHOLD = 50

# How long face can be temporarily missing
NO_FACE_LIMIT = 3

last_seen_time = time.time()

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

    # Patient detected
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

        # Update last seen time
        last_seen_time = time.time()

        # Movement status
        if movement > MOVEMENT_THRESHOLD:
            status = "POSSIBLE ABNORMAL MOVEMENT"
        else:
            status = "NORMAL MOVEMENT"

        previous_position = current_position

        # Draw patient box
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

        # Don't immediately say NO PATIENT
        time_since_seen = time.time() - last_seen_time

        if time_since_seen < NO_FACE_LIMIT:
            status = "TRACKING PATIENT..."
        else:
            status = "NO PATIENT"
            previous_position = None

    # Display status
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
        "MedGuardian AI - Improved Movement Monitoring",
        frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows() 