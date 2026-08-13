import cv2
import os
import math
import time

# -----------------------------
# Load Face Detection Model
# -----------------------------

cascade_path = os.path.join(
    "assets",
    "models",
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_path)

# -----------------------------
# Camera
# -----------------------------

camera = cv2.VideoCapture(0)

# -----------------------------
# Tracking variables
# -----------------------------

previous_position = None
last_movement_time = time.time()

INACTIVITY_LIMIT = 10
MOVEMENT_THRESHOLD = 20


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

    # Default values
    status = "NO PATIENT"
    risk_level = "NO PATIENT"
    movement = 0
    inactivity_time = 0

    # -----------------------------
    # Patient detected
    # -----------------------------

    if len(faces) > 0:

        x, y, w, h = faces[0]

        center_x = x + w // 2
        center_y = y + h // 2

        current_position = (center_x, center_y)

        # Calculate movement
        if previous_position is not None:

            movement = math.sqrt(
                (center_x - previous_position[0]) ** 2 +
                (center_y - previous_position[1]) ** 2
            )

            if movement > 5:
                last_movement_time = time.time()

        previous_position = current_position

        inactivity_time = time.time() - last_movement_time

        status = "PATIENT ACTIVE"

        # -----------------------------
        # Risk assessment
        # -----------------------------

        if movement > MOVEMENT_THRESHOLD:

            risk_level = "HIGH RISK"

        elif inactivity_time >= INACTIVITY_LIMIT:

            risk_level = "WARNING"

        else:

            risk_level = "LOW RISK"

        # -----------------------------
        # Patient box
        # -----------------------------

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

    else:

        previous_position = None

    # -----------------------------
    # Dashboard Panel
    # -----------------------------

    cv2.rectangle(
        frame,
        (10, 10),
        (350, 190),
        (50, 50, 50),
        -1
    )

    cv2.putText(
        frame,
        "MEDGUARDIAN AI",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Status: {status}",
        (25, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Movement: {int(movement)}",
        (25, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Inactivity: {int(inactivity_time)} sec",
        (25, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Risk: {risk_level}",
        (25, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # -----------------------------
    # Show Dashboard
    # -----------------------------

    cv2.imshow(
        "MedGuardian AI - Patient Dashboard",
        frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows() 