import cv2
import os
import math
import time

# -----------------------------
# Face detection model
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

patient_id = 1

previous_position = None

last_seen_time = time.time()

last_movement_time = time.time()

# Face can temporarily disappear
MAX_NO_FACE_TIME = 3

# Movement threshold
MOVEMENT_THRESHOLD = 20


while True:

    ret, frame = camera.read()

    if not ret:
        print("Camera not detected!")
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    movement = 0

    # -----------------------------
    # Patient detected
    # -----------------------------

    if len(faces) > 0:

        # Use the largest detected face
        face = max(
            faces,
            key=lambda item: item[2] * item[3]
        )

        x, y, w, h = face

        center_x = x + w // 2
        center_y = y + h // 2

        current_position = (
            center_x,
            center_y
        )

        # -----------------------------
        # Calculate movement
        # -----------------------------

        if previous_position is not None:

            old_x, old_y = previous_position

            movement = math.sqrt(
                (center_x - old_x) ** 2 +
                (center_y - old_y) ** 2
            )

            if movement > 5:
                last_movement_time = time.time()

        previous_position = current_position

        # Update last seen time
        last_seen_time = time.time()

        # -----------------------------
        # Risk level
        # -----------------------------

        inactivity_time = (
            time.time() - last_movement_time
        )

        if movement > MOVEMENT_THRESHOLD:

            risk_level = "HIGH RISK"

        elif inactivity_time >= 10:

            risk_level = "WARNING"

        else:

            risk_level = "LOW RISK"

        # -----------------------------
        # Draw patient
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

        # -----------------------------
        # Patient information
        # -----------------------------

        cv2.putText(
            frame,
            f"Movement: {int(movement)}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Inactivity: {int(inactivity_time)} sec",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Risk: {risk_level}",
            (20, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Patient Status: ACTIVE",
            (20, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

    else:

        # -----------------------------
        # Temporary face loss
        # -----------------------------

        time_without_face = (
            time.time() - last_seen_time
        )

        if time_without_face < MAX_NO_FACE_TIME:

            cv2.putText(
                frame,
                "Patient 1 - TRACKING...",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "Patient 1 - NO PATIENT",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

    # -----------------------------
    # Dashboard title
    # -----------------------------

    cv2.putText(
        frame,
        "MEDGUARDIAN AI - STABLE PATIENT TRACKING",
        (20, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # -----------------------------
    # Display
    # -----------------------------

    cv2.imshow(
        "MedGuardian AI - Stable Single Patient",
        frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()      