import cv2
import os
import time

# Haar Cascade model
cascade_path = os.path.join(
    "assets",
    "models",
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_path)

camera = cv2.VideoCapture(0)

# Previous patient position
previous_position = None

# Time when movement was last detected
last_movement_time = time.time()

# Warning after 10 seconds of very little movement
INACTIVITY_LIMIT = 10


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

        # Monitor the first detected patient
        x, y, w, h = faces[0]

        center_x = x + w // 2
        center_y = y + h // 2

        current_position = (center_x, center_y)

        # Check movement
        if previous_position is not None:

            movement = (
                abs(current_position[0] - previous_position[0]) +
                abs(current_position[1] - previous_position[1])
            )

            if movement > 5:
                last_movement_time = time.time()

        previous_position = current_position

        # Calculate inactivity time
        inactivity_time = time.time() - last_movement_time

        # Status
        if inactivity_time >= INACTIVITY_LIMIT:
            status = "WARNING: PATIENT INACTIVE"
        else:
            status = "PATIENT ACTIVE"

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

        # Display inactivity time
        cv2.putText(
            frame,
            f"Inactivity: {int(inactivity_time)} sec",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    else:

        previous_position = None
        status = "NO PATIENT"

    # Display monitoring status
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
        "MedGuardian AI - Inactivity Monitoring",
        frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()  