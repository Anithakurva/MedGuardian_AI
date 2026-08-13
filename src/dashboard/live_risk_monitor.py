import cv2
import os
import math
import time

from alert_system import trigger_alert
from risk_monitor import calculate_risk, get_risk_message 
from patient_monitoring_log import log_patient_data  


# ==========================================
# FACE DETECTION MODEL
# ==========================================

cascade_path = os.path.join(
    "assets",
    "models",
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_path)


# ==========================================
# CAMERA
# ==========================================

camera = cv2.VideoCapture(0)


# ==========================================
# PATIENT
# ==========================================

patient_id = 1

previous_position = None

last_seen_time = time.time()

last_movement_time = time.time()


# ==========================================
# MAIN LOOP
# ==========================================

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


    # ==========================================
    # PATIENT DETECTED
    # ==========================================

    if len(faces) > 0:

        # Select largest face
        x, y, w, h = max(
            faces,
            key=lambda face: face[2] * face[3]
        )

        center_x = x + w // 2
        center_y = y + h // 2

        current_position = (
            center_x,
            center_y
        )


        # ==========================================
        # MOVEMENT
        # ==========================================

        movement = 0

        if previous_position is not None:

            old_x, old_y = previous_position

            movement = math.sqrt(
                (center_x - old_x) ** 2 +
                (center_y - old_y) ** 2
            )

            if movement > 5:
                last_movement_time = time.time()


        previous_position = current_position

        last_seen_time = time.time()


        # ==========================================
        # INACTIVITY
        # ==========================================

        inactivity_time = (
            time.time() - last_movement_time
        )


        # ==========================================
        # RISK MONITOR
        # ==========================================

        risk_level = calculate_risk(
            movement,
            inactivity_time
        )

        risk_message = get_risk_message(
            risk_level
        )


        # ==========================================
        # ALERT SYSTEM
        # ==========================================

        trigger_alert(
            "Patient 1",
            risk_level,
            risk_message
        ) 
        log_patient_data(
            "Patient 1",
             movement,
             inactivity_time,
             risk_level,
             risk_message
) 


        # ==========================================
        # FACE BOX
        # ==========================================

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        # ==========================================
        # PATIENT ID
        # ==========================================

        cv2.putText(
            frame,
            "Patient 1",
            (x, y + h + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        # ==========================================
        # MOVEMENT
        # ==========================================

        cv2.putText(
            frame,
            f"Movement: {int(movement)}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        # ==========================================
        # INACTIVITY
        # ==========================================

        cv2.putText(
            frame,
            f"Inactivity: {int(inactivity_time)} sec",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        # ==========================================
        # RISK
        # ==========================================

        cv2.putText(
            frame,
            f"Risk: {risk_level}",
            (20, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )


        # ==========================================
        # MESSAGE
        # ==========================================

        cv2.putText(
            frame,
            risk_message,
            (20, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


    else:

        # ==========================================
        # NO FACE
        # ==========================================

        time_without_face = (
            time.time() - last_seen_time
        )

        cv2.putText(
            frame,
            "Patient 1 - TRACKING...",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )


    # ==========================================
    # HEADER
    # ==========================================

    cv2.putText(
        frame,
        "MEDGUARDIAN AI - LIVE RISK MONITOR",
        (20, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ==========================================
    # DISPLAY
    # ==========================================

    cv2.imshow(
        "MedGuardian AI - Live Risk Monitor",
        frame
    )


    # ==========================================
    # EXIT
    # ==========================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# RELEASE CAMERA
# ==========================================

camera.release()
cv2.destroyAllWindows()     