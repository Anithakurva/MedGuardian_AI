import cv2
import os
import math
import time

from risk_monitor import calculate_risk, get_risk_message
from alert_system import trigger_alert
from patient_monitoring_log import log_patient_data


# ==========================================
# SETTINGS
# ==========================================

LOG_INTERVAL = 5
last_log_time = 0

MAX_MISSING_FRAMES = 60
MATCH_DISTANCE = 600


# ==========================================
# FACE DETECTION
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
# PATIENT DATA
# ==========================================

next_patient_id = 1
patients = {}


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

    current_ids = set()


    # ==========================================
    # SINGLE PATIENT
    # ==========================================

    if len(faces) == 1:

        x, y, w, h = faces[0]

        center_x = x + w // 2
        center_y = y + h // 2


        # ------------------------------------------
        # PATIENT ID
        # ------------------------------------------

        if len(patients) > 0:

            patient_id = min(patients.keys())

        else:

            patient_id = 1
            next_patient_id = 2

            patients[patient_id] = {
                "position": (center_x, center_y),
                "last_movement_time": time.time(),
                "missing": 0
            }


        # ------------------------------------------
        # MOVEMENT
        # ------------------------------------------

        old_position = patients[patient_id]["position"]

        movement = math.sqrt(
            (center_x - old_position[0]) ** 2 +
            (center_y - old_position[1]) ** 2
        )


        if movement > 5:

            patients[patient_id]["last_movement_time"] = time.time()


        # ------------------------------------------
        # INACTIVITY
        # ------------------------------------------

        inactivity_time = (
            time.time()
            - patients[patient_id]["last_movement_time"]
        )


        # ------------------------------------------
        # RISK
        # ------------------------------------------

        risk_level = calculate_risk(
            movement,
            inactivity_time
        )

        risk_message = get_risk_message(
            risk_level
        )


        # ------------------------------------------
        # CSV LOGGING - EVERY 5 SECONDS
        # ------------------------------------------

        current_time = time.time()

        if current_time - last_log_time >= LOG_INTERVAL:

            log_patient_data(
                f"Patient {patient_id}",
                movement,
                inactivity_time,
                risk_level,
                risk_message
            )

            last_log_time = current_time


        # ------------------------------------------
        # ALERT
        # ------------------------------------------

        trigger_alert(
            f"Patient {patient_id}",
            risk_level,
            risk_message
        )


        # ------------------------------------------
        # UPDATE PATIENT
        # ------------------------------------------

        patients[patient_id]["position"] = (
            center_x,
            center_y
        )

        patients[patient_id]["missing"] = 0

        current_ids.add(patient_id)


        # ------------------------------------------
        # FACE BOX
        # ------------------------------------------

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Patient {patient_id}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        # ------------------------------------------
        # MOVEMENT
        # ------------------------------------------

        cv2.putText(
            frame,
            f"Movement: {int(movement)}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        # ------------------------------------------
        # INACTIVITY
        # ------------------------------------------

        cv2.putText(
            frame,
            f"Inactivity: {int(inactivity_time)} sec",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        # ------------------------------------------
        # RISK
        # ------------------------------------------

        cv2.putText(
            frame,
            f"Risk: {risk_level}",
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )


        # ------------------------------------------
        # MESSAGE
        # ------------------------------------------

        cv2.putText(
            frame,
            risk_message,
            (20, 205),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )


    # ==========================================
    # MULTIPLE PATIENTS
    # ==========================================

    elif len(faces) > 1:

        for (x, y, w, h) in faces:

            center_x = x + w // 2
            center_y = y + h // 2

            best_id = None
            best_distance = float("inf")


            # --------------------------------------
            # FIND CLOSEST PATIENT
            # --------------------------------------

            for patient_id, data in patients.items():

                if patient_id in current_ids:
                    continue

                old_x, old_y = data["position"]

                distance = math.sqrt(
                    (center_x - old_x) ** 2 +
                    (center_y - old_y) ** 2
                )

                if distance < best_distance:

                    best_distance = distance
                    best_id = patient_id


            # --------------------------------------
            # EXISTING PATIENT
            # --------------------------------------

            if (
                best_id is not None
                and best_distance < MATCH_DISTANCE
            ):

                patient_id = best_id


            # --------------------------------------
            # NEW PATIENT
            # --------------------------------------

            else:

                patient_id = next_patient_id
                next_patient_id += 1

                patients[patient_id] = {
                    "position": (
                        center_x,
                        center_y
                    ),
                    "last_movement_time": time.time(),
                    "missing": 0
                }


            # --------------------------------------
            # MOVEMENT
            # --------------------------------------

            old_x, old_y = patients[patient_id]["position"]

            movement = math.sqrt(
                (center_x - old_x) ** 2 +
                (center_y - old_y) ** 2
            )


            if movement > 5:

                patients[patient_id]["last_movement_time"] = time.time()


            # --------------------------------------
            # INACTIVITY
            # --------------------------------------

            inactivity_time = (
                time.time()
                - patients[patient_id]["last_movement_time"]
            )


            # --------------------------------------
            # RISK
            # --------------------------------------

            risk_level = calculate_risk(
                movement,
                inactivity_time
            )

            risk_message = get_risk_message(
                risk_level
            )


            # --------------------------------------
            # ALERT
            # --------------------------------------

            trigger_alert(
                f"Patient {patient_id}",
                risk_level,
                risk_message
            )


            # --------------------------------------
            # UPDATE
            # --------------------------------------

            patients[patient_id]["position"] = (
                center_x,
                center_y
            )

            patients[patient_id]["missing"] = 0

            current_ids.add(patient_id)


            # --------------------------------------
            # DRAW
            # --------------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Patient {patient_id}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


    # ==========================================
    # NO FACE
    # ==========================================

    else:

        pass


    # ==========================================
    # REMOVE LOST PATIENTS
    # ==========================================

    for patient_id in list(patients.keys()):

        if patient_id not in current_ids:

            patients[patient_id]["missing"] += 1

            if (
                patients[patient_id]["missing"]
                > MAX_MISSING_FRAMES
            ):

                del patients[patient_id]


    # ==========================================
    # HEADER
    # ==========================================

    cv2.putText(
        frame,
        "MEDGUARDIAN AI - MULTI PATIENT RISK MONITOR",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ==========================================
    # PATIENT COUNT
    # ==========================================

    cv2.putText(
        frame,
        f"Patients Detected: {len(faces)}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ==========================================
    # DISPLAY
    # ==========================================

    cv2.imshow(
        "MedGuardian AI - Multi Patient Risk Monitor",
        frame
    )


    # ==========================================
    # EXIT
    # ==========================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows() 