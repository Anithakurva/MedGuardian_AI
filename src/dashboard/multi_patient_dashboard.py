import cv2
import os
import math
import time

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
# PATIENT TRACKING
# ==========================================

next_patient_id = 1

# Stores active patients
patients = {}

# Number of frames a patient can be missing
MAX_MISSING_FRAMES = 20

# Distance used when multiple patients exist
MATCH_DISTANCE = 300

# ==========================================
# RISK SETTINGS
# ==========================================

MOVEMENT_THRESHOLD = 20
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

    current_patient_ids = set()

    # ==========================================
    # DETECTED FACES
    # ==========================================

    for (x, y, w, h) in faces:

        center_x = x + w // 2
        center_y = y + h // 2

        current_position = (center_x, center_y)

        matched_id = None
        minimum_distance = float("inf")

        # ==========================================
        # IF ONLY ONE PATIENT IS ACTIVE
        # KEEP SAME ID
        # ==========================================

        active_patients = [
            patient_id
            for patient_id, data in patients.items()
            if data["missing"] <= MAX_MISSING_FRAMES
        ]

        if len(faces) == 1 and len(active_patients) == 1:

            matched_id = active_patients[0]

        else:

            # ==========================================
            # MULTI-PATIENT MATCHING
            # ==========================================

            for patient_id in active_patients:

                if patient_id in current_patient_ids:
                    continue

                old_x, old_y = patients[patient_id]["position"]

                distance = math.sqrt(
                    (center_x - old_x) ** 2 +
                    (center_y - old_y) ** 2
                )

                if distance < minimum_distance:

                    minimum_distance = distance
                    matched_id = patient_id

            # Only accept nearest patient if reasonably close
            if (
                matched_id is not None
                and minimum_distance > MATCH_DISTANCE
            ):
                matched_id = None

        # ==========================================
        # CREATE NEW PATIENT
        # ==========================================

        if matched_id is None:

            patient_id = next_patient_id
            next_patient_id += 1

            patients[patient_id] = {
                "position": current_position,
                "previous_position": current_position,
                "missing": 0,
                "last_movement": time.time()
            }

        else:

            patient_id = matched_id

        current_patient_ids.add(patient_id)

        patient = patients[patient_id]

        # ==========================================
        # MOVEMENT CALCULATION
        # ==========================================

        old_x, old_y = patient["previous_position"]

        movement = math.sqrt(
            (center_x - old_x) ** 2 +
            (center_y - old_y) ** 2
        )

        # ==========================================
        # UPDATE MOVEMENT TIME
        # ==========================================

        if movement > 5:

            patient["last_movement"] = time.time()

        inactivity_time = (
            time.time() - patient["last_movement"]
        )

        # ==========================================
        # RISK LEVEL
        # ==========================================

        if movement > MOVEMENT_THRESHOLD:

            risk_level = "HIGH RISK"

        elif inactivity_time >= INACTIVITY_LIMIT:

            risk_level = "WARNING"

        else:

            risk_level = "LOW RISK"

        # ==========================================
        # UPDATE PATIENT DATA
        # ==========================================

        patient["previous_position"] = current_position
        patient["position"] = current_position
        patient["missing"] = 0

        # ==========================================
        # DRAW FACE BOX
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
            f"Patient {patient_id}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        # ==========================================
        # RISK
        # ==========================================

        cv2.putText(
            frame,
            f"Risk: {risk_level}",
            (x, y + h + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        # ==========================================
        # MOVEMENT
        # ==========================================

        cv2.putText(
            frame,
            f"Movement: {int(movement)}",
            (x, y + h + 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    # ==========================================
    # HANDLE PATIENTS NOT DETECTED THIS FRAME
    # ==========================================

    for patient_id in list(patients.keys()):

        if patient_id not in current_patient_ids:

            patients[patient_id]["missing"] += 1

    # ==========================================
    # REMOVE PATIENT ONLY AFTER MANY MISSED FRAMES
    # ==========================================

    for patient_id in list(patients.keys()):

        if patients[patient_id]["missing"] > MAX_MISSING_FRAMES:

            del patients[patient_id]

    # ==========================================
    # DASHBOARD HEADER
    # ==========================================

    cv2.rectangle(
        frame,
        (10, 10),
        (390, 80),
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
        f"Patients Detected: {len(faces)}",
        (25, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    # ==========================================
    # SHOW DASHBOARD
    # ==========================================

    cv2.imshow(
        "MedGuardian AI - Multi Patient Dashboard",
        frame
    )

    # ==========================================
    # PRESS Q TO EXIT
    # ==========================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows() 