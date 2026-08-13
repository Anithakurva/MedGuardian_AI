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
# PATIENT TRACKING SETTINGS
# ==========================================

next_patient_id = 1

patients = {}

MATCH_DISTANCE = 180
MAX_MISSING_FRAMES = 15

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

    current_patient_ids = set()
    current_positions = {}

    # ==========================================
    # DETECT FACES
    # ==========================================

    detected_faces = []

    for (x, y, w, h) in faces:

        center_x = x + w // 2
        center_y = y + h // 2

        detected_faces.append({
            "box": (x, y, w, h),
            "center": (center_x, center_y)
        })

    # ==========================================
    # MATCH FACES WITH EXISTING PATIENTS
    # ==========================================

    matches = []

    for face_index, face in enumerate(detected_faces):

        center_x, center_y = face["center"]

        best_patient = None
        best_distance = float("inf")

        for patient_id, patient in patients.items():

            if patient_id in current_patient_ids:
                continue

            old_x, old_y = patient["position"]

            distance = math.sqrt(
                (center_x - old_x) ** 2 +
                (center_y - old_y) ** 2
            )

            if distance < best_distance:

                best_distance = distance
                best_patient = patient_id

        if (
            best_patient is not None
            and best_distance < MATCH_DISTANCE
        ):

            matches.append(
                (
                    face_index,
                    best_patient,
                    best_distance
                )
            )

    # ==========================================
    # ACCEPT MATCHES
    # ==========================================

    matched_faces = set()

    for face_index, patient_id, distance in matches:

        if patient_id in current_patient_ids:
            continue

        matched_faces.add(face_index)
        current_patient_ids.add(patient_id)

    # ==========================================
    # PROCESS EACH FACE
    # ==========================================

    for face_index, face in enumerate(detected_faces):

        x, y, w, h = face["box"]

        center_x, center_y = face["center"]

        # ------------------------------------------
        # Find matched patient
        # ------------------------------------------

        patient_id = None

        for match_face, match_id, distance in matches:

            if match_face == face_index:

                if match_id not in current_patient_ids:
                    continue

                patient_id = match_id
                break

        # ------------------------------------------
        # Create new patient
        # ------------------------------------------

        if patient_id is None:

            patient_id = next_patient_id
            next_patient_id += 1

            patients[patient_id] = {
                "position": (center_x, center_y),
                "previous_position": (
                    center_x,
                    center_y
                ),
                "last_movement": time.time(),
                "missing": 0
            }

            current_patient_ids.add(patient_id)

        patient = patients[patient_id]

        # ------------------------------------------
        # Calculate movement
        # ------------------------------------------

        old_x, old_y = patient["previous_position"]

        movement = math.sqrt(
            (center_x - old_x) ** 2 +
            (center_y - old_y) ** 2
        )

        # ------------------------------------------
        # Update movement time
        # ------------------------------------------

        if movement > 5:

            patient["last_movement"] = time.time()

        inactivity_time = (
            time.time()
            - patient["last_movement"]
        )

        # ------------------------------------------
        # Risk level
        # ------------------------------------------

        if movement > MOVEMENT_THRESHOLD:

            risk_level = "HIGH RISK"

        elif inactivity_time >= INACTIVITY_LIMIT:

            risk_level = "WARNING"

        else:

            risk_level = "LOW RISK"

        # ------------------------------------------
        # Update patient information
        # ------------------------------------------

        patient["previous_position"] = (
            center_x,
            center_y
        )

        patient["position"] = (
            center_x,
            center_y
        )

        patient["missing"] = 0

        current_positions[patient_id] = (
            center_x,
            center_y
        )

        # ==========================================
        # DRAW PATIENT
        # ==========================================

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
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Risk: {risk_level}",
            (x, y + h + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

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
    # HANDLE MISSING PATIENTS
    # ==========================================

    for patient_id in list(patients.keys()):

        if patient_id not in current_patient_ids:

            patients[patient_id]["missing"] += 1

    # ==========================================
    # REMOVE LONG-MISSING PATIENTS
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
        (400, 80),
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
    # SHOW WINDOW
    # ==========================================

    cv2.imshow(
        "MedGuardian AI - Stable Multi Patient",
        frame
    )

    # ==========================================
    # EXIT
    # ==========================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows() 