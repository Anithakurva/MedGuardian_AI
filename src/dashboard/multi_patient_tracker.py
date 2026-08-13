import cv2
import os
import math


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

MAX_MISSING_FRAMES = 60
MATCH_DISTANCE = 600


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
    # SINGLE PATIENT MODE
    # ==========================================

    if len(faces) == 1:

        x, y, w, h = faces[0]

        center_x = x + w // 2
        center_y = y + h // 2


        # If a patient already exists,
        # keep the same ID regardless of movement.

        if len(patients) > 0:

            patient_id = min(patients.keys())

        else:

            patient_id = 1
            next_patient_id = 2


        patients[patient_id] = {
            "position": (center_x, center_y),
            "missing": 0
        }

        current_ids.add(patient_id)


        # Face box

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        # Patient ID

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
    # MULTIPLE PATIENT MODE
    # ==========================================

    elif len(faces) > 1:

        for (x, y, w, h) in faces:

            center_x = x + w // 2
            center_y = y + h // 2

            best_id = None
            best_distance = float("inf")


            # Find closest existing patient

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


            # Existing patient

            if (
                best_id is not None
                and best_distance < MATCH_DISTANCE
            ):

                patient_id = best_id


            # New patient

            else:

                patient_id = next_patient_id
                next_patient_id += 1

                patients[patient_id] = {
                    "position": (
                        center_x,
                        center_y
                    ),
                    "missing": 0
                }


            # Update patient

            patients[patient_id]["position"] = (
                center_x,
                center_y
            )

            patients[patient_id]["missing"] = 0

            current_ids.add(patient_id)


            # Face box

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )


            # Patient ID

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
    # NO FACE DETECTED
    # ==========================================

    else:

        # Keep existing patients temporarily

        for patient_id in patients:

            patients[patient_id]["missing"] += 1


    # ==========================================
    # UPDATE MISSING PATIENTS
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
    # DASHBOARD
    # ==========================================

    cv2.rectangle(
        frame,
        (10, 10),
        (390, 75),
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
        (25, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    # ==========================================
    # SHOW
    # ==========================================

    cv2.imshow(
        "MedGuardian AI - Multi Patient Tracker",
        frame
    )


    # ==========================================
    # EXIT
    # ==========================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows() 