import cv2
import os
import math


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
# PATIENT MEMORY
# ==========================================

patients = {}

next_patient_id = 1

MAX_DISTANCE = 350
MAX_MISSED_FRAMES = 30


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

    detected_faces = []

    for (x, y, w, h) in faces:

        center_x = x + w // 2
        center_y = y + h // 2

        detected_faces.append(
            (center_x, center_y, x, y, w, h)
        )


    matched_patients = set()


    # ==========================================
    # SINGLE PATIENT MODE
    # ==========================================
    # If only one face is detected, always keep
    # the same Patient 1 ID even during movement.

    if len(detected_faces) == 1:

        center_x, center_y, x, y, w, h = detected_faces[0]

        patient_id = 1

        patients[patient_id] = {
            "x": center_x,
            "y": center_y,
            "missed": 0
        }

        matched_patients.add(patient_id)

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


    # ==========================================
    # MULTI-PATIENT MODE
    # ==========================================

    elif len(detected_faces) > 1:

        for (
            center_x,
            center_y,
            x,
            y,
            w,
            h
        ) in detected_faces:

            best_patient = None
            best_distance = float("inf")


            for patient_id, data in patients.items():

                if patient_id in matched_patients:
                    continue

                old_x = data["x"]
                old_y = data["y"]

                distance = math.sqrt(
                    (center_x - old_x) ** 2 +
                    (center_y - old_y) ** 2
                )

                if distance < best_distance:

                    best_distance = distance
                    best_patient = patient_id


            if (
                best_patient is not None
                and best_distance < MAX_DISTANCE
            ):

                patient_id = best_patient

                patients[patient_id]["x"] = center_x
                patients[patient_id]["y"] = center_y
                patients[patient_id]["missed"] = 0

            else:

                patient_id = next_patient_id
                next_patient_id += 1

                patients[patient_id] = {
                    "x": center_x,
                    "y": center_y,
                    "missed": 0
                }


            matched_patients.add(patient_id)


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
    # UPDATE MISSED PATIENTS
    # ==========================================

    for patient_id in list(patients.keys()):

        if patient_id not in matched_patients:

            patients[patient_id]["missed"] += 1

            if (
                patients[patient_id]["missed"]
                > MAX_MISSED_FRAMES
            ):
                del patients[patient_id]


    # ==========================================
    # PATIENT COUNT
    # ==========================================

    cv2.putText(
        frame,
        f"Patients Detected: {len(faces)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )


    # ==========================================
    # DISPLAY
    # ==========================================

    cv2.imshow(
        "MedGuardian AI - Multiple Patient Tracking",
        frame
    )


    # ==========================================
    # EXIT
    # ==========================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows() 