import cv2
import os
import math

# Haar Cascade model
cascade_path = os.path.join(
    "assets",
    "models",
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_path)

camera = cv2.VideoCapture(0)

# Patient information
patients = {}
next_patient_id = 1

# Tracking settings
MAX_DISTANCE = 250
MAX_MISSED_FRAMES = 20


def calculate_distance(x1, y1, x2, y2):
    return math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


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

    detected_faces = []

    # Store detected face information
    for (x, y, w, h) in faces:

        center_x = x + w // 2
        center_y = y + h // 2

        detected_faces.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "center_x": center_x,
            "center_y": center_y
        })

    matched_patients = set()

    # Match detected faces with existing patients
    for face in detected_faces:

        center_x = face["center_x"]
        center_y = face["center_y"]

        best_patient_id = None
        best_distance = float("inf")

        for patient_id, patient in patients.items():

            if patient_id in matched_patients:
                continue

            distance = calculate_distance(
                center_x,
                center_y,
                patient["center_x"],
                patient["center_y"]
            )

            if distance < best_distance:
                best_distance = distance
                best_patient_id = patient_id

        # Existing patient
        if (
            best_patient_id is not None
            and best_distance < MAX_DISTANCE
        ):

            patient_id = best_patient_id

            patients[patient_id]["center_x"] = center_x
            patients[patient_id]["center_y"] = center_y
            patients[patient_id]["w"] = face["w"]
            patients[patient_id]["h"] = face["h"]
            patients[patient_id]["missed"] = 0

        # New patient
        else:

            patient_id = next_patient_id
            next_patient_id += 1

            patients[patient_id] = {
                "center_x": center_x,
                "center_y": center_y,
                "w": face["w"],
                "h": face["h"],
                "missed": 0
            }

        matched_patients.add(patient_id)

        # Draw patient box
        x = face["x"]
        y = face["y"]
        w = face["w"]
        h = face["h"]

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

    # Handle patients temporarily not detected
    for patient_id in list(patients.keys()):

        if patient_id not in matched_patients:

            patients[patient_id]["missed"] += 1

            if patients[patient_id]["missed"] > MAX_MISSED_FRAMES:
                del patients[patient_id]

    # Display current detected count
    cv2.putText(
        frame,
        f"Patients Detected: {len(faces)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.imshow(
        "MedGuardian AI - Stable Multi Patient Tracking",
        frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()  