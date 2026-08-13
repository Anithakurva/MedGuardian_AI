import csv
import os
from datetime import datetime


# ==========================================
# MEDGUARDIAN AI - PATIENT MONITORING LOG
# ==========================================

LOG_FILE = os.path.join(
    "outputs",
    "patient_monitoring_log.csv"
)


def initialize_log():

    os.makedirs("outputs", exist_ok=True)

    if not os.path.exists(LOG_FILE):

        with open(
            LOG_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "Timestamp",
                "Patient ID",
                "Movement",
                "Inactivity (sec)",
                "Risk Level",
                "Message"
            ])


def log_patient_data(
    patient_id,
    movement,
    inactivity_time,
    risk_level,
    message
):

    initialize_log()

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with open(
        LOG_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            timestamp,
            patient_id,
            round(movement, 2),
            round(inactivity_time, 2),
            risk_level,
            message
        ])


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    log_patient_data(
        "Patient 1",
        5,
        3,
        "LOW RISK",
        "Patient movement is normal."
    )

    log_patient_data(
        "Patient 1",
        28,
        1,
        "HIGH RISK",
        "Immediate attention required!"
    )

    print("Patient monitoring data logged successfully.")
    print(f"Log file: {LOG_FILE}")  