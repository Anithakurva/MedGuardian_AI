import csv
import os
from collections import Counter

import matplotlib.pyplot as plt


# ==========================================
# MEDGUARDIAN AI
# PROFESSIONAL MONITORING DASHBOARD
# ==========================================

LOG_FILE = os.path.join(
    "outputs",
    "patient_monitoring_log.csv"
)


# ==========================================
# READ MONITORING DATA
# ==========================================

def read_monitoring_data():

    if not os.path.exists(LOG_FILE):

        print("Monitoring log file not found!")

        return []

    with open(
        LOG_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        return list(reader)


# ==========================================
# CREATE DASHBOARD
# ==========================================

def create_dashboard():

    data = read_monitoring_data()

    if not data:

        print("No monitoring data available.")

        return


    # ==========================================
    # RISK COUNTS
    # ==========================================

    risk_levels = [
        row["Risk Level"]
        for row in data
    ]

    risk_count = Counter(risk_levels)

    low_count = risk_count.get(
        "LOW RISK",
        0
    )

    warning_count = risk_count.get(
        "WARNING",
        0
    )

    high_count = risk_count.get(
        "HIGH RISK",
        0
    )


    # ==========================================
    # PATIENT INFORMATION
    # ==========================================

    patients = set(
        row["Patient ID"]
        for row in data
    )

    total_patients = len(patients)


    # ==========================================
    # LATEST RECORD
    # ==========================================

    latest = data[-1]

    latest_patient = latest["Patient ID"]
    latest_movement = latest["Movement"]
    latest_inactivity = latest["Inactivity (sec)"]
    latest_risk = latest["Risk Level"]
    latest_message = latest["Message"]
    latest_timestamp = latest["Timestamp"]


    # ==========================================
    # TERMINAL SUMMARY
    # ==========================================

    print()
    print("=" * 60)
    print("              MEDGUARDIAN AI")
    print("        PATIENT MONITORING DASHBOARD")
    print("=" * 60)

    print()

    print(f"Total Patients Monitored : {total_patients}")

    print()

    print("-----------------------------------------------")
    print("RISK SUMMARY")
    print("-----------------------------------------------")

    print(f"LOW RISK                  : {low_count}")
    print(f"WARNING                   : {warning_count}")
    print(f"HIGH RISK                 : {high_count}")

    print()

    print("-----------------------------------------------")
    print("LATEST PATIENT STATUS")
    print("-----------------------------------------------")

    print(f"Patient ID                : {latest_patient}")
    print(f"Movement                  : {latest_movement}")
    print(f"Inactivity                : {latest_inactivity} sec")
    print(f"Risk Level                : {latest_risk}")
    print(f"Message                   : {latest_message}")
    print(f"Timestamp                 : {latest_timestamp}")

    print()

    print("=" * 60)


    # ==========================================
    # GRAPH 1 - RISK DISTRIBUTION
    # ==========================================

    labels = [
        "LOW RISK",
        "WARNING",
        "HIGH RISK"
    ]

    values = [
        low_count,
        warning_count,
        high_count
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        labels,
        values
    )

    plt.title(
        "MedGuardian AI - Risk Distribution"
    )

    plt.xlabel(
        "Risk Level"
    )

    plt.ylabel(
        "Number of Records"
    )

    plt.tight_layout()


    # ==========================================
    # GRAPH 2 - RISK HISTORY
    # ==========================================

    risk_values = []

    for row in data:

        if row["Risk Level"] == "LOW RISK":

            risk_values.append(1)

        elif row["Risk Level"] == "WARNING":

            risk_values.append(2)

        elif row["Risk Level"] == "HIGH RISK":

            risk_values.append(3)


    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        risk_values
    )

    plt.title(
        "MedGuardian AI - Risk History"
    )

    plt.xlabel(
        "Monitoring Record"
    )

    plt.ylabel(
        "Risk Level"
    )

    plt.yticks(
        [1, 2, 3],
        [
            "LOW RISK",
            "WARNING",
            "HIGH RISK"
        ]
    )

    plt.grid()

    plt.tight_layout()


    # ==========================================
    # GRAPH 3 - MOVEMENT HISTORY
    # ==========================================

    movement_values = []

    for row in data:

        try:

            movement_values.append(
                float(row["Movement"])
            )

        except ValueError:

            movement_values.append(0)


    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        movement_values
    )

    plt.title(
        "MedGuardian AI - Patient Movement History"
    )

    plt.xlabel(
        "Monitoring Record"
    )

    plt.ylabel(
        "Movement"
    )

    plt.grid()

    plt.tight_layout()


    # ==========================================
    # GRAPH 4 - INACTIVITY HISTORY
    # ==========================================

    inactivity_values = []

    for row in data:

        try:

            inactivity_values.append(
                float(row["Inactivity (sec)"])
            )

        except ValueError:

            inactivity_values.append(0)


    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        inactivity_values
    )

    plt.title(
        "MedGuardian AI - Patient Inactivity History"
    )

    plt.xlabel(
        "Monitoring Record"
    )

    plt.ylabel(
        "Inactivity (seconds)"
    )

    plt.grid()

    plt.tight_layout()


    # ==========================================
    # SHOW DASHBOARD GRAPHS
    # ==========================================

    plt.show()


# ==========================================
# RUN DASHBOARD
# ==========================================

if __name__ == "__main__":

    create_dashboard()