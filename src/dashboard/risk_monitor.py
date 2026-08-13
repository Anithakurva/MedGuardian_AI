import time

# ==========================================
# MEDGUARDIAN AI - RISK MONITOR
# ==========================================

MOVEMENT_THRESHOLD = 20
WARNING_INACTIVITY = 10
HIGH_RISK_INACTIVITY = 20


def calculate_risk(movement, inactivity_time):
    """
    Calculate patient risk level based on
    movement and inactivity.
    """

    # Sudden/high movement
    if movement >= MOVEMENT_THRESHOLD:
        return "HIGH RISK"

    # Long inactivity
    elif inactivity_time >= HIGH_RISK_INACTIVITY:
        return "HIGH RISK"

    # Moderate inactivity
    elif inactivity_time >= WARNING_INACTIVITY:
        return "WARNING"

    # Normal condition
    else:
        return "LOW RISK"


def get_risk_message(risk_level):

    if risk_level == "HIGH RISK":
        return "Immediate attention required!"

    elif risk_level == "WARNING":
        return "Please monitor the patient."

    else:
        return "Patient movement is normal."


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    print("================================")
    print("   MEDGUARDIAN AI RISK MONITOR")
    print("================================")

    test_cases = [
        (5, 2),
        (10, 12),
        (25, 3),
        (5, 22)
    ]

    for movement, inactivity in test_cases:

        risk = calculate_risk(
            movement,
            inactivity
        )

        message = get_risk_message(risk)

        print()
        print(f"Movement: {movement}")
        print(f"Inactivity: {inactivity} sec")
        print(f"Risk: {risk}")
        print(f"Message: {message}")  