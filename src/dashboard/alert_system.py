import time

ALERT_COOLDOWN = 10 
last_alert_time = 0


def trigger_alert(patient_id, risk_level, message):
    global last_alert_time

    current_time = time.time()

    # Prevent repeated alerts every frame
    if current_time - last_alert_time < ALERT_COOLDOWN:
        return

    if risk_level == "HIGH RISK":

        print()
        print("========================================")
        print("🚨 MEDGUARDIAN AI - CRITICAL ALERT 🚨")
        print("========================================")
        print(f"Patient: {patient_id}")
        print(f"Risk Level: {risk_level}")
        print(f"Alert: {message}")
        print("Immediate attention required!")
        print("========================================")

        last_alert_time = current_time

    elif risk_level == "WARNING":

        print()
        print("========================================")
        print("⚠️ MEDGUARDIAN AI - WARNING")
        print("========================================")
        print(f"Patient: {patient_id}")
        print(f"Risk Level: {risk_level}")
        print(f"Message: {message}")
        print("========================================")

        last_alert_time = current_time        