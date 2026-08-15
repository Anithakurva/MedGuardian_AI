import streamlit as st
import pandas as pd
import os
st.button("🔄 Refresh Monitoring Data")
# ==========================================
# MEDGUARDIAN AI - PROFESSIONAL WEB DASHBOARD
# ==========================================

LOG_FILE = os.path.join(
    "outputs",
    "patient_monitoring_log.csv"
)

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="MedGuardian AI",
    page_icon="🏥",
    layout="wide"
)

# ==========================================
# TITLE
# ==========================================

st.title("🏥 MedGuardian AI")
st.caption(
    "Intelligent Patient Monitoring and Early Warning System"
)
if st.button("🔄 Refresh Monitoring Data"):
    st.rerun()
st.divider()

# ==========================================
# READ DATA
# ==========================================

if not os.path.exists(LOG_FILE):

    st.error("❌ Monitoring log file not found.")

    st.stop()

data = pd.read_csv(LOG_FILE)

if data.empty:

    st.warning("No monitoring data available.")

    st.stop()

# ==========================================
# DATA PREPARATION
# ==========================================

data["Movement"] = pd.to_numeric(
    data["Movement"],
    errors="coerce"
).fillna(0)

data["Inactivity (sec)"] = pd.to_numeric(
    data["Inactivity (sec)"],
    errors="coerce"
).fillna(0)

data["Timestamp"] = pd.to_datetime(
    data["Timestamp"],
    errors="coerce"
)

# ==========================================
# SUMMARY
# ==========================================

total_patients = data["Patient ID"].nunique()

total_records = len(data)

low_count = (
    data["Risk Level"] == "LOW RISK"
).sum()

warning_count = (
    data["Risk Level"] == "WARNING"
).sum()

high_count = (
    data["Risk Level"] == "HIGH RISK"
).sum()

# ==========================================
# TOP METRICS
# ==========================================

st.subheader("📊 Monitoring Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "👤 Patients Monitored",
    total_patients
)

col2.metric(
    "📝 Total Records",
    total_records
)

col3.metric(
    "⚠️ Warnings",
    warning_count
)

col4.metric(
    "🚨 High Risk",
    high_count
)

st.divider()

# ==========================================
# LATEST PATIENT STATUS
# ==========================================
# ==========================================
# PATIENT-WISE MONITORING
# ==========================================

st.subheader("👥 Patient-wise Monitoring")

patient_ids = data["Patient ID"].unique()

for patient_id in patient_ids:

    patient_data = data[
        data["Patient ID"] == patient_id
    ]

    latest_patient = patient_data.iloc[-1]

    patient_risk = latest_patient["Risk Level"]

    if patient_risk == "LOW RISK":
        risk_display = "🟢 LOW RISK"

    elif patient_risk == "WARNING":
        risk_display = "🟡 WARNING"

    else:
        risk_display = "🔴 HIGH RISK"

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Patient",
        patient_id
    )

    col2.metric(
        "Movement",
        f'{latest_patient["Movement"]:.2f}'
    )

    col3.metric(
        "Inactivity",
        f'{latest_patient["Inactivity (sec)"]:.2f} sec'
    )

    col4.write(
        f"**Risk Level:** {risk_display}"
    )

    st.caption(
        f'**Status:** {latest_patient["Message"]}'
    )
latest = data.iloc[-1]

st.subheader("🩺 Latest Patient Status")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Patient",
    latest["Patient ID"]
)

col2.metric(
    "Movement",
    f'{latest["Movement"]:.2f}'
)

col3.metric(
    "Inactivity",
    f'{latest["Inactivity (sec)"]:.2f} sec'
)

risk = latest["Risk Level"]

if risk == "LOW RISK":
    col4.success(f"🟢 {risk}")

elif risk == "WARNING":
    col4.warning(f"🟡 {risk}")

else:
    col4.error(f"🔴 {risk}")

st.info(
    f'**Message:** {latest["Message"]}'
)

st.caption(
    f'Last Updated: {latest["Timestamp"]}'
)

st.divider()

# ==========================================
# RISK DISTRIBUTION
# ==========================================

st.subheader("📊 Risk Distribution")

risk_data = pd.DataFrame(
    {
        "Risk Level": [
            "LOW RISK",
            "WARNING",
            "HIGH RISK"
        ],
        "Records": [
            low_count,
            warning_count,
            high_count
        ]
    }
)

st.bar_chart(
    risk_data.set_index("Risk Level")
)

st.divider()

# ==========================================
# RISK HISTORY
# ==========================================

st.subheader("📈 Risk History")

risk_mapping = {
    "LOW RISK": 1,
    "WARNING": 2,
    "HIGH RISK": 3
}

risk_history = data["Risk Level"].map(
    risk_mapping
)

risk_chart = pd.DataFrame(
    {
        "Risk Level": risk_history
    }
)

st.line_chart(risk_chart)

st.caption(
    "1 = LOW RISK | 2 = WARNING | 3 = HIGH RISK"
)

st.divider()

# ==========================================
# MOVEMENT HISTORY
# ==========================================

st.subheader("🚶 Patient Movement History")

movement_chart = data[
    ["Timestamp", "Movement"]
].copy()

movement_chart = movement_chart.dropna(
    subset=["Timestamp"]
)

movement_chart = movement_chart.set_index(
    "Timestamp"
)

st.line_chart(
    movement_chart["Movement"]
)

st.divider()

# ==========================================
# INACTIVITY HISTORY
# ==========================================

st.subheader("⏱️ Patient Inactivity History")

inactivity_chart = data[
    ["Timestamp", "Inactivity (sec)"]
].copy()

inactivity_chart = inactivity_chart.dropna(
    subset=["Timestamp"]
)

inactivity_chart = inactivity_chart.set_index(
    "Timestamp"
)

st.line_chart(
    inactivity_chart["Inactivity (sec)"]
)

st.divider()

# ==========================================
# RECENT RECORDS
# ==========================================

st.subheader("📋 Recent Monitoring Records")

recent_data = data.tail(20).copy()

st.dataframe(
    recent_data,
    use_container_width=True,
    hide_index=True
)

st.divider()

# ==========================================
# PROJECT DISCLAIMER
# ==========================================

st.caption(
    "MedGuardian AI is an academic monitoring prototype. "
    "It does not diagnose diseases or replace medical professionals."
)     