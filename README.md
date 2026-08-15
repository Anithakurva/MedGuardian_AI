# MedGuardian AI

## Intelligent Multi-Patient Monitoring and Early Warning System

MedGuardian AI is an AI-based patient monitoring system designed to continuously monitor patients using computer vision and real-time monitoring techniques.

The system detects patients through a camera, tracks their identity, monitors movement and inactivity, calculates a risk level, generates alerts, and stores monitoring data for analysis.

> **Disclaimer:** MedGuardian AI is a monitoring and early-warning system. It does not diagnose diseases and does not replace medical professionals.

---

## Problem Statement

In hospitals and healthcare environments, continuously monitoring multiple patients can be difficult for medical staff.

A patient may experience unusual movement or prolonged inactivity between manual observations.

MedGuardian AI aims to provide an automated monitoring system that can detect such changes and generate early warning alerts.

---

## Objectives

- Detect patients using computer vision.
- Track patients and maintain stable patient IDs.
- Monitor patient movement.
- Calculate patient inactivity duration.
- Classify patient risk levels.
- Generate warning and critical alerts.
- Store monitoring information in CSV format.
- Provide a visual monitoring dashboard.
- Analyze risk, movement, and inactivity history.

---

## Key Features

### 1. Face Detection

The system uses OpenCV-based face detection to identify patients from the camera feed.

### 2. Patient Tracking

Detected patients are assigned patient IDs and tracked across frames.

### 3. Movement Monitoring

The system calculates patient movement based on changes in detected face position.

### 4. Inactivity Monitoring

The system measures how long a patient remains inactive.

### 5. Risk Classification

Patients are classified into three risk levels:

- **LOW RISK** – Normal patient activity.
- **WARNING** – Increased inactivity requiring monitoring.
- **HIGH RISK** – Condition requiring immediate attention according to the configured monitoring rules.

### 6. Alert System

The system generates warning and critical alerts based on the calculated risk level.

### 7. CSV Monitoring Logs

Patient monitoring records are stored in:

```text
outputs/patient_monitoring_log.csv     


























MedGuardian_AI/
│
├── assets/
│   ├── patient.jpg
│   └── models/
│       └── haarcascade_frontalface_default.xml
│
├── docs/
│   ├── Architecture.md
│   ├── Dataset_Research.md
│   ├── Literature_Survey.md
│   ├── Objectives.md
│   ├── Problem_Statement.md
│   ├── Project_Roadmap.md
│   ├── Project_Vision.md
│   ├── Scope.md
│   ├── Technology_Selection.md
│   └── Workflow.md
│
├── outputs/
│   └── patient_monitoring_log.csv
│
├── research/
│   ├── 01_Problem_Research.md
│   ├── 02_Existing_Solutions.md
│   ├── 03_Datasets.md
│   ├── 04_Model_Ideas.md
│   └── 05_Project_Questions.md
│
├── src/
│   ├── dashboard/
│   ├── patient_detection/
│   ├── patient_monitoring/
│   └── patient_tracking/
│
├── PROJECT_JOURNAL.md
├── requirements.txt
└── README.md  



### 8. Web Monitoring Dashboard

MedGuardian AI provides a Streamlit-based web dashboard for visualizing patient monitoring data.

The dashboard displays:

- Total patients monitored
- Total monitoring records
- Warning and high-risk counts
- Patient-wise monitoring status
- Risk distribution
- Risk history
- Patient movement history
- Patient inactivity history
- Recent monitoring records
- Refresh monitoring data option   



## How to Run

### Install Dependencies

```bash
pip install -r requirements.txt  


### Run the Monitoring System

```bash
python src/dashboard/multi_patient_risk_monitor.py



streamlit run app.py



