# Real-Time Crime Detection and Prevention from CCTV Footage

## Project Overview

This project implements an intelligent **real-time crime detection and prevention system** using CCTV camera footage. It automates video surveillance, detecting suspicious or criminal activities and alerting users or emergency services. The system combines **computer vision, deep learning, and a GUI-based video monitoring interface**.

Key components include:

1. **Frame Generation & Motion Detection** – Extract frames with significant motion using OpenCV.
2. **Deep Learning Model** – Uses SlowFast networks for anomaly and crime detection.
3. **Graphical User Interface (GUI)** – Displays live CCTV footage, triggers alerts, and allows user feedback.
4. **Web Application** – Enables video recording, playback, and saving directly from the browser.

This project was developed as a **Capstone Project** for the B.Tech Computer Science & Engineering program at **PES University**, leveraging the **UCF-Crime dataset**.

---

## Key Features

### 1️⃣ Motion Detection and Frame Processing

* Detects motion in CCTV streams using frame differencing and background subtraction.
* Generates and sharpens frames with OpenCV for better feature extraction.
* Creates frame strips for model input to capture temporal features.

### 2️⃣ Deep Learning Crime Detection

* **SlowFast Network** for video anomaly detection.
* Detects activities like robbery, fighting, vandalism, arson, shoplifting, and more.
* Triggers alerts for detected crimes and enables user verification.

### 3️⃣ Web and GUI Integration

* **GUI Application (Tkinter)** for real-time monitoring and alerts.
* **Web App (HTML/CSS/JS)** for video recording, saving, and playback in-browser.
* Allows users to **report false positives** to improve model retraining.

### 4️⃣ Dataset and Model

* Uses **UCF-Crime Dataset** (\~127 hours of surveillance videos).
* Covers 13 anomalies: Abuse, Burglary, Robbery, Stealing, Shooting, Shoplifting, Assault, Fighting, Arson, Explosion, Road Accident, Vandalism, and Normal Events.
* Implements frame-stripping and motion detection for preprocessing.

---

## System Architecture

```
CCTV Camera / Video Feed
           │
           ▼
    Frame Generation (processInput.py)
           │
           ▼
    Motion Detection & Frame Sharpening (OpenCV)
           │
           ▼
    Frame Strip Generation → SlowFast Deep Learning Model (learning_model.py / model.ipynb)
           │
           ▼
    Detection Result → GUI (gui_final.py) / WebApp (webapp.html + index.js + index.css)
           │
           ├── User Alert / Alarm
           └── Optional Retraining
```

---

## Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/realtime-crime-detection.git
cd realtime-crime-detection
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### Example `requirements.txt`

```
numpy
opencv-python
imutils
pandas
matplotlib
scikit-learn
tensorflow / pytorch  # Choose based on model implementation
```

**Optional for WebApp:**

```
flask
selenium
```

---

## Usage

### 1. Real-Time Detection (GUI)

```bash
python gui_final.py
```

* Opens a GUI for monitoring CCTV feed.
* Alerts users if suspicious activity is detected.
* Users can report false alarms to improve the model.

### 2. WebApp Recording and Playback

Open **webapp.html** in a browser:

* Start/Stop recording with buttons.
* Save recorded media to local storage.
* Preview live webcam feed with playback support.

### 3. Model Training and Evaluation

Use `model.ipynb` to:

* Train the SlowFast network on preprocessed frames.
* Evaluate model performance using the UCF-Crime dataset.
* Save and load trained weights for real-time inference.

---

## Insights & Results

* **Automated Surveillance**: Reduces human supervision dependency.
* **Real-Time Detection**: Alerts generated within seconds of suspicious activity.
* **Dataset Coverage**: Detects 13 anomalies with high accuracy.
* **Extensible System**: Supports GUI and web-based video monitoring.

---

## Project Files

* **processInput.py** – Frame generation and motion detection.
* **framegen.py** – Frame strip generation for model input.
* **learning\_model.py** – Model implementation.
* **model.ipynb** – Model training notebook.
* **gui\_final.py** – GUI application for live monitoring.
* **webapp.html / index.js / index.css** – Web interface for recording and playback.
* **report.docx** – Detailed project report with literature survey and methodology.

---

## Contact

* **Author**: Mohit Sai Gutha
* **[Email](mailto:your.email@example.com)**
* **[LinkedIn](https://www.linkedin.com/in/mohitsaigutha)**

© 2025 Mohit Sai Gutha | Real-Time Crime Detection System