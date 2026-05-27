from flask import Flask, render_template, request, jsonify, send_file, Response
import os
import csv
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import io

app = Flask(__name__)

# ==================== CONFIG ====================
ATTENDANCE_CSV = os.path.join('attendance', 'attendance.csv')
NAMES_CSV = os.path.join('attendance', 'names.csv')
KNOWN_FACES_DIR = os.path.join('static', 'known_faces')

os.makedirs('attendance', exist_ok=True)
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs('static/images', exist_ok=True)

# ==================== INIT ====================
def init_csv():
    if not os.path.exists(ATTENDANCE_CSV):
        with open(ATTENDANCE_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'name', 'timestamp', 'date', 'time', 'status'])

init_csv()

# ==================== FACE DETECTOR ====================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
    return faces

# ==================== HELPERS ====================
def now():
    dt = datetime.now()
    return {
        "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "date": dt.strftime("%Y-%m-%d"),
        "time": dt.strftime("%H:%M:%S")
    }

# ==================== ROUTES ====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/attendance')
def attendance():
    return render_template('attendance.html')

# ===================================================
# 🔥 PROCESS FRAME (REPLACES cv2.VideoCapture)
# ===================================================
@app.route('/api/process-frame', methods=['POST'])
def process_frame():
    file = request.files['image']

    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    faces = detect_faces(frame)

    return jsonify({
        "faces_detected": len(faces)
    })

# ===================================================
# MARK ATTENDANCE
# ===================================================
@app.route('/api/mark-attendance', methods=['POST'])
def mark_attendance():
    data = request.json
    user_id = data.get("id")
    name = data.get("name")

    if not user_id or not name:
        return jsonify({"error": "Missing data"}), 400

    t = now()

    # check duplicates per day
    if os.path.exists(ATTENDANCE_CSV):
        df = pd.read_csv(ATTENDANCE_CSV)
        today = t["date"]
        if len(df[(df["id"] == user_id) & (df["date"] == today)]) > 0:
            return jsonify({"message": "Already marked today"}), 200

    with open(ATTENDANCE_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([user_id, name, t["timestamp"], t["date"], t["time"], "Present"])

    return jsonify({
        "success": True,
        "message": "Attendance marked",
        "time": t["time"]
    })

# ===================================================
# REGISTER USER (NO CAMERA ON SERVER)
# ===================================================
@app.route('/api/register', methods=['POST'])
def register_user():
    user_id = request.form.get("id")
    name = request.form.get("name")

    if not user_id or not name:
        return jsonify({"error": "Missing data"}), 400

    # duplicate check
    if os.path.exists(NAMES_CSV):
        df = pd.read_csv(NAMES_CSV)
        if user_id in df["id"].astype(str).values:
            return jsonify({"error": "User already exists"}), 400

    with open(NAMES_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([user_id, name])

    return jsonify({
        "success": True,
        "message": "User registered successfully"
    })
