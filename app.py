from flask import Flask, render_template, send_file, redirect, url_for, Response, jsonify, request
import os
import csv
import cv2
import pandas as pd
from datetime import datetime
import json
import io
import threading
import time
import base64
import numpy as np

app = Flask(__name__)

# ==================== CONFIGURATION ====================
ATTENDANCE_CSV = os.path.join('attendance', 'attendance.csv')
NAMES_CSV = os.path.join('attendance', 'names.csv')
KNOWN_FACES_DIR = os.path.join('static', 'known_faces')

os.makedirs('attendance', exist_ok=True)
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs('static/images', exist_ok=True)

# Global variables
camera = None
camera_lock = threading.Lock()
face_cascade = None
marked_today_cache = set()
session_date = None

# Store face encodings for matching
face_encodings_cache = {}
face_names_cache = {}
face_ids_cache = {}

# ==================== HELPER FUNCTIONS ====================

def init_csv_files():
    """Initialize CSV files with consistent headers"""
    if not os.path.exists(ATTENDANCE_CSV):
        with open(ATTENDANCE_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'name', 'timestamp', 'date', 'time', 'status'])

def get_today_date():
    return datetime.now().strftime("%Y-%m-%d")

def get_current_time():
    return datetime.now().strftime("%H:%M:%S")

def get_full_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def refresh_marked_cache():
    global marked_today_cache, session_date
    today = get_today_date()
    
    if session_date != today:
        marked_today_cache = set()
        session_date = today
    
    if os.path.exists(ATTENDANCE_CSV):
        with open(ATTENDANCE_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('date') == today:
                    marked_today_cache.add(row.get('id'))

def is_already_marked_today(user_id):
    refresh_marked_cache()
    return str(user_id) in marked_today_cache

def save_attendance_record(user_id, user_name):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    date = now.strftime("%Y-%m-%d")
    time_only = now.strftime("%H:%M:%S")
    
    if is_already_marked_today(user_id):
        return False, "Already marked today"
    
    file_exists = os.path.exists(ATTENDANCE_CSV)
    with open(ATTENDANCE_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['id', 'name', 'timestamp', 'date', 'time', 'status'])
        writer.writerow([user_id, user_name, timestamp, date, time_only, 'Present'])
    
    marked_today_cache.add(str(user_id))
    print(f"✅ Attendance saved: {user_name} (ID: {user_id}) at {time_only}")
    return True, f"Attendance marked for {user_name}"

def init_face_detector():
    global face_cascade
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    print("Face detector initialized")
    return face_cascade

def detect_faces(frame):
    if face_cascade is None:
        init_face_detector()
    
    if face_cascade is None:
        return []
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
    
    converted_faces = []
    for (x, y, w, h) in faces:
        converted_faces.append((int(x), int(y), int(w), int(h)))
    
    return converted_faces

def load_face_encodings():
    """Load face images for matching"""
    global face_encodings_cache, face_names_cache, face_ids_cache
    
    face_encodings_cache = {}
    face_names_cache = {}
    face_ids_cache = {}
    
    # Load registered users
    users = []
    if os.path.exists(NAMES_CSV):
        with open(NAMES_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                users.append({'id': row['id'], 'name': row['name']})
                print(f"Found registered user: {row['name']} (ID: {row['id']})")
    
    # Load face images
    for user in users:
        face_path = os.path.join(KNOWN_FACES_DIR, f"{user['id']}.jpg")
        if os.path.exists(face_path):
            img = cv2.imread(face_path)
            if img is not None:
                face_encodings_cache[user['id']] = img
                face_names_cache[user['id']] = user['name']
                face_ids_cache[user['id']] = user['id']
                print(f"Loaded face image for {user['name']}")
            else:
                print(f"Failed to load face image for {user['name']}")
        else:
            print(f"Face image not found: {face_path}")
    
    print(f"Total loaded faces: {len(face_encodings_cache)}")

def advanced_face_compare(detected_face, registered_face):
    """Advanced face comparison using multiple methods"""
    try:
        # Resize to same dimensions
        detected_resized = cv2.resize(detected_face, (200, 200))
        registered_resized = cv2.resize(registered_face, (200, 200))
        
        # Convert to grayscale
        detected_gray = cv2.cvtColor(detected_resized, cv2.COLOR_BGR2GRAY)
        registered_gray = cv2.cvtColor(registered_resized, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Histogram comparison
        hist_detected = cv2.calcHist([detected_gray], [0], None, [256], [0, 256])
        hist_registered = cv2.calcHist([registered_gray], [0], None, [256], [0, 256])
        hist_score = cv2.compareHist(hist_detected, hist_registered, cv2.HISTCMP_CORREL)
        
        # Method 2: Template matching (using smaller region - eyes/nose area)
        h, w = detected_gray.shape
        center_y, center_x = h // 2, w // 2
        face_center = detected_gray[center_y-30:center_y+30, center_x-30:center_x+30]
        reg_center = registered_gray[center_y-30:center_y+30, center_x-30:center_x+30]
        
        if face_center.size > 0 and reg_center.size > 0:
            template_score = cv2.matchTemplate(face_center, reg_center, cv2.TM_CCOEFF_NORMED)[0][0]
        else:
            template_score = 0
        
        # Method 3: Mean squared error
        mse = np.mean((detected_gray.astype(float) - registered_gray.astype(float)) ** 2)
        mse_score = 1 - min(1, mse / 10000)  # Convert MSE to similarity score
        
        # Combined score (weighted average)
        combined_score = (hist_score * 0.5) + (float(template_score) * 0.3) + (mse_score * 0.2)
        
        return float(combined_score)
        
    except Exception as e:
        print(f"Error in advanced_face_compare: {e}")
        return 0.0

def find_best_match(detected_face):
    """Find the best matching registered face"""
    best_score = 0.0
    best_name = None
    best_id = None
    
    if len(face_encodings_cache) == 0:
        print("No registered faces in cache")
        return None, None, 0.0
    
    print(f"Comparing against {len(face_encodings_cache)} registered faces...")
    
    for user_id, registered_face in face_encodings_cache.items():
        try:
            score = advanced_face_compare(detected_face, registered_face)
            print(f"  Match with {face_names_cache.get(user_id)}: score = {score:.4f}")
            
            if score > best_score and score > 0.45:  # Threshold 0.45 (lowered for better matching)
                best_score = score
                best_name = face_names_cache.get(user_id)
                best_id = face_ids_cache.get(user_id)
        except Exception as e:
            print(f"Error comparing face for {user_id}: {e}")
    
    if best_name:
        print(f"✅ Best match: {best_name} (ID: {best_id}) with score {best_score:.4f}")
    else:
        print(f"❌ No match found. Best score was {best_score:.4f} (below threshold 0.45)")
    
    return best_name, best_id, best_score

# ==================== ROUTES ====================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/attendance')
def attendance_page():
    return render_template('attendance.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# ==================== API ENDPOINTS ====================

@app.route('/api/users')
def api_get_users():
    try:
        users = []
        if os.path.exists(NAMES_CSV):
            with open(NAMES_CSV, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    users.append({'id': row['id'], 'name': row['name']})
        print(f"Returning {len(users)} registered users")
        return jsonify({'users': users})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def api_get_stats():
    try:
        user_count = 0
        if os.path.exists(NAMES_CSV):
            with open(NAMES_CSV, 'r', encoding='utf-8') as f:
                user_count = sum(1 for _ in f) - 1
        
        attendance_count = 0
        today_count = 0
        today = get_today_date()
        
        if os.path.exists(ATTENDANCE_CSV):
            df = pd.read_csv(ATTENDANCE_CSV)
            attendance_count = len(df)
            if 'date' in df.columns:
                today_count = len(df[df['date'] == today])
        
        return jsonify({
            'total_students': max(0, user_count),
            'total_attendance': attendance_count,
            'today_attendance': today_count,
            'last_update': get_full_timestamp(),
            'today_date': today
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/register', methods=['POST'])
def api_register():
    try:
        user_id = request.form.get('id')
        user_name = request.form.get('name')
        image_data = request.form.get('image')
        
        if not user_id or not user_name:
            return jsonify({'error': 'ID and name required'}), 400
        
        if not image_data:
            return jsonify({'error': 'No face image provided'}), 400
        
        # Decode image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        # Check if user exists
        if os.path.exists(NAMES_CSV):
            with open(NAMES_CSV, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['id'] == user_id:
                        return jsonify({'error': 'User ID already exists'}), 400
                    if row['name'] == user_name:
                        return jsonify({'error': 'Username already exists'}), 400
        
        # Detect face
        faces = detect_faces(frame)
        if len(faces) == 0:
            return jsonify({'error': 'No face detected. Please position yourself clearly.'}), 400
        
        # Save face image
        x, y, w, h = faces[0]
        face_crop = frame[y:y+h, x:x+w]
        face_path = os.path.join(KNOWN_FACES_DIR, f'{user_id}.jpg')
        cv2.imwrite(face_path, face_crop)
        
        # Save to names CSV
        file_exists = os.path.exists(NAMES_CSV)
        with open(NAMES_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['id', 'name'])
            writer.writerow([user_id, user_name])
        
        # Reload face encodings
        load_face_encodings()
        
        return jsonify({'success': True, 'message': f'User {user_name} registered successfully'})
        
    except Exception as e:
        print(f"Registration error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recognize-face', methods=['POST'])
def recognize_face():
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'message': 'No image data'}), 400
        
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'success': False, 'message': 'Invalid image'}), 400
        
        faces = detect_faces(frame)
        
        if len(faces) == 0:
            return jsonify({'success': False, 'message': 'No face detected'})
        
        x, y, w, h = faces[0]
        detected_face = frame[y:y+h, x:x+w]
        
        if len(face_encodings_cache) == 0:
            return jsonify({'success': False, 'message': 'No registered users. Please register first.'})
        
        best_name, best_id, confidence = find_best_match(detected_face)
        
        if best_name is None:
            return jsonify({
                'success': False, 
                'message': 'Face not recognized. Please register first.'
            })
        
        confidence_pct = float(confidence * 100)
        success, message = save_attendance_record(best_id, best_name)
        
        return jsonify({
            'success': success,
            'message': message,
            'name': best_name,
            'id': best_id,
            'confidence': round(confidence_pct, 2)
        })
        
    except Exception as e:
        print(f"Recognition error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/detect-face', methods=['POST'])
def detect_face():
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False}), 400
        
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'success': False}), 400
        
        faces = detect_faces(frame)
        
        if len(faces) == 0:
            return jsonify({'success': True, 'face_detected': False})
        
        x, y, w, h = faces[0]
        detected_face = frame[y:y+h, x:x+w]
        best_name, best_id, confidence = find_best_match(detected_face)
        
        confidence_val = float(confidence * 100) if confidence else 0.0
        
        response_data = {
            'success': True,
            'face_detected': True,
            'x': int(x),
            'y': int(y),
            'w': int(w),
            'h': int(h),
            'confidence': round(confidence_val, 2)
        }
        
        if best_name:
            response_data['name'] = str(best_name)
            response_data['id'] = str(best_id)
        else:
            response_data['name'] = None
            response_data['id'] = None
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Detection error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/export/excel')
def export_excel():
    try:
        if os.path.exists(ATTENDANCE_CSV):
            df = pd.read_csv(ATTENDANCE_CSV)
        else:
            df = pd.DataFrame(columns=['id', 'name', 'timestamp', 'date', 'time', 'status'])
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Attendance', index=False)
        
        output.seek(0)
        filename = f'attendance_{get_today_date()}.xlsx'
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/export/csv')
def export_csv():
    try:
        if os.path.exists(ATTENDANCE_CSV):
            return send_file(
                ATTENDANCE_CSV,
                mimetype='text/csv',
                as_attachment=True,
                download_name=f'attendance_{get_today_date()}.csv'
            )
        else:
            return jsonify({'error': 'No data'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/export/json')
def export_json():
    try:
        if os.path.exists(ATTENDANCE_CSV):
            df = pd.read_csv(ATTENDANCE_CSV)
            data = df.to_dict('records')
        else:
            data = []
        
        return jsonify({
            'export_date': get_full_timestamp(),
            'today_date': get_today_date(),
            'total_records': len(data),
            'data': data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== VIDEO STREAM ====================

def get_camera():
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            for i in range(3):
                camera = cv2.VideoCapture(i)
                if camera.isOpened():
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    break
            else:
                camera = None
    return camera

def release_camera():
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None

@app.route('/video')
def video_feed():
    def generate():
        cap = get_camera()
        if cap is None:
            return
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            faces = detect_faces(frame)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Face", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ==================== INITIALIZATION ====================

def initialize_system():
    init_csv_files()
    init_face_detector()
    load_face_encodings()
    refresh_marked_cache()
    
    print("=" * 60)
    print("🎓 FACE RECOGNITION ATTENDANCE SYSTEM")
    print("=" * 60)
    print(f"📍 Access: http://localhost:5000")
    print(f"📅 Today's date: {get_today_date()}")
    print(f"👥 Registered users: {len(face_encodings_cache)}")
    print("=" * 60)

# ==================== CLEANUP ====================
import atexit
@atexit.register
def cleanup():
    release_camera()

# ==================== MAIN ====================
if __name__ == '__main__':
    initialize_system()
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)