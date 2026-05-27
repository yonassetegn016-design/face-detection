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

# ==================== HELPER FUNCTIONS ====================

def init_csv_files():
    """Initialize CSV files with consistent headers"""
    if not os.path.exists(ATTENDANCE_CSV):
        with open(ATTENDANCE_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'name', 'timestamp', 'date', 'time', 'status'])

def get_today_date():
    """Get today's date as string"""
    return datetime.now().strftime("%Y-%m-%d")

def get_current_time():
    """Get current time as string"""
    return datetime.now().strftime("%H:%M:%S")

def get_full_timestamp():
    """Get full timestamp"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def refresh_marked_cache():
    """Refresh the in-memory cache of who was marked today"""
    global marked_today_cache, session_date
    today = get_today_date()
    
    # Reset cache if date changed
    if session_date != today:
        marked_today_cache = set()
        session_date = today
    
    # Load existing records from today
    if os.path.exists(ATTENDANCE_CSV):
        with open(ATTENDANCE_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('date') == today:
                    marked_today_cache.add(row.get('id'))

def is_already_marked_today(user_id):
    """Check if user already marked today"""
    refresh_marked_cache()
    return str(user_id) in marked_today_cache

def save_attendance_record(user_id, user_name):
    """Save attendance record with proper date and time"""
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    date = now.strftime("%Y-%m-%d")
    time_only = now.strftime("%H:%M:%S")
    
    # Check if already marked today
    if is_already_marked_today(user_id):
        return False, "Already marked today"
    
    # Save to CSV
    file_exists = os.path.exists(ATTENDANCE_CSV)
    with open(ATTENDANCE_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['id', 'name', 'timestamp', 'date', 'time', 'status'])
        writer.writerow([user_id, user_name, timestamp, date, time_only, 'Present'])
    
    # Update cache
    marked_today_cache.add(str(user_id))
    
    print(f"✅ Attendance saved: {user_name} (ID: {user_id}) at {time_only} on {date}")
    return True, "Attendance marked successfully"

def init_face_detector():
    """Initialize face cascade classifier"""
    global face_cascade
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    return face_cascade

def get_camera():
    """Get camera instance with proper error handling"""
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            # Try different camera indices
            for i in range(3):
                camera = cv2.VideoCapture(i)
                if camera.isOpened():
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    print(f"Camera initialized on index {i}")
                    break
            else:
                camera = None
                print("Warning: Could not open any camera")
    return camera

def release_camera():
    """Release camera resources"""
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
            print("Camera released")

def detect_faces(frame):
    """Detect faces in frame"""
    if face_cascade is None:
        init_face_detector()
    
    if face_cascade is None:
        return []
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
    return faces

# ==================== ROUTES ====================

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/attendance')
def attendance_page():
    """Attendance page"""
    return render_template('attendance.html')

@app.route('/register')
def register_page():
    """Registration page"""
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard page"""
    return render_template('dashboard.html')

# ==================== VIDEO STREAM (FIXED) ====================

@app.route('/video')
def video_feed():
    """MJPEG video stream with proper MIME type"""
    def generate():
        cap = get_camera()
        if cap is None:
            # Return an error image if camera not available
            error_img = cv2.imread('static/images/camera_error.jpg')
            if error_img is None:
                error_img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_img, "Camera Not Available", (150, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', error_img)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            return
        
        frame_counter = 0
        
        while True:
            success, frame = cap.read()
            if not success:
                # Try to reconnect camera
                release_camera()
                time.sleep(1)
                cap = get_camera()
                if cap is None:
                    break
                continue
            
            # Detect faces every 30 frames (for performance)
            frame_counter += 1
            if frame_counter % 30 == 0:
                faces = detect_faces(frame)
            
            # Draw rectangles for detected faces
            if 'faces' in locals():
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, "Face Detected", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add timestamp overlay
            current_time = get_current_time()
            cv2.putText(frame, current_time, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        release_camera()
    
    return Response(generate(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# ==================== API ENDPOINTS ====================

@app.route('/api/attendance')
def api_get_attendance():
    """Get attendance records with consistent date format"""
    try:
        if not os.path.exists(ATTENDANCE_CSV):
            return jsonify({'data': [], 'total': 0})
        
        df = pd.read_csv(ATTENDANCE_CSV)
        
        # Ensure consistent date format
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        # Convert to records
        records = df.to_dict('records')
        
        return jsonify({
            'data': records,
            'total': len(records),
            'today_date': get_today_date()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/today-attendance')
def api_today_attendance():
    """Get today's attendance only"""
    try:
        if not os.path.exists(ATTENDANCE_CSV):
            return jsonify({'data': [], 'total': 0})
        
        df = pd.read_csv(ATTENDANCE_CSV)
        today = get_today_date()
        
        # Filter for today
        if 'date' in df.columns:
            df = df[df['date'] == today]
        
        return jsonify({
            'data': df.to_dict('records'),
            'total': len(df),
            'date': today
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mark-attendance', methods=['POST'])
def api_mark_attendance():
    """Mark attendance for a user"""
    try:
        data = request.json
        user_id = data.get('id')
        user_name = data.get('name')
        
        if not user_id or not user_name:
            return jsonify({'error': 'ID and name required'}), 400
        
        success, message = save_attendance_record(user_id, user_name)
        
        return jsonify({
            'success': success,
            'message': message,
            'name': user_name,
            'timestamp': get_full_timestamp()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/users')
def api_get_users():
    """Get registered users"""
    try:
        users = []
        if os.path.exists(NAMES_CSV):
            with open(NAMES_CSV, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    users.append({'id': row['id'], 'name': row['name']})
        return jsonify({'users': users})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def api_get_stats():
    """Get statistics"""
    try:
        # Count registered users
        user_count = 0
        if os.path.exists(NAMES_CSV):
            with open(NAMES_CSV, 'r', encoding='utf-8') as f:
                user_count = sum(1 for _ in f) - 1
        
        # Count attendance
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
    """Register a new user"""
    try:
        user_id = request.form.get('id')
        user_name = request.form.get('name')
        
        if not user_id or not user_name:
            return jsonify({'error': 'ID and name required'}), 400
        
        # Check if user exists
        if os.path.exists(NAMES_CSV):
            with open(NAMES_CSV, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['id'] == user_id:
                        return jsonify({'error': 'User ID already exists'}), 400
                    if row['name'] == user_name:
                        return jsonify({'error': 'Username already exists'}), 400
        
        # Capture face
        cap = get_camera()
        if cap is None:
            return jsonify({'error': 'Camera not available'}), 500
        
        # Wait for face detection
        face_captured = False
        best_frame = None
        
        for attempt in range(10):
            ret, frame = cap.read()
            if ret:
                faces = detect_faces(frame)
                if len(faces) > 0:
                    best_frame = frame
                    face_captured = True
                    break
            time.sleep(0.1)
        
        if not face_captured:
            return jsonify({'error': 'No face detected. Please position yourself clearly.'}), 400
        
        # Save face image
        face_path = os.path.join(KNOWN_FACES_DIR, f'{user_id}.jpg')
        cv2.imwrite(face_path, best_frame)
        
        # Save to names CSV
        file_exists = os.path.exists(NAMES_CSV)
        with open(NAMES_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['id', 'name'])
            writer.writerow([user_id, user_name])
        
        return jsonify({'success': True, 'message': f'User {user_name} registered successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== EXPORT ROUTES (FIXED) ====================

@app.route('/export/excel')
def export_excel():
    """Export to Excel with correct date format"""
    try:
        if os.path.exists(ATTENDANCE_CSV):
            df = pd.read_csv(ATTENDANCE_CSV)
        else:
            df = pd.DataFrame(columns=['id', 'name', 'timestamp', 'date', 'time', 'status'])
        
        # Ensure date format is consistent
        if 'date' in df.columns and len(df) > 0:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        # Remove duplicates (optional)
        if len(df) > 0 and 'id' in df.columns and 'date' in df.columns:
            df = df.drop_duplicates(subset=['id', 'date'], keep='first')
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Attendance', index=False)
            
            if len(df) > 0 and 'name' in df.columns:
                summary = df.groupby('name').size().reset_index(name='Total Days Present')
                summary.to_excel(writer, sheet_name='Summary', index=False)
            
            if len(df) > 0 and 'date' in df.columns:
                daily = df.groupby('date').size().reset_index(name='Present Count')
                daily = daily.sort_values('date')
                daily.to_excel(writer, sheet_name='Daily Summary', index=False)
        
        output.seek(0)
        filename = f'attendance_{get_today_date()}_{get_current_time().replace(":", "-")}.xlsx'
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        print(f"Export error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/export/csv')
def export_csv():
    """Export to CSV with correct date format"""
    try:
        if os.path.exists(ATTENDANCE_CSV):
            # Read and process data
            df = pd.read_csv(ATTENDANCE_CSV)
            
            # Ensure date format
            if 'date' in df.columns and len(df) > 0:
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            
            # Remove duplicates
            if len(df) > 0 and 'id' in df.columns:
                df = df.drop_duplicates(subset=['id', 'date'], keep='first')
            
            # Create CSV in memory
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            
            filename = f'attendance_{get_today_date()}_{get_current_time().replace(":", "-")}.csv'
            
            return Response(
                output.getvalue(),
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename={filename}'}
            )
        else:
            return jsonify({'error': 'No data available'}), 404
            
    except Exception as e:
        print(f"CSV export error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/export/json')
def export_json():
    """Export to JSON"""
    try:
        if os.path.exists(ATTENDANCE_CSV):
            df = pd.read_csv(ATTENDANCE_CSV)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            
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

@app.route('/api/clear-attendance', methods=['POST'])
def clear_attendance():
    """Clear all attendance records"""
    try:
        if os.path.exists(ATTENDANCE_CSV):
            os.remove(ATTENDANCE_CSV)
        init_csv_files()
        
        global marked_today_cache
        marked_today_cache = set()
        
        return jsonify({'success': True, 'message': 'All attendance records cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recognize-face', methods=['POST'])
def recognize_face():
    """Recognize face from uploaded image and mark attendance"""
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'message': 'No image data'}), 400
        
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'success': False, 'message': 'Invalid image'}), 400
        
        # Detect face
        faces = detect_faces(frame)
        
        if len(faces) == 0:
            return jsonify({'success': False, 'message': 'No face detected. Please position your face clearly.'})
        
        # For demo, check registered users
        users = get_registered_users()
        
        if len(users) == 0:
            return jsonify({'success': False, 'message': 'No registered users. Please register first.'})
        
        # Simple matching - for demo, use first user
        # In production, implement proper face matching
        user = users[0]
        
        # Mark attendance
        success, message = save_attendance_record(user['id'], user['name'])
        
        return jsonify({
            'success': success,
            'message': message,
            'name': user['name'],
            'time': get_current_time()
        })
        
    except Exception as e:
        print(f"Recognition error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500
# ==================== INITIALIZATION ====================

def initialize_system():
    """Initialize the system on startup"""
    init_csv_files()
    init_face_detector()
    refresh_marked_cache()
    
    print("=" * 60)
    print("🎓 FACE RECOGNITION ATTENDANCE SYSTEM")
    print("=" * 60)
    print(f"📍 Access: http://localhost:5000")
    print(f"📊 Dashboard: http://localhost:5000/dashboard")
    print(f"📅 Today's date: {get_today_date()}")
    print(f"✅ Each student recorded ONLY ONCE per day")
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
    
   