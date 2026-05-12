import cv2
import os
import csv
from datetime import datetime

MODEL_PATH = 'trainer/trainer.yml'
ATTENDANCE_CSV = 'attendance/attendance.csv'

def ensure_attendance_csv(path=ATTENDANCE_CSV):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id','time','date'])

def mark_attendance(user_id):
    ensure_attendance_csv()
    now = datetime.now()
    time_str = now.strftime('%H:%M:%S')
    date_str = now.strftime('%Y-%m-%d')
    with open(ATTENDANCE_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([user_id, time_str, date_str])
    print(f'Marked attendance for {user_id} at {date_str} {time_str}')

def recognize_loop():
    if not os.path.exists(MODEL_PATH):
        print('Model not found. Train the model first with train_model.py')
        return
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ret, img = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            if confidence < 70:
                cv2.putText(img, f'ID {id}', (x+5, y-5), font, 1, (0,255,0), 2)
                mark_attendance(id)
            else:
                cv2.putText(img, 'Unknown', (x+5, y-5), font, 1, (0,0,255), 2)
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.imshow('camera', img)
        k = cv2.waitKey(10) & 0xff
        if k == 27: # ESC
            break
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    recognize_loop()
