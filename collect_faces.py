import cv2
import os

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def collect_faces(dataset_path='dataset', face_id=1, samples=30):
    ensure_dir(dataset_path)
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print('Error: could not open webcam')
        return

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count = 0
    print('Starting face capture. Look at the camera and press q to quit early.')
    while True:
        ret, img = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            file_path = os.path.join(dataset_path, f'user.{face_id}.{count}.jpg')
            cv2.imwrite(file_path, face_img)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, str(count), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
        cv2.imshow('image', img)
        k = cv2.waitKey(100) & 0xff
        if k == ord('q') or count >= samples:
            break

    print(f'Total {count} face samples collected for ID: {face_id}')
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Collect faces for a user ID')
    parser.add_argument('--id', type=int, default=1, help='Numeric user id')
    parser.add_argument('--samples', type=int, default=30, help='Number of samples to collect')
    parser.add_argument('--out', type=str, default='dataset', help='Output dataset directory')
    args = parser.parse_args()
    collect_faces(dataset_path=args.out, face_id=args.id, samples=args.samples)
