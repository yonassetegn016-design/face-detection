import cv2
import numpy as np
from pathlib import Path
import os

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def get_images_and_labels(dataset_path='dataset'):
    image_paths = [p for p in Path(dataset_path).glob('*.jpg')]
    face_samples = []
    ids = []
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    for image_path in image_paths:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # Expect filename format user.<id>.<count>.jpg
        fname = image_path.name
        parts = fname.split('.')
        if len(parts) < 3:
            continue
        id = int(parts[1])
        faces = face_detector.detectMultiScale(img)
        for (x, y, w, h) in faces:
            face_samples.append(img[y:y+h, x:x+w])
            ids.append(id)
    return face_samples, ids

def train_and_save(dataset_path='dataset', trainer_path='trainer', model_name='trainer.yml'):
    ensure_dir(trainer_path)
    faces, ids = get_images_and_labels(dataset_path)
    if len(faces) == 0:
        print('No faces found in dataset. Please collect faces first.')
        return
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    model_file = os.path.join(trainer_path, model_name)
    recognizer.write(model_file)
    print(f'Training completed. Model saved to {model_file}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train LBPH face recognizer')
    parser.add_argument('--dataset', default='dataset')
    parser.add_argument('--out', default='trainer')
    args = parser.parse_args()
    train_and_save(dataset_path=args.dataset, trainer_path=args.out)
