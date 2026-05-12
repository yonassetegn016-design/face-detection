# FaceRecognitionAttendance

Simple face-recognition-based attendance system using OpenCV and Flask.

Usage:

- Install dependencies: python -m pip install -r requirements.txt
- Collect faces: python collect_faces.py --id 1 --samples 30
- Train model: python train_model.py
- Run recognition: python recognize.py
- View/download attendance via web UI: python app.py then open http://localhost:5000

Files:

- `collect_faces.py` - Captures face images and stores them in `dataset/`.
- `train_model.py` - Trains an LBPH face recognizer and saves model to `trainer/`.
- `recognize.py` - Runs live recognition and appends to `attendance/attendance.csv`.
- `app.py` - Simple Flask app to download the attendance CSV.

Notes:

- This is a minimal demo. For production, add identity mapping (id->name), deduplication in CSV, and security.
