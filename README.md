📦 Object Detector (YOLOv8)
This one was a jump — from writing all the logic myself to using an actual AI model that already knows what 80 different objects look like. You point the camera at anything and it tells you what it sees, with a confidence score.

**Tech:** `Ultralytics YOLOv8` `OpenCV`

**How it works:**
- YOLOv8 (You Only Look Once) runs inference on each camera frame
- It outputs bounding box coordinates, class labels, and confidence scores
- I built the UI on top — FPS counter, color-coded boxes per class, object summary

**Features:**
- 🔍 Detects 80 object classes: person, cup, phone, car, dog, cat, and more
- 📊 Confidence score shown per detection
- 🎯 Adjustable confidence threshold at runtime (`+` / `-` keys)
- 📈 Rolling average FPS display
- 💾 Save screenshot with `S`

**What I learned:** The difference between rule-based Computer Vision and AI model inference, how YOLO processes frames, bounding boxes, and why confidence thresholds matter.

```bash
pip install ultralytics opencv-python
python object_detector.py
```

> 💡 First run auto-downloads `yolov8n.pt` (~6MB). Cached after that — no re-download needed.

---

👤 Face Recognition Attendance
A two-part system: register a face once, and after that the camera automatically detects and logs who's present. No manual input. Saves to a CSV file with timestamps.

**Tech:** `face_recognition` `OpenCV` `CSV`

**How it works:**
- `register_face.py` captures 5 photos of your face and converts them into 128-number vectors called encodings — essentially a numerical "fingerprint" of your face
- `attendance.py` compares every detected face against the stored encodings
- If the distance between the live face and a stored encoding is within the tolerance threshold, it's a match — and attendance is logged

**Features:**
- 📸 Register as many people as you want
- ✅ Auto-logs presence with name, date, and time
- 🚫 No duplicate entries within the same session
- 📋 New CSV file generated per day (`attendance_YYYYMMDD.csv`)
- 🟢 Green box = recognized, 🔴 Red box = unknown

**What I learned:** How face encodings work, face distance vs. tolerance, CSV file handling in Python, and how to manage session state so the same person isn't logged twice.

```bash
pip install face_recognition opencv-python
python register_face.py   # Step 1 — register your face
python attendance.py      # Step 2 — run the system
```

> ⚠️ `face_recognition` installs `dlib` as a dependency — first install takes a few minutes. That's normal.

---

🗺️ Where I'm headed

```
🟢 Computer Vision
   └── Virtual Paint          ✅ Done
   └── Eye Blink Detector     — coming soon
   └── Gesture Volume Control — coming soon

🟡 AI + Automation
   └── Object Detector        ✅ Done
   └── Face Recognition       ✅ Done
   └── Auto File Organizer    — coming soon
   └── Web Scraper + Alerts   — coming soon

🔵 Robotics
   └── Arduino + Python       — coming soon
   └── Hand Gesture Robot     — coming soon

🏆 Dream Project
   └── Hand Gesture Controlled Robot — the goal
```

---

⚙️ Requirements

```bash
# Computer Vision projects
pip install opencv-python mediapipe numpy

# Object Detection
pip install ultralytics

# Face Recognition
pip install face_recognition
```

---

📌 About

Computer Engineering student building toward AI + Robotics Engineering. These projects are part of a structured self-learning path — each one introduces new concepts, new libraries, and a new level of complexity.

Every script in this repo is written with detailed comments explaining not just *what* the code does, but *why* — because I think that's the only way to actually learn it.

**Stack:** Python 3.11 · OpenCV · MediaPipe · YOLOv8 · face_recognition
