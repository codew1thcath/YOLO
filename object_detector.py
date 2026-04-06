import cv2
import time
from collections import deque, Counter
from datetime    import datetime
from ultralytics import YOLO

MODEL_FILE = "yolov8n.pt"

# ─── Target classes only ────────────────────────────────────────────────────
# These are the COCO class names we want to detect.
# Backpack + handbag + suitcase are all treated as "bag" variants.
TARGET_CLASSES = {
    "person",
    "cup",
    "sports ball",
    "cell phone",
    "handbag",
    "backpack",
    "suitcase",
}

# Friendly display names (maps raw COCO name → what shows on screen)
DISPLAY_NAME = {
    "person":      "Person",
    "cup":         "Cup",
    "sports ball": "Ball",
    "cell phone":  "Phone",
    "handbag":     "Bag",
    "backpack":    "Bag",
    "suitcase":    "Bag",
}

# Per-class colors (BGR)
CLASS_COLOR = {
    "person":      (0,   255, 100),   # Green
    "cup":         (0,   180, 255),   # Orange
    "sports ball": (255, 50,  50 ),   # Blue
    "cell phone":  (180, 0,   255),   # Purple
    "handbag":     (0,   255, 255),   # Yellow
    "backpack":    (0,   200, 220),   # Teal
    "suitcase":    (50,  180, 255),   # Light orange
}

print(f"Loading YOLO model: {MODEL_FILE} ...")
model = YOLO(MODEL_FILE)
print("Model loaded! Starting camera...")
print(f"Detecting: {', '.join(sorted(TARGET_CLASSES))}\n")

confidence_threshold = 0.5
CONF_STEP            = 0.05

show_fps   = True
show_count = True

FPS_BUFFER_SIZE = 30
fps_times       = deque(maxlen=FPS_BUFFER_SIZE)
fps_display     = 0.0

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ─── Build a set of class IDs to pass to YOLO (faster filtering) ─────────────
TARGET_IDS = {cid for cid, name in model.names.items() if name in TARGET_CLASSES}


def draw_box(frame, x1, y1, x2, y2, label, confidence, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"{label} {confidence:.0%}"
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )

    label_y = y1 - 10 if y1 - 10 > text_h else y1 + text_h + 10
    cv2.rectangle(
        frame,
        (x1, label_y - text_h - 6),
        (x1 + text_w + 4, label_y + baseline),
        color, -1
    )
    cv2.putText(
        frame, text,
        (x1 + 2, label_y - 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (0, 0, 0), 2, cv2.LINE_AA
    )


def draw_ui(frame, fps, obj_count, conf_thresh, show_fps_flag):
    h, w = frame.shape[:2]

    # Bottom info bar
    cv2.rectangle(frame, (0, h - 50), (w, h), (20, 20, 20), -1)

    if show_fps_flag:
        fps_color = (0, 255, 100) if fps >= 25 else (0, 165, 255) if fps >= 15 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, fps_color, 2, cv2.LINE_AA)

    cv2.putText(frame, f"Objects: {obj_count}", (w // 2 - 60, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, f"Conf: {conf_thresh:.0%}  [+/-]", (w - 200, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.putText(frame, "Q=Quit  S=Save  F=FPS", (w - 240, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    # Legend (top-left, below summary bar)
    legend_items = [
        ("Person",  CLASS_COLOR["person"]),
        ("Cup",     CLASS_COLOR["cup"]),
        ("Ball",    CLASS_COLOR["sports ball"]),
        ("Phone",   CLASS_COLOR["cell phone"]),
        ("Bag",     CLASS_COLOR["handbag"]),
    ]
    legend_y = 55
    for name, color in legend_items:
        cv2.rectangle(frame, (10, legend_y - 10), (22, legend_y + 2), color, -1)
        cv2.putText(frame, name, (28, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        legend_y += 20


print("Object Detector started!")
print("Controls: Q=Quit | S=Save | F=Toggle FPS | +/-=Confidence\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    fps_times.append(time.time())

    if len(fps_times) >= 2:
        elapsed     = fps_times[-1] - fps_times[0]
        fps_display = (len(fps_times) - 1) / elapsed

    # ── Run YOLO, pre-filtered to our target class IDs only ──────────────────
    results = model(
        frame,
        conf    = confidence_threshold,
        classes = list(TARGET_IDS),   # <── only detect our chosen classes
        stream  = True,
        verbose = False
    )

    detected_objects = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id        = int(box.cls[0])
            raw_label       = model.names[class_id]
            confidence      = float(box.conf[0])

            # Map to friendly display name
            display_label = DISPLAY_NAME.get(raw_label, raw_label.title())
            color         = CLASS_COLOR.get(raw_label, (200, 200, 200))

            draw_box(frame, x1, y1, x2, y2, display_label, confidence, color)
            detected_objects.append(display_label)

    # ── Summary bar ──────────────────────────────────────────────────────────
    if detected_objects:
        counts  = Counter(detected_objects)
        summary = "  ".join([f"{name}:{n}" for name, n in counts.items()])
        cv2.rectangle(frame, (0, 0), (len(summary) * 11 + 10, 35), (20, 20, 20), -1)
        cv2.putText(frame, summary, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 150), 2, cv2.LINE_AA)

    draw_ui(frame, fps_display, len(detected_objects), confidence_threshold, show_fps)
    cv2.imshow("Object Detector — Person / Cup / Ball / Phone / Bag", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("Exiting...")
        break
    elif key == ord('s'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"detection_{timestamp}.png"
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
    elif key == ord('f'):
        show_fps = not show_fps
        print(f"FPS display: {'ON' if show_fps else 'OFF'}")
    elif key in (ord('+'), ord('=')):
        confidence_threshold = min(0.95, confidence_threshold + CONF_STEP)
        print(f"Confidence threshold: {confidence_threshold:.0%}")
    elif key == ord('-'):
        confidence_threshold = max(0.05, confidence_threshold - CONF_STEP)
        print(f"Confidence threshold: {confidence_threshold:.0%}")

cap.release()
cv2.destroyAllWindows()
print("Object Detector closed.")