import cv2
import time
from collections         import deque
from datetime            import datetime
from ultralytics         import YOLO

MODEL_FILE = "yolov8n.pt"   # Nano model — mabilis para sa webcam

print(f"Loading YOLO model: {MODEL_FILE} ...")
model = YOLO(MODEL_FILE)    # I-load ang model (mag-do-download kung wala pa)
print("Model loaded! Starting camera...")

confidence_threshold = 0.5   # 50% confidence minimum (pwede baguhin sa runtime)
CONF_STEP            = 0.05  # Gaano kalaki ang taas/baba ng confidence per keypress

show_fps   = True   # Ipakita ba ang FPS counter?
show_count = True   # Ipakita ba ang bilang ng detected objects?

# Lista ng BGR colors para sa bounding boxes (bawat klase = ibang kulay)
BOX_COLORS = [
    (0,   255, 100),   # Green
    (0,   180, 255),   # Orange
    (255, 50,  50 ),   # Blue
    (180, 0,   255),   # Purple
    (0,   255, 255),   # Yellow
    (255, 0,   150),   # Pink
    (100, 255, 0  ),   # Lime
    (255, 200, 0  ),   # Cyan-ish
]

FPS_BUFFER_SIZE = 30                  # Gaano karaming frames ang ia-average
fps_times       = deque(maxlen=FPS_BUFFER_SIZE)  # Circular buffer ng timestamps
fps_display     = 0.0                 # Kasalukuyang iniistoring FPS value


cap = cv2.VideoCapture(0)            # 0 = default webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def draw_box(frame, x1, y1, x2, y2, label, confidence, color):
    """
    frame      = image kung saan iguguhit ang box
    x1,y1      = upper-left corner ng box (pixels)
    x2,y2      = lower-right corner ng box (pixels)
    label      = pangalan ng detected object (e.g. "person", "cup")
    confidence = certainty ng AI (0.0 to 1.0)
    color      = BGR tuple ng kulay ng box
    """
    # -- Iguhit ang bounding box rectangle ---------------------
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # -- Gumawa ng label text: "person 87%" --------------------
    text = f"{label} {confidence:.0%}"

    # Kumuha ng laki ng text para maayos ang placement ng background
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )

    # -- Iguhit ang filled rectangle bilang background ng label -
    # Nakalagay ito sa ITAAS ng bounding box (y1 - text_h - 8)
    # Kung malapit sa taas ng screen, ilipat sa loob ng box
    label_y = y1 - 10 if y1 - 10 > text_h else y1 + text_h + 10
    cv2.rectangle(
        frame,
        (x1, label_y - text_h - 6),
        (x1 + text_w + 4, label_y + baseline),
        color, -1   # -1 = filled rectangle
    )

    # -- Iguhit ang label text sa ibabaw ng background ----------
    # Gumamit ng dark text para makita sa maliwanag na background
    cv2.putText(
        frame, text,
        (x1 + 2, label_y - 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (0, 0, 0),   # Black text
        2, cv2.LINE_AA
    )

def draw_ui(frame, fps, obj_count, conf_thresh, show_fps_flag):
    h, w = frame.shape[:2]

    # -- Dark panel sa ibaba ng screen (para sa info display) ---
    cv2.rectangle(frame, (0, h - 50), (w, h), (20, 20, 20), -1)

    # -- FPS Counter (kung naka-enable) -------------------------
    if show_fps_flag:
        fps_color = (0, 255, 100) if fps >= 25 else (0, 165, 255) if fps >= 15 else (0, 0, 255)
        # Green = mabilis (25+), Orange = katamtaman (15-24), Red = mabagal (<15)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, fps_color, 2, cv2.LINE_AA)

    # -- Object count ------------------------------------------
    count_text = f"Objects: {obj_count}"
    cv2.putText(frame, count_text, (w // 2 - 60, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # -- Confidence threshold indicator ------------------------
    conf_text = f"Conf: {conf_thresh:.0%}  [+/-]"
    cv2.putText(frame, conf_text, (w - 200, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    # -- Keyboard hints sa itaas-kanan -------------------------
    hints = "Q=Quit  S=Save  F=FPS"
    cv2.putText(frame, hints, (w - 240, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

print("\nObject Detector started!")
print("Controls: Q=Quit | S=Save | F=Toggle FPS | +/-=Confidence")
print(f"Confidence threshold: {confidence_threshold:.0%}\n")

while True:
    # -- Basahin ang frame --------------------------------------
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)   # Mirror effect

    # -- I-record ang oras ng frame na ito para sa FPS ----------
    fps_times.append(time.time())

    # -- Compute ang FPS gamit ang rolling average --------------
    # Kinukuha natin ang elapsed time sa pagitan ng pinaka-luma at
    # pinaka-bagong frame sa ating buffer
    if len(fps_times) >= 2:
        elapsed   = fps_times[-1] - fps_times[0]   # Total elapsed time
        fps_display = (len(fps_times) - 1) / elapsed  # Frames / segundo
    results = model(
        frame,
        conf    = confidence_threshold,
        stream  = True,    # Generator mode = mas memory-efficient
        verbose = False    # Huwag mag-print ng detection logs sa terminal
    )

    detected_objects = []   # Lista ng lahat ng na-detect sa frame na ito

    # -- I-parse ang results ------------------------------------
    # Ang results ay isang generator ng Detection objects.
    # Bawat result ay may .boxes na naglalaman ng lahat ng detections.
    for result in results:
        boxes = result.boxes   # Lahat ng detected boxes sa frame na ito

        if boxes is None:
            continue   # Walang na-detect sa frame na ito

        for box in boxes:
            # -- Kumuha ng bounding box coordinates (sa pixels) -
            # box.xyxy[0] = [x1, y1, x2, y2] tensor
            # .int().tolist() = i-convert sa integers list
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # -- Kumuha ng class ID at pangalan -----------------
            # box.cls[0] = class index (e.g. 0 = person, 39 = bottle)
            class_id = int(box.cls[0])

            # model.names = dictionary ng {class_id: class_name}
            label = model.names[class_id]

            # -- Kumuha ng confidence score --------------------
            # box.conf[0] = confidence ng detection (0.0 to 1.0)
            confidence = float(box.conf[0])

            # -- Piliin ang kulay para sa klase na ito ---------
            # Gamit ang modulo para mag-cycle sa BOX_COLORS list
            color = BOX_COLORS[class_id % len(BOX_COLORS)]

            # -- Iguhit ang bounding box at label ---------------
            draw_box(frame, x1, y1, x2, y2, label, confidence, color)

            # I-record ang detected object para sa count
            detected_objects.append(label)

    # -- Ipakita ang summary kung may mga na-detect ------------
    # Gumamit ng Counter-like approach: ibida ang bawat klase at count
    if detected_objects:
        # Gumawa ng summary text (e.g. "person:2  cup:1  phone:1")
        from collections import Counter
        counts  = Counter(detected_objects)
        summary = "  ".join([f"{name}:{n}" for name, n in counts.items()])

        # Ipakita ang summary sa itaas ng frame
        cv2.rectangle(frame, (0, 0), (len(summary) * 11 + 10, 35), (20, 20, 20), -1)
        cv2.putText(frame, summary, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 150), 2, cv2.LINE_AA)

    # -- Iguhit ang UI elements --------------------------------
    draw_ui(frame, fps_display, len(detected_objects),
            confidence_threshold, show_fps)

    # -- Ipakita ang frame ------------------------------------
    cv2.imshow("Object Detector (YOLOv8)", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        # Q = Quit
        print("Exiting...")
        break

    elif key == ord('s'):
        # S = Save current frame as PNG
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"detection_{timestamp}.png"
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")

    elif key == ord('f'):
        # F = Toggle FPS display on/off
        show_fps = not show_fps
        print(f"FPS display: {'ON' if show_fps else 'OFF'}")

    elif key == ord('+') or key == ord('='):
        # + = Taas ang confidence threshold (mas strict)
        confidence_threshold = min(0.95, confidence_threshold + CONF_STEP)
        print(f"Confidence threshold: {confidence_threshold:.0%}")

    elif key == ord('-'):
        # - = Baba ang confidence threshold (mas loose)
        confidence_threshold = max(0.05, confidence_threshold - CONF_STEP)
        print(f"Confidence threshold: {confidence_threshold:.0%}")

cap.release()
cv2.destroyAllWindows()
print("Object Detector closed.")
