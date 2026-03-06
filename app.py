"""
Real-time Object Detection + Tracking
YOLOv8 + Deep SORT | Video routes | Screenshot with SQLite
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from flask import Flask, Response, jsonify, request, render_template, send_from_directory
from flask_cors import CORS
import threading
import queue
import time
import os
import sqlite3
from datetime import datetime
from collections import defaultdict
import math
import json

app = Flask(__name__)
CORS(app)

VIDEO_ROUTES = {
    "traffic": {
        "label": "Traffic (cars, bus, person)",
        "path": None,
        "classes": ["car", "bus", "person"],
    },
    "robber": {
        "label": "Robber",
        "path": None,
        "classes": ["person"],
    },
    "crook": {
        "label": "Crook",
        "path": None,
        "classes": ["person"],
    },
}

DEFAULT_VIDEO_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "video-detection",
    "2103099-uhd_2560_1440_30fps.mp4",
)

video_capture = None
processing_thread = None
frame_queue = queue.Queue(maxsize=10)
current_frame_lock = threading.Lock()
current_frame_bytes = None
is_processing = False
active_route = "traffic"
track_history = defaultdict(list)
risk_scores = defaultdict(lambda: 50)
frame_count = 0
start_time = None

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots.db")
SCREENSHOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")


def init_db():
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS screenshot_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL,
            feed_route TEXT NOT NULL,
            created_at TEXT NOT NULL,
            detections_json TEXT
        )"""
    )
    conn.commit()
    conn.close()


def save_screenshot_to_db(file_path, feed_route, detections_json):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO screenshot_log (file_path, feed_route, created_at, detections_json)
           VALUES (?, ?, ?, ?)""",
        (file_path, feed_route, datetime.utcnow().isoformat(), detections_json),
    )
    conn.commit()
    conn.close()


class ObjectDetectionSystem:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tracker = None

    def load_model(self):
        self.model = YOLO("yolov8n.pt")
        self.model.to(self.device)

    def load_tracker(self):
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            max_cosine_distance=0.3,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=(self.device == "cuda"),
        )

    def get_class_name(self, class_id):
        if self.model and hasattr(self.model, "names"):
            return self.model.names[class_id]
        return f"class_{class_id}"

    def calculate_speed(self, points):
        if len(points) < 2:
            return 0
        dx = points[-1][0] - points[-2][0]
        dy = points[-1][1] - points[-2][1]
        return math.sqrt(dx * dx + dy * dy)

    def process_frame(self, frame):
        results = self.model(frame, conf=0.5)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    w = x2 - x1
                    h = y2 - y1
                    detections.append(([int(x1), int(y1), int(w), int(h)], conf, cls))

        if detections:
            tracks = self.tracker.update_tracks(detections, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                l, t, r, b = map(int, track.to_ltrb())
                cls = track.get_det_class()
                conf = track.get_det_conf()
                class_name = self.get_class_name(cls)

                center = ((l + r) // 2, (t + b) // 2)
                track_history[track_id].append(center)
                if len(track_history[track_id]) > 30:
                    track_history[track_id].pop(0)

                speed = self.calculate_speed(track_history[track_id])
                risk_scores[track_id] = min(
                    100,
                    risk_scores[track_id] + 1 if speed > 10 else risk_scores[track_id] - 1,
                )

                cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ID:{track_id} {class_name} {conf:.2f}",
                    (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        return frame

    def get_detections_list(self, frame):
        if self.model is None:
            return []
        results = self.model(frame, conf=0.5)
        out = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                out.append({"class": self.get_class_name(cls), "confidence": round(conf, 2)})
        return out


detection_system = ObjectDetectionSystem()


def get_video_source():
    route_config = VIDEO_ROUTES.get(active_route, VIDEO_ROUTES["traffic"])
    path = route_config.get("path") or DEFAULT_VIDEO_PATH
    if path and os.path.isfile(path):
        return path
    return DEFAULT_VIDEO_PATH


def video_loop():
    global video_capture, is_processing, current_frame_bytes, frame_count, start_time
    start_time = time.time()

    while is_processing:
        if video_capture is None or not video_capture.isOpened():
            time.sleep(0.2)
            continue

        ret, frame = video_capture.read()
        if not ret:
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        processed = detection_system.process_frame(frame)
        _, buffer = cv2.imencode(".jpg", processed)
        frame_bytes = buffer.tobytes()
        frame_count += 1

        with current_frame_lock:
            current_frame_bytes = frame_bytes

        if not frame_queue.full():
            frame_queue.put(frame_bytes)
        time.sleep(0.03)


def generate_frames():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
        else:
            time.sleep(0.01)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/health")
def health():
    global frame_count, start_time
    fps = 0
    if start_time and frame_count > 0:
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
    return jsonify({
        "status": "active" if is_processing else "inactive",
        "device": detection_system.device,
        "fps": round(fps, 1),
        "latency": f"{int(33)}ms",
    })


@app.route("/api/routes", methods=["GET"])
def get_routes():
    return jsonify(
        {
            k: {"label": v["label"], "active": k == active_route}
            for k, v in VIDEO_ROUTES.items()
        }
    )


@app.route("/api/switch_route", methods=["POST"])
def switch_route():
    global video_capture, active_route

    data = request.get_json() or {}
    route = data.get("route", "traffic")
    if route not in VIDEO_ROUTES:
        return jsonify({"error": "Invalid route"}), 400

    custom_path = data.get("video_path", "").strip()
    if custom_path:
        VIDEO_ROUTES[route]["path"] = custom_path if os.path.isfile(custom_path) else None

    active_route = route
    source = get_video_source()

    if video_capture is not None:
        video_capture.release()
    video_capture = cv2.VideoCapture(source)

    if not video_capture.isOpened():
        return jsonify({"error": "Failed to open video source"}), 400

    return jsonify({"status": "ok", "route": route, "source": source})


def _do_take_screenshot():
    global current_frame_bytes, active_route

    with current_frame_lock:
        frame_bytes = current_frame_bytes

    if frame_bytes is None:
        return None, "No live frame available"

    filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    file_path = os.path.join(SCREENSHOTS_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(frame_bytes)

    frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame_cv = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
    detections = []
    if frame_cv is not None:
        detections = detection_system.get_detections_list(frame_cv)
    detections_json = json.dumps(detections)

    save_screenshot_to_db(filename, active_route, detections_json)

    return {
        "saved": True,
        "filename": filename,
        "image_url": f"/screenshots/{filename}",
        "feed_route": active_route,
        "detections": detections,
    }, None


@app.route("/api/take-screenshot", methods=["POST"])
def take_screenshot():
    data, err = _do_take_screenshot()
    if err:
        return jsonify({"error": err}), 400
    return jsonify(data)


@app.route("/capture", methods=["POST"])
def capture():
    data, err = _do_take_screenshot()
    if err:
        return jsonify({"error": err}), 400
    return jsonify(data)


@app.route("/screenshots/<path:filename>")
def serve_screenshot(filename):
    return send_from_directory(SCREENSHOTS_DIR, filename)


@app.route("/api/screenshots", methods=["GET"])
def list_screenshots():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT id, file_path, feed_route, created_at, detections_json
           FROM screenshot_log ORDER BY id DESC LIMIT 100"""
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        out.append(
            {
                "id": r["id"],
                "file_path": r["file_path"],
                "feed_route": r["feed_route"],
                "created_at": r["created_at"],
                "image_url": f"/screenshots/{r['file_path']}",
                "detections": json.loads(r["detections_json"]) if r["detections_json"] else [],
            }
        )
    return jsonify(out)


if __name__ == "__main__":
    init_db()
    os.makedirs("templates", exist_ok=True)

    source = get_video_source()
    if not os.path.isfile(source):
        os.makedirs(os.path.dirname(source) or ".", exist_ok=True)

    video_capture = cv2.VideoCapture(source)
    if not video_capture.isOpened():
        video_capture = cv2.VideoCapture(0)

    detection_system.load_model()
    detection_system.load_tracker()

    is_processing = True
    processing_thread = threading.Thread(target=video_loop)
    processing_thread.daemon = True
    processing_thread.start()

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
