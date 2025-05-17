from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import base64
import time
import numpy as np
import threading
import requests
from ultralytics import YOLO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

model = YOLO("tesbest.pt")
cap = cv2.VideoCapture("Kalibrasi Benar.mp4")

REAL_DISTANCE = 2.0  # meter
ESP8266_URL = "http://192.168.191.184/update"  # ganti dengan IP ESP8266

target_width = 800
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
target_height = int(target_width * original_height / original_width)

prev_frame_time = 0

track_memory = {}  # id: {"centroid": (x,y), "t1":..., "t2":..., "sent": False, "class": "motor/mobil"}
object_id = 0

# Variabel untuk menyimpan jumlah kendaraan unik
total_motor = 0
total_mobil = 0

def send_to_esp(speed):
    """Fungsi untuk mengirim data ke ESP8266 dengan penanganan error"""
    try:
        # Perubahan parameter dari 'speed' menjadi 'kecepatan' sesuai ESP8266
        response = requests.get(f"{ESP8266_URL}?kecepatan={int(speed)}", timeout=1)
        if response.status_code == 200:
            print(f"Berhasil mengirim kecepatan {speed} ke ESP8266")
        else:
            print(f"Gagal mengirim data ke ESP8266. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error saat mengirim ke ESP8266: {e}")

def generate_frames():
    global prev_frame_time, object_id, total_motor, total_mobil

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (target_width, target_height))
        current_time = time.time()
        fps = 1 / (current_time - prev_frame_time) if prev_frame_time != 0 else 0
        prev_frame_time = current_time

        results = model(frame)[0]

        line1_y = int(target_height * 0.5)
        line2_y = int(target_height * 0.625)

        cv2.line(frame, (0, line1_y), (target_width, line1_y), (0, 255, 0), 2)
        cv2.line(frame, (0, line2_y), (target_width, line2_y), (0, 255, 0), 2)

        current_boxes = []

        for box in results.boxes.data:
            x1, y1, x2, y2, conf, cls = box.cpu().numpy()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            class_name = results.names[int(cls)]

            color = (0, 255, 255) if class_name == "motor" else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            matched_id = None
            for tid, data in track_memory.items():
                prev_cx, prev_cy = data["centroid"]
                if abs(prev_cx - cx) < 50 and abs(prev_cy - cy) < 50:
                    matched_id = tid
                    break

            if matched_id is None:
                matched_id = object_id
                object_id += 1

            current_boxes.append(matched_id)
            track = track_memory.setdefault(matched_id, {
                "centroid": (cx, cy),
                "t1": None,
                "t2": None,
                "sent": False,
                "class": class_name
            })

            track["centroid"] = (cx, cy)

            if line1_y - 5 < cy < line1_y + 5 and track["t1"] is None:
                track["t1"] = time.time()

            if line2_y - 5 < cy < line2_y + 5 and track["t2"] is None:
                track["t2"] = time.time()

            if track["t1"] and track["t2"] and not track["sent"]:
                time_diff = track["t2"] - track["t1"]
                if time_diff > 0:
                    speed = round((REAL_DISTANCE / time_diff) * 3.6, 2)
                    cv2.putText(frame, f"{speed} km/h", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                    # Tambah hitungan kendaraan
                    if track["class"] == "mobil":
                        total_mobil += 1
                    else:
                        total_motor += 1

                    # Kirim data ke web
                    socketio.emit("data_info", {
                        "kecepatan": speed,
                        "fps": int(fps),
                        "jumlah_mobil": total_mobil,
                        "jumlah_motor": total_motor
                    })

                    # Kirim ke ESP dalam thread terpisah
                    threading.Thread(target=send_to_esp, args=(speed,)).start()

                    track["sent"] = True

        # Hapus track lama
        for tid in list(track_memory):
            if tid not in current_boxes:
                if time.time() - (track_memory[tid].get("t2") or track_memory[tid].get("t1") or 0) > 2:
                    del track_memory[tid]

        # Tampilkan FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Kirim frame ke web
        _, buffer = cv2.imencode(".jpg", frame)
        frame_encoded = base64.b64encode(buffer).decode("utf-8")
        socketio.emit("video_frame", frame_encoded)

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("connect")
def connect():
    socketio.start_background_task(generate_frames)

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8000)