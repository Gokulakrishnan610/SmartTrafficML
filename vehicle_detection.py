import cv2
import os
import urllib.request
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Source: '0' = webcam, or 'http://ip/cam-hi.jpg' for ESP32-CAM
SOURCE = '0'

vehicle_count_file = "vehicle_count_south.txt"
if os.path.exists(vehicle_count_file):
    os.remove(vehicle_count_file)

vehicle_classes = ['car', 'motorbike', 'bus', 'truck', 'bicycle', 'motorcycle']
ambulance_class = 'truck'
confidence_threshold = 0.5


def log_vehicle_count(count, previous_count, ambulance_detected):
    if count != previous_count:
        with open(vehicle_count_file, 'w') as f:
            now = datetime.now()
            ts = now.strftime('%H:%M:%S')
            amb = "ambulance true" if ambulance_detected else "ambulance false"
            f.write(f'Vehicle{count},{ts} {amb}\n')


def get_frame_from_url(url):
    """Fetch a JPEG frame from an HTTP URL (ESP32-CAM)."""
    img_resp = urllib.request.urlopen(url, timeout=5)
    arr = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    return cv2.imdecode(arr, -1)


def main():
    previous_vehicle_count = 0

    # Decide source: webcam index or HTTP URL
    use_webcam = SOURCE.isdigit() or SOURCE == '0'
    cap = cv2.VideoCapture(int(SOURCE)) if use_webcam else None

    print(f"📷 Source: {'Webcam ' + SOURCE if use_webcam else SOURCE}")

    while True:
        try:
            if use_webcam:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("⚠️  Webcam read failed, retrying…")
                    continue
            else:
                frame = get_frame_from_url(SOURCE)
                if frame is None:
                    continue

            results = model(frame, imgsz=640, verbose=False)[0]
            vehicle_count = 0
            ambulance_detected = False

            for box in results.boxes:
                if box.conf > confidence_threshold:
                    cls_id = int(box.cls)
                    label = model.names[cls_id]

                    if label in vehicle_classes:
                        vehicle_count += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    if label == ambulance_class and box.conf > confidence_threshold:
                        ambulance_detected = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, "Ambulance", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            log_vehicle_count(vehicle_count, previous_vehicle_count, ambulance_detected)
            previous_vehicle_count = vehicle_count

            cv2.putText(frame, f"Vehicles (South): {vehicle_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if ambulance_detected:
                cv2.putText(frame, "Ambulance Detected!", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('South Road — Vehicle Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Error: {e}")

    if cap:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
