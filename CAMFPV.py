import cv2
import numpy as np
import time
import json
import os
from ultralytics import YOLO
from collections import defaultdict, deque
import logging
from pymavlink import mavutil

# === Настройка логирования ===
logging.basicConfig(filename='detections.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# === Пути и параметры ===
model = YOLO('/home/orangepi/Documents/YOLO/best_rknn_model')  # Убедись, что модель совместима с rknn

# Параметры экрана
screen_width = 720
screen_height = 576
cv2.namedWindow("Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Камеры: 4 камеры, расположены под углом 90° друг к другу (360° / 4)
CAMERA_COUNT = 4
CAMERA_IDS = [0, 1, 2, 3]  # Порядок: 0=вперёд, 1=вправо, 2=назад, 3=влево
CAMERA_ANGLES = [0, 90, 180, 270]  # Азимут каждой камеры относительно курса БПЛА (в градусах)

# Размер кропа для детекции
CROP_SIZE = 320
offset_buffer = deque(maxlen=20)

# Буфер для трекинга объектов (по ID)
track_history = defaultdict(list)

# Переменные для GPS и MAVLink
current_gps = {'lat': 0, 'lon': 0, 'alt': 0, 'heading': 0}  # Текущие координаты и курс
gps_lock = False

# === MAVLink: подключение к полётному контроллеру ===
def connect_mavlink(port='/dev/serial0', baud=921600):
    try:
        master = mavutil.mavlink_connection(port, baud=baud)
        logging.info(f"MAVLink connected to {port}")
        return master
    except Exception as e:
        logging.error(f"Failed to connect to MAVLink: {e}")
        return None

def read_mavlink_gps(master):
    global current_gps, gps_lock
    msg = master.recv_match(type=['GPS_RAW_INT', 'VFR_HUD'], blocking=False)
    if msg:
        if msg.get_type() == 'GPS_RAW_INT':
            if msg.fix_type >= 3:  # 3D fix
                current_gps['lat'] = msg.lat * 1e-7  # в градусы
                current_gps['lon'] = msg.lon * 1e-7
                current_gps['alt'] = msg.alt * 1e-3  # в метры
                gps_lock = True
        elif msg.get_type() == 'VFR_HUD':
            current_gps['heading'] = msg.heading  # 0-360 градусов

# === Кроп центра кадра ===
def crop_frame(frame, center_x, center_y, size):
    crop_x1 = max(center_x - size // 2, 0)
    crop_x2 = min(center_x + size // 2, frame.shape[1])
    crop_y1 = max(center_y - size // 2, 0)
    crop_y2 = min(center_y + size // 2, frame.shape[0])
    return frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()

# === Сохранение данных о пожаре ===
def save_fire_alert(camera_id, obj_class, confidence, distance_m, direction_deg, lat, lon):
    data = {
        "timestamp": time.time(),
        "camera_id": camera_id,
        "class": obj_class,
        "confidence": confidence,
        "distance_m": round(distance_m, 2),
        "direction_deg": round(direction_deg, 2),
        "drone_gps": {
            "lat": lat,
            "lon": lon,
            "alt": current_gps['alt']
        }
    }
    tmp_filename = 'fire_alert_tmp.json'
    final_filename = 'fire_alert.json'
    try:
        with open(tmp_filename, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_filename, final_filename)
        logging.info(f"Fire alert saved: {data}")
    except Exception as e:
        logging.error(f"Error saving fire alert: {e}")

# === Оценка расстояния до объекта (упрощённо) ===
def estimate_distance(box_width_px, focal_length_px=500, real_width_m=10):
    """
    Пример: если дым занимает 100px при фокусе 500px, и реальный размер 10м,
    то расстояние ~ (focal * real) / width = (500*10)/100 = 50м
    """
    if box_width_px <= 0:
        return float('inf')
    return (focal_length_px * real_width_m) / box_width_px

# === Определение абсолютного направления пожара ===
def calculate_absolute_direction(camera_azimuth, offset_x, frame_width, drone_heading):
    """
    camera_azimuth: направление камеры (0,90,180,270)
    offset_x: смещение объекта от центра кадра (в пикселях)
    frame_width: ширина кропа (320)
    drone_heading: курс БПЛА (0-360)
    Возвращает: абсолютный азимут (в градусах) от севера
    """
    # Угол отклонения от центра камеры
    angle_per_pixel = 60 / frame_width  # Пример: 60° FOV
    offset_angle = offset_x * angle_per_pixel  # градусы

    # Абсолютное направление камеры в мире
    absolute_camera_dir = (camera_azimuth + drone_heading) % 360

    # Направление на объект
    object_direction = (absolute_camera_dir + offset_angle) % 360
    return object_direction

# === Отображение ===
def display_frame(frame_resized, cropped_frame, fps, camera_id, fire_detected):
    detection_preview_size = 200
    cropped_resized = cv2.resize(cropped_frame, (detection_preview_size, detection_preview_size))
    y_offset = screen_height - detection_preview_size
    x_offset = screen_width - detection_preview_size
    frame_resized[y_offset:y_offset + detection_preview_size, x_offset:x_offset + detection_preview_size] = cropped_resized

    cv2.putText(frame_resized, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame_resized, f"Camera: {camera_id}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    color = (0, 0, 255) if fire_detected else (0, 255, 0)
    cv2.putText(frame_resized, f"Status: {'FIRE!' if fire_detected else 'OK'}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Detection", frame_resized)

# === Основной цикл ===
def main():
    global current_gps, gps_lock

    # === Инициализация камер ===
    caps = []
    for cam_id in CAMERA_IDS:
        cap = cv2.VideoCapture(cam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        if not cap.isOpened():
            logging.error(f"Camera {cam_id} failed to open.")
            cap = None
        caps.append(cap)

    # === Подключение MAVLink ===
    mavlink_master = connect_mavlink()
    if not mavlink_master:
        logging.warning("Running without GPS data (MAVLink failed).")

    prev_time = time.time()
    camera_index = 0  # Текущая камера
    fire_alert_cooldown = 0  # Анти-спам (1 раз в 5 сек)

    while True:
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if current_time != prev_time else 0
        prev_time = current_time

        # === Чтение кадра с текущей камеры ===
        cap = caps[camera_index]
        if cap is None:
            camera_index = (camera_index + 1) % CAMERA_COUNT
            continue

        ret, frame = cap.read()
        if not ret:
            logging.warning(f"Failed to grab frame from camera {camera_index}. Skipping.")
            camera_index = (camera_index + 1) % CAMERA_COUNT
            continue

        # === Обрезка центра кадра ===
        center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
        cropped_frame = crop_frame(frame, center_x, center_y, CROP_SIZE)

        # === YOLO детекция ===
        results = model(cropped_frame, imgsz=640, conf=0.5)
        fire_detected = False
        offset_x, offset_y = 0, 0

        if results and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().tolist()

            for i, (x1, y1, x2, y2) in enumerate(boxes):
                cls = classes[i]
                conf = confidences[i]

                # Предположим, что класс 0 = "fire", 1 = "smoke"
                if cls in [0, 1]:  # Огонь или дым
                    x = int((x1 + x2) / 2)
                    y = int((y1 + y2) / 2)
                    w = x2 - x1

                    cropped_center_x = CROP_SIZE // 2
                    offset_x = x - cropped_center_x
                    offset_y = y - cropped_center_y

                    # === Расчет расстояния и направления ===
                    distance = estimate_distance(w, focal_length_px=400, real_width_m=15)
                    direction = calculate_absolute_direction(
                        CAMERA_ANGLES[camera_index],
                        offset_x,
                        CROP_SIZE,
                        current_gps['heading']
                    )

                    # === Сохранение тревоги (с задержкой) ===
                    if current_time - fire_alert_cooldown > 5 and gps_lock:
                        save_fire_alert(
                            camera_id=camera_index,
                            obj_class="fire" if cls == 0 else "smoke",
                            confidence=conf,
                            distance_m=distance,
                            direction_deg=direction,
                            lat=current_gps['lat'],
                            lon=current_gps['lon']
                        )
                        fire_alert_cooldown = current_time

                    fire_detected = True
                    break  # Только первый объект

        # === Обновление GPS ===
        if mavlink_master:
            read_mavlink_gps(mavlink_master)

        # === Отображение ===
        frame_resized = cv2.resize(frame, (screen_width, screen_height))
        display_frame(frame_resized, cropped_frame, fps, camera_index, fire_detected)

        # === Смена камеры ===
        camera_index = (camera_index + 1) % CAMERA_COUNT

        # === Выход ===
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Exiting program.")
            break

        time.sleep(0.05)  # ~20ms между камерами → 5 кадров/сек на камеру

    # === Очистка ===
    for cap in caps:
        if cap:
            cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()