import cv2
import numpy as np
import time
import json
import os
import math
from ultralytics import YOLO
from collections import defaultdict, deque
import logging
from pymavlink import mavutil

# === Настройка логирования ===
logging.basicConfig(filename='detections.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# === Загрузка модели ===
model = YOLO('/home/orangepi/Documents/YOLO/best_rknn_model')  # Убедись, что путь верный

# === Параметры отображения ===
screen_width = 720
screen_height = 576
cv2.namedWindow("Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# === Настройки камер (360° охват) ===
CAMERA_COUNT = 4
CAMERA_IDS = [0, 1, 2, 3]  # USB-камеры: 0=вперёд, 1=вправо, 2=назад, 3=влево
CAMERA_ANGLES = [0, 90, 180, 270]  # Направление камеры относительно корпуса БПЛА

# === Параметры обработки ===
CROP_SIZE = 320  # Размер центрального кропа для YOLO

# === GPS и MAVLink ===
current_gps = {'lat': 0.0, 'lon': 0.0, 'alt': 0.0, 'heading': 0.0}
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
                current_gps['lat'] = msg.lat * 1e-7  # градусы
                current_gps['lon'] = msg.lon * 1e-7
                current_gps['alt'] = msg.alt * 1e-3  # метры
                gps_lock = True
        elif msg.get_type() == 'VFR_HUD':
            current_gps['heading'] = msg.heading  # 0–360°

# === Кроп центра кадра ===
def crop_frame(frame, center_x, center_y, size):
    h, w = frame.shape[:2]
    x1 = max(center_x - size // 2, 0)
    x2 = min(center_x + size // 2, w)
    y1 = max(center_y - size // 2, 0)
    y2 = min(center_y + size // 2, h)
    return frame[y1:y2, x1:x2].copy()

# === Расчёт новых GPS-координат по расстоянию и направлению ===
def add_distance_at_angle(lat, lon, distance_m, bearing_deg):
    """
    Сдвигает точку (lat, lon) на distance_m метров в направлении bearing_deg
    Возвращает (new_lat, new_lon)
    """
    R = 6371000  # радиус Земли в метрах
    bearing = math.radians(bearing_deg)

    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    new_lat = math.asin(
        math.sin(lat_rad) * math.cos(distance_m / R) +
        math.cos(lat_rad) * math.sin(distance_m / R) * math.cos(bearing)
    )

    new_lon = lon_rad + math.atan2(
        math.sin(bearing) * math.sin(distance_m / R) * math.cos(lat_rad),
        math.cos(distance_m / R) - math.sin(lat_rad) * math.sin(new_lat)
    )

    return math.degrees(new_lat), math.degrees(new_lon)

# === Сохранение в GeoJSON (готово к QGIS) ===
def save_fire_geojson(camera_id, obj_class, confidence, distance_m, direction_deg, drone_lat, drone_lon):
    """
    Сохраняет обнаружение пожара в формате GeoJSON
    QGIS сможет открыть и обновлять этот файл в реальном времени
    """
    try:
        fire_lat, fire_lon = add_distance_at_angle(drone_lat, drone_lon, distance_m, direction_deg)
    except Exception as e:
        logging.warning(f"Geo calculation failed: {e}")
        fire_lat, fire_lon = drone_lat, drone_lon  # fallback

    geojson = {
        "type": "FeatureCollection",
        "name": "fire_alert",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "class": obj_class,
                    "confidence": round(confidence, 2),
                    "distance_m": round(distance_m, 1),
                    "direction": round(direction_deg, 1),
                    "camera_id": camera_id,
                    "drone_lat": round(drone_lat, 6),
                    "drone_lon": round(drone_lon, 6),
                    "timestamp": time.time(),
                    "time": time.strftime("%H:%M:%S", time.localtime()),
                    "icon": "fire" if obj_class == "fire" else "smoke"
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [fire_lon, fire_lat]  # GeoJSON: [lon, lat]
                }
            }
        ]
    }

    tmp_filename = 'fire_alert_tmp.geojson'
    final_filename = 'fire_alert.geojson'
    try:
        with open(tmp_filename, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, indent=2, ensure_ascii=False)
        os.replace(tmp_filename, final_filename)
        logging.info(f"✅ Fire alert saved: {obj_class} at {fire_lat:.6f}, {fire_lon:.6f}")
    except Exception as e:
        logging.error(f"❌ Failed to save GeoJSON: {e}")

# === Оценка расстояния до объекта ===
def estimate_distance(box_width_px, focal_length_px=400, real_width_m=15):
    """
    Простая триангуляция: чем больше объект в кадре — тем ближе
    focal_length_px — подбирается по калибровке камеры
    real_width_m — предполагаемая ширина дыма/огня (можно усреднить)
    """
    if box_width_px <= 0:
        return float('inf')
    return (focal_length_px * real_width_m) / box_width_px

# === Расчёт абсолютного направления пожара (от севера) ===
def calculate_absolute_direction(camera_azimuth, offset_x, frame_width, drone_heading):
    """
    camera_azimuth: физическое направление камеры (0,90,180,270)
    offset_x: смещение объекта от центра кадра (в пикселях)
    frame_width: ширина кропа (например, 320)
    drone_heading: курс БПЛА (0–360°)
    Возвращает: абсолютный азимут на объект (от севера)
    """
    fov_deg = 60  # Пример: угол обзора камеры (подбери под свою оптику)
    angle_per_pixel = fov_deg / frame_width
    offset_angle = offset_x * angle_per_pixel  # градусы отклонения от центра

    absolute_camera_dir = (camera_azimuth + drone_heading) % 360
    object_direction = (absolute_camera_dir + offset_angle) % 360
    return object_direction

# === Отображение на экране ===
def display_frame(frame_resized, cropped_frame, fps, camera_id, fire_detected):
    preview_size = 200
    cropped_resized = cv2.resize(cropped_frame, (preview_size, preview_size))
    y_offset = screen_height - preview_size
    x_offset = screen_width - preview_size
    frame_resized[y_offset:y_offset + preview_size, x_offset:x_offset + preview_size] = cropped_resized

    cv2.putText(frame_resized, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame_resized, f"Cam: {camera_id}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
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
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            logging.error(f"❌ Camera {cam_id} failed to open.")
            caps.append(None)
        else:
            caps.append(cap)

    # === Подключение к полётному контроллеру ===
    mavlink_master = connect_mavlink()
    if not mavlink_master:
        logging.warning("⚠️ MAVLink not connected. Running without GPS.")

    prev_time = time.time()
    camera_index = 0
    fire_alert_cooldown = 0  # Анти-спам: 1 алерт в 5 сек

    while True:
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        # === Захват кадра с текущей камеры ===
        cap = caps[camera_index]
        if cap is None:
            camera_index = (camera_index + 1) % CAMERA_COUNT
            continue

        ret, frame = cap.read()
        if not ret:
            logging.warning(f"⚠️ Failed to grab frame from camera {camera_index}")
            camera_index = (camera_index + 1) % CAMERA_COUNT
            continue

        # === Обрезка центра кадра для YOLO ===
        center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
        cropped_frame = crop_frame(frame, center_x, center_y, CROP_SIZE)

        # === YOLO детекция ===
        results = model(cropped_frame, imgsz=320, conf=0.5)
        fire_detected = False

        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().tolist()

            for i, (x1, y1, x2, y2) in enumerate(boxes):
                cls = classes[i]
                conf = confidences[i]

                # Классы: 0=огонь, 1=дым (уточни под свою модель)
                if cls in [0, 1]:
                    obj_class = "fire" if cls == 0 else "smoke"
                    w = x2 - x1
                    x = int((x1 + x2) / 2)

                    # === Расчёт расстояния и направления ===
                    distance = estimate_distance(w, focal_length_px=400, real_width_m=15)
                    direction = calculate_absolute_direction(
                        CAMERA_ANGLES[camera_index],
                        x - CROP_SIZE // 2,  # offset_x
                        CROP_SIZE,
                        current_gps['heading']
                    )

                    # === Сохранение в GeoJSON (если GPS есть и прошло 5 сек) ===
                    if current_time - fire_alert_cooldown > 5 and gps_lock:
                        save_fire_geojson(
                            camera_id=camera_index,
                            obj_class=obj_class,
                            confidence=conf,
                            distance_m=distance,
                            direction_deg=direction,
                            drone_lat=current_gps['lat'],
                            drone_lon=current_gps['lon']
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

        # === Переключение на следующую камеру ===
        camera_index = (camera_index + 1) % CAMERA_COUNT

        # === Выход по 'q' ===
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("🛑 Exiting program.")
            break

        time.sleep(0.05)  # ~20 мс на камеру → 5 FPS на камеру

    # === Очистка ресурсов ===
    for cap in caps:
        if cap:
            cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()