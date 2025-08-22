# 🌲 LesnikFPV — Autonomous Wildfire Detection System

> **Лесник. Видит дым — реагирует мгновенно.**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)
![YOLO](https://img.shields.io/badge/Model-YOLOv8-orange)
![Platform](https://img.shields.io/badge/Platform-Raspberry_Pi_5-orange)

**LesnikFPV** — open-source система обнаружения лесных пожаров на базе дронов и Raspberry Pi.  
Использует **4 камеры (360°)**, **YOLO-детекцию** и **MAVLink-телеметрию**, чтобы находить дым и огонь в реальном времени и автоматически определять их GPS-координаты.

🚀 Предназначена для автономного патрулирования лесов, интеграции с наземными станциями и быстрой реакции спасательных служб.

---
![LesnikFPV в QGIS](https://github.com/VohminV/LesnikFPV/raw/main/demo.jpg)
---
## 🎯 Возможности

- ✅ **Обнаружение дыма и огня** в реальном времени (YOLO + RKNN)
- 📸 **4 камеры** с наклоном 30° — охват 360°
- 🌍 **Точная геопривязка очага** (GPS + курс + расстояние)
- 📡 **Выход в GeoJSON** — готов к визуализации в **QGIS**
- 📻 Поддержка **LoRa-передачи алертов** на наземную станцию
- 🧭 Использование **MAVLink** для получения GPS, курса и высоты
- 🖼 Визуализация на борту: FPS, камера, статус
- 🔧 Гибкая калибровка: фокус, FOV, угол наклона

---

## 🖼 Архитектура системы
```
4 камеры (360°) → Попеременный захват → YOLO детекция
       ↓
Расчёт: расстояние + направление
       ↓
Геопривязка (drone GPS + offset)
       ↓
Сохранение: fire_alert.geojson
       ↓
[LoRa] → Наземная станция → QGIS / Оповещение
```
---

## 🛠 Технологии

- **Язык**: Python 3.8+
- **Модель**: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (поддержка `.pt`, `.rknn`)
- **Видео**: OpenCV + `cv2.VideoCapture`
- **Геодезия**: прямая геодезическая задача (на основе сферы Земли)
- **Данные дрона**: **MAVLink** (через `/dev/serial0`)
- **Передача данных**: **LoRa** (через UART)
- **Визуализация**: QGIS (с автообновлением GeoJSON)
- **Платформа**: Raspberry Pi 5 (рекомендуется)

---

## 📂 Структура проекта 
```
LesnikFPV/
├── fire_detection_geojson.py   # Основной скрипт (360° + GeoJSON)
├── detections.log              # Лог детекций
├── fire_alert.geojson          # Выходной файл для QGIS
├── models                      # YOLO-модель
├── requirements.txt            # Зависимости
└── README.md
```

---

## 📤 Формат вывода: fire_alert.geojson
```
{
  "type": "FeatureCollection",
  "name": "fire_alert",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "class": "smoke",
        "confidence": 0.78,
        "distance_m": 45.2,
        "direction": 135.6,
        "camera_id": 1,
        "drone_lat": 55.755821,
        "drone_lon": 37.617632,
        "time": "14:23:18"
      },
      "geometry": {
        "type": "Point",
        "coordinates": [37.618123, 55.755410]
      }
    }
  ]
}
```
---

## 📊 Интеграция с QGIS

LesnikFPV автоматически генерирует файл fire_alert.geojson, который можно напрямую открыть в QGIS для визуализации местоположения пожара.

✅ Как подключить 

    В QGIS:
        Слой → Добавить слой → Добавить векторный слой
        Выбери файл fire_alert.geojson
         
🌍 Добавление подложки (карты) 

По умолчанию фон белый. Чтобы увидеть местность: 

    Установите плагин QuickMapServices (в Модули → Управления модулями..)
    Добавьте фон (Интернет → QuickMapServices → Панель поиска справа, ввод OpenStreetMap)

---

## 📡 Передача по LoRa (опционально) 

Система может отправлять fire_alert.geojson через LoRa-модуль.

---

## 📄 Лицензия 

MIT License

---

## 🙌 Автор 

Разработчик Вохмин Виктор
Для настоящих лесников — тех, кто охраняет лес каждый день. 

🌲 Помогаем лесу — технологиями будущего.

🔧 Open-source. Open-hardware. Open-mind. 
