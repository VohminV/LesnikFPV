import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QHBoxLayout, QVBoxLayout, QWidget, QSlider
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from ultralytics import YOLO
import torch

# Проверка устройства
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Используется устройство: {device}")

# Загрузка модели YOLO
model = YOLO('best.pt')  # Можно использовать yolov8s.pt и т.д.
model.to(device)


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage, str)  # Сигнал: изображение + текст (FPS)

    def __init__(self):
        super().__init__()
        self.running = False
        self.video_path = None

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.change_pixmap_signal.emit(QImage(), "Ошибка: не удалось открыть видео.")
            return

        fps_counter = 0
        start_time = cv2.getTickCount()

        while self.running:
            ret, frame = cap.read()
            if not ret:
                self.change_pixmap_signal.emit(QImage(), "Видео завершено.")
                self.stop()
                break

            # Изменение размера кадра до 640x640
            frame_resized = cv2.resize(frame, (640, 640))

            # Выполнение детекции
            results = model(frame_resized, verbose=False)
            annotated_frame = results[0].plot()  # Рисуем bounding box'ы

            # Подсчёт FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                current = cv2.getTickCount()
                time_elapsed = (current - start_time) / cv2.getTickFrequency()
                fps = fps_counter / time_elapsed
                fps_text = f"FPS: {fps:.1f}"
            else:
                fps_text = f"FPS: {fps_counter / ((cv2.getTickCount() - start_time) / cv2.getTickFrequency()):.1f}"

            # Конвертация OpenCV -> QImage
            rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            scaled_image = qt_image.scaled(640, 640, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            self.change_pixmap_signal.emit(scaled_image, fps_text)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 Видео Анализ с GUI (PyQt5)")
        self.setGeometry(100, 100, 800, 700)
        self.setStyleSheet("background-color: #2c3e50; color: white;")

        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Элементы интерфейса
        self.video_label = QLabel("Здесь будет видео", self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 640)
        self.video_label.setStyleSheet("QLabel { background-color: #34495e; border: 1px solid #95a5a6; }")

        self.fps_label = QLabel("FPS: —", self)
        self.fps_label.setStyleSheet("font-size: 16px; font-weight: bold;")

        self.btn_open = QPushButton("📁 Выбрать видео", self)
        self.btn_open.setStyleSheet(self.button_style())
        self.btn_open.clicked.connect(self.open_file)

        self.btn_toggle = QPushButton("▶️ Старт", self)
        self.btn_toggle.setStyleSheet(self.button_style())
        self.btn_toggle.clicked.connect(self.toggle_playback)
        self.btn_toggle.setEnabled(False)

        # Макет
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.btn_open)
        control_layout.addWidget(self.btn_toggle)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.fps_label)
        layout.addLayout(control_layout)

        central_widget.setLayout(layout)

        # Поток обработки видео
        self.thread = VideoThread()

        # Подключение сигнала
        self.thread.change_pixmap_signal.connect(self.update_image)

    def button_style(self):
        return """
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1f618d;
            }
        """

    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Открыть видео", "", "Видео файлы (*.mp4 *.avi *.mov *.mkv)")
        if filename:
            self.thread.video_path = filename
            self.btn_toggle.setEnabled(True)
            self.video_label.setText(f"Видео загружено:\n{filename.split('/')[-1]}")

    def toggle_playback(self):
        if self.thread.running:
            self.thread.stop()
            self.btn_toggle.setText("▶️ Старт")
        else:
            self.thread.start()
            self.btn_toggle.setText("⏹ Стоп")

    def update_image(self, qt_image, fps_text):
        if qt_image.isNull():
            self.video_label.setText(fps_text)
        else:
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))
        self.fps_label.setText(fps_text)


# Запуск приложения
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())