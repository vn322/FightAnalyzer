# app.py — финальная версия
import sys
import os
import cv2
import numpy as np
import pandas as pd
from collections import deque
import math
from ultralytics import YOLO
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QSpinBox, QComboBox, QProgressBar,
    QTableWidget, QTableWidgetItem, QTabWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
import matplotlib
matplotlib.use('Agg')  # без GUI
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Настройки цветов
COLORS = {
    'left': (0, 0, 255),       # BGR — красный (точки)
    'right': (255, 0, 0),      # синий (точки)
    'candidate': (255, 255, 0),# жёлтый (кандидаты)
    'left_box': (0, 0, 180),   # ✅ светло-красный (тонкий бокс)
    'right_box': (180, 0, 0),  # ✅ светло-синий (тонкий бокс)
}

ZONE_NAMES = [
    'top-left', 'top-center', 'top-right',
    'mid-left', 'center', 'mid-right',
    'bottom-left', 'bottom-center', 'bottom-right'
]

class FightAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🥋 FightAnalyzer — Изменение дистанции")
        self.resize(1400, 900)
        
        # Данные
        self.video_path = None
        self.candidates = []
        self.left_idx = None
        self.right_idx = None
        self.scale_px_per_m = None
        self.df = None
        self.yolo = None
        
        # Начальные координаты центров
        self.cx_l = None
        self.cy_l = None
        self.cx_r = None
        self.cy_r = None
        
        self._init_ui()
        self._load_models()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Ввод роста — компактно, без пустых промежутков
        h_layout = QHBoxLayout()
        
        # Левый спортсмен
        left_group = QHBoxLayout()
        left_group.addWidget(QLabel("Рост левого (см):"))
        self.height_left = QSpinBox()
        self.height_left.setRange(100, 250)
        self.height_left.setValue(175)
        self.height_left.setFixedWidth(80)  # ✅ фиксированная ширина
        left_group.addWidget(self.height_left)
        h_layout.addLayout(left_group)
        h_layout.addSpacing(20)  # небольшой отступ

        # Правый спортсмен
        right_group = QHBoxLayout()
        right_group.addWidget(QLabel("Рост правого (см):"))
        self.height_right = QSpinBox()
        self.height_right.setRange(100, 250)
        self.height_right.setValue(175)
        self.height_right.setFixedWidth(80)  # ✅ фиксированная ширина
        right_group.addWidget(self.height_right)
        h_layout.addLayout(right_group)
        h_layout.addStretch()

        self.btn_load = QPushButton("📂 Загрузить видео")
        self.btn_load.setMinimumHeight(40)
        self.btn_load.clicked.connect(self.load_video)
        h_layout.addWidget(self.btn_load)
        layout.addLayout(h_layout)

        # Строка с масштабом (появится после калибровки)
        self.scale_label = QLabel("Масштаб: —")
        self.scale_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        layout.addWidget(self.scale_label)

        # Визуализация первого кадра
        self.image_label = QLabel("Изображение не загружено")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(400)
        layout.addWidget(self.image_label)

        # Выбор спортсменов — тоже компактнее
        h_layout2 = QHBoxLayout()
        h_layout2.addWidget(QLabel("Левый:"))
        self.combo_left = QComboBox()
        self.combo_left.setFixedWidth(100)
        h_layout2.addWidget(self.combo_left)
        h_layout2.addSpacing(20)

        h_layout2.addWidget(QLabel("Правый:"))
        self.combo_right = QComboBox()
        self.combo_right.setFixedWidth(100)
        h_layout2.addWidget(self.combo_right)
        h_layout2.addSpacing(20)

        self.btn_analyze = QPushButton("🚀 Запустить анализ")
        self.btn_analyze.setMinimumHeight(40)
        self.btn_analyze.clicked.connect(self.start_analysis)
        self.btn_analyze.setEnabled(False)
        h_layout2.addWidget(self.btn_analyze)
        h_layout2.addStretch()
        layout.addLayout(h_layout2)

        # Прогресс
        self.progress = QProgressBar()
        self.progress.setMinimumHeight(30)
        layout.addWidget(self.progress)

        # Вкладки: данные, графики
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Таблица — 11 столбцов, включая stance_width_cm
        self.table = QTableWidget(0, 11)
        self.table.setHorizontalHeaderLabels([
            'frame', 'time_sec', 'dist_lr_m', 'v_left', 'a_left',
            'v_right', 'a_right', 'x_left', 'y_left', 'zone_left', 'stance_width_cm'
        ])
        self.tabs.addTab(self.table, "📊 Данные")

        # Графики
        self.plot_widget = QWidget()
        plot_layout = QVBoxLayout(self.plot_widget)
        self.canvas_speed = FigureCanvas(plt.Figure(figsize=(6, 3)))
        self.canvas_dist = FigureCanvas(plt.Figure(figsize=(6, 3)))
        plot_layout.addWidget(self.canvas_speed)
        plot_layout.addWidget(self.canvas_dist)
        self.tabs.addTab(self.plot_widget, "📈 Графики")

        # Кнопки экспорта
        h_layout3 = QHBoxLayout()
        self.btn_csv = QPushButton("📥 CSV")
        self.btn_csv.setMinimumHeight(40)
        self.btn_csv.clicked.connect(self.export_csv)
        self.btn_csv.setEnabled(False)
        h_layout3.addWidget(self.btn_csv)

        self.btn_video = QPushButton("🎥 Видео")
        self.btn_video.setMinimumHeight(40)
        self.btn_video.clicked.connect(self.export_video)
        self.btn_video.setEnabled(False)
        h_layout3.addWidget(self.btn_video)
        h_layout3.addStretch()
        layout.addLayout(h_layout3)

    def _load_models(self):
        try:
            self.yolo = YOLO("yolov8n.pt")
        except Exception as e:
            print(f"YOLO error: {e}")
            self.yolo = None

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите видео", "", "Video files (*.mp4 *.avi *.mov)"
        )
        if not path:
            return

        self.video_path = path
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return

        # Детекция первого кадра
        self.candidates = []
        if self.yolo:
            try:
                results = self.yolo(frame, classes=[0], conf=0.3)
                boxes = results[0].boxes.xyxy.cpu().numpy()
                self.candidates = self.filter_human_boxes(boxes, frame.shape)
            except:
                pass

        if not self.candidates:
            # fallback HOG
            self.candidates = self.hog_detect_people(frame)

        # Визуализация
        vis = frame.copy()
        for i, (x1, y1, x2, y2, cx, cy) in enumerate(self.candidates):
            cv2.rectangle(vis, (x1, y1), (x2, y2), COLORS['candidate'], 2)
            cv2.putText(vis, f"C{i+1}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['candidate'], 2)

        self._show_image(vis)
        self.combo_left.clear()
        self.combo_right.clear()
        for i in range(len(self.candidates)):
            self.combo_left.addItem(f"C{i+1}")
            self.combo_right.addItem(f"C{i+1}")
        if len(self.candidates) >= 2:
            self.combo_left.setCurrentIndex(0)
            self.combo_right.setCurrentIndex(1)
        self.btn_analyze.setEnabled(True)

    def filter_human_boxes(self, boxes, frame_shape, min_conf=0.3):
        h_img, w_img = frame_shape[:2]
        candidates = []
        min_height = 0.15 * h_img
        for box in boxes:
            if len(box) == 6:
                x1, y1, x2, y2, conf, _ = box
            else:
                x1, y1, x2, y2 = box
                conf = 1.0
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w_box = x2 - x1
            h_box = y2 - y1
            if h_box < min_height or conf < min_conf:
                continue
            aspect = w_box / h_box if h_box > 0 else 0
            if 0.2 <= aspect <= 1.0:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                candidates.append((x1, y1, x2, y2, cx, cy))
        return candidates

    def hog_detect_people(self, frame):
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        boxes, _ = hog.detectMultiScale(frame, winStride=(8,8), padding=(16,16), scale=1.05)
        if len(boxes) == 0:
            return []
        h_img = frame.shape[0]
        valid = []
        for (x, y, w, h) in boxes:
            if h > 0.15 * h_img and 0.2 < w/h < 1.0:
                valid.append((x, y, x+w, y+h, x+w/2, y+h/2))
        return valid

    def _show_image(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio
        ))

    def compute_velocity_acceleration(self, history):
        if len(history) < 3:
            return 0.0, 0.0
        xs = np.array([p[0] for p in history], dtype=np.float64)
        ys = np.array([p[1] for p in history], dtype=np.float64)
        ts = np.array([p[2] for p in history], dtype=np.float64)

        vx = np.gradient(xs, ts)
        vy = np.gradient(ys, ts)
        v_px_per_s = np.sqrt(vx[-1]**2 + vy[-1]**2)

        ax = np.gradient(vx, ts)
        ay = np.gradient(vy, ts)
        a_px_per_s2 = np.sqrt(ax[-1]**2 + ay[-1]**2)

        # 🔥 Делители — как в analysis2.csv
        v_mps = v_px_per_s / 100.0
        a_mps2 = a_px_per_s2 / 10000.0

        if len(vx) > 1 and v_mps < np.sqrt((vx[-2]/100.0)**2 + (vy[-2]/100.0)**2):
            a_mps2 = -abs(a_mps2)

        return float(v_mps), float(a_mps2)

    def pixel_to_meter(self, pixel_dist):
        return pixel_dist / self.scale_px_per_m if self.scale_px_per_m else 0.0

    def get_zone(self, x, y, w, h):
        zones = [
            (0, 0, w//3, h//3), (w//3, 0, 2*w//3, h//3), (2*w//3, 0, w, h//3),
            (0, h//3, w//3, 2*h//3), (w//3, h//3, 2*w//3, 2*h//3), (2*w//3, h//3, w, 2*h//3),
            (0, 2*h//3, w//3, h), (w//3, 2*h//3, 2*w//3, h), (2*w//3, 2*h//3, w, h),
        ]
        for i, (x1, y1, x2, y2) in enumerate(zones):
            if x1 <= x < x2 and y1 <= y < y2:
                return ZONE_NAMES[i]
        return 'center'

    def start_analysis(self):
        left_idx = self.combo_left.currentIndex()
        right_idx = self.combo_right.currentIndex()
        if left_idx == right_idx:
            return

        self.left_idx = left_idx
        self.right_idx = right_idx

        # Получаем боксы и центры
        x1_l, y1_l, x2_l, y2_l, cx_l, cy_l = self.candidates[left_idx]
        x1_r, y1_r, x2_r, y2_r, cx_r, cy_r = self.candidates[right_idx]

        # Сохраняем начальные центры
        self.cx_l = cx_l
        self.cy_l = cy_l
        self.cx_r = cx_r
        self.cy_r = cy_r

        # Калибровка масштаба по высоте бокса
        scale_l = (y2_l - y1_l) / (self.height_left.value() * 0.8 / 100.0)
        scale_r = (y2_r - y1_r) / (self.height_right.value() * 0.8 / 100.0)
        self.scale_px_per_m = (scale_l + scale_r) / 2

        # ✅ Обновляем масштаб на экране
        self.scale_label.setText(f"Масштаб: 1 м = {self.scale_px_per_m:.1f} px")

        # Анализ
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w, h = int(cap.get(3)), int(cap.get(4))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        pos_history = {'left': deque(maxlen=5), 'right': deque(maxlen=5)}
        data_rows = []

        # Открываем видео для записи
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.ann_path = os.path.join(os.path.dirname(self.video_path), "annotated.mp4")
        out = cv2.VideoWriter(self.ann_path, fourcc, fps, (w, h))

        self.progress.setMaximum(total_frames)
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # Детекция и трекинг
            cands = []
            if self.yolo:
                try:
                    res = self.yolo(frame, classes=[0], conf=0.3)
                    boxes = res[0].boxes.xyxy.cpu().numpy()
                    cands = self.filter_human_boxes(boxes, frame.shape)
                except:
                    pass
            if not cands:
                cands = self.hog_detect_people(frame)

            def find_nearest(cands, tx, ty):
                best, best_d = None, float('inf')
                for box in cands:
                    cx, cy = box[4], box[5]
                    d = (cx - tx)**2 + (cy - ty)**2
                    if d < best_d and d < 300**2:
                        best_d, best = d, box
                return best

            left_box = find_nearest(cands, self.cx_l, self.cy_l)
            right_box = find_nearest(cands, self.cx_r, self.cy_r)
            if left_box:
                _, _, _, _, self.cx_l, self.cy_l = left_box
            if right_box:
                _, _, _, _, self.cx_r, self.cy_r = right_box

            # Расчёты
            dt = 1.0 / fps
            pos_history['left'].append((self.cx_l, self.cy_l, frame_idx * dt))
            pos_history['right'].append((self.cx_r, self.cy_r, frame_idx * dt))

            v_l, a_l = self.compute_velocity_acceleration(pos_history['left'])
            v_r, a_r = self.compute_velocity_acceleration(pos_history['right'])

            dist_m = self.pixel_to_meter(math.hypot(self.cx_l - self.cx_r, self.cy_l - self.cy_r))
            zone_l = self.get_zone(self.cx_l, self.cy_l, w, h)

            # ✅ Ширина стойки (в см) — по ширине бокса
            stance_width_cm = 0.0
            if left_box:
                x1, y1, x2, y2, _, _ = left_box
                width_px = x2 - x1
                width_m = width_px / self.scale_px_per_m
                stance_width_cm = width_m * 100  # метры → сантиметры

            data_rows.append([
                frame_idx, frame_idx * dt, dist_m,
                v_l, a_l, v_r, a_r,
                self.cx_l, self.cy_l, zone_l,
                round(stance_width_cm, 1)  # ✅ добавлено
            ])

            # Аннотация видео
            ann = frame.copy()
            
            # 🔴 Светло-красный тонкий бокс для левого
            if left_box:
                x1, y1, x2, y2, _, _ = left_box
                cv2.rectangle(ann, (x1, y1), (x2, y2), COLORS['left_box'], 2)
            
            # 🔵 Светло-синий тонкий бокс для правого
            if right_box:
                x1, y1, x2, y2, _, _ = right_box
                cv2.rectangle(ann, (x1, y1), (x2, y2), COLORS['right_box'], 2)
            
            # Центры и линия
            cv2.circle(ann, (int(self.cx_l), int(self.cy_l)), 10, COLORS['left'], -1)
            cv2.circle(ann, (int(self.cx_r), int(self.cy_r)), 10, COLORS['right'], -1)
            cv2.line(ann, (int(self.cx_l), int(self.cy_l)), (int(self.cx_r), int(self.cy_r)), (0,0,0), 2)
            cv2.putText(ann, f"Dist: {dist_m:.2f}m", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(ann, f"L: v{v_l:.2f} a{a_l:.2f}", (int(self.cx_l)-40, int(self.cy_l)-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['left'], 2)
            cv2.putText(ann, f"R: v{v_r:.2f} a{a_r:.2f}", (int(self.cx_r)-40, int(self.cy_r)-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['right'], 2)
            cv2.putText(ann, f"Frame: {frame_idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(ann, f"Width: {stance_width_cm:.0f}cm", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            out.write(ann)
            self.progress.setValue(frame_idx + 1)
            QApplication.processEvents()

        cap.release()
        out.release()  # ✅ закрыт здесь, до экспорта

        # ✅ 11 столбцов теперь
        self.df = pd.DataFrame(data_rows, columns=[
            'frame', 'time_sec', 'dist_lr_m', 'v_left_mps', 'a_left_mps2',
            'v_right_mps', 'a_right_mps2', 'x_left', 'y_left', 'zone_left', 'stance_width_cm'
        ])

        # Заполнить таблицу
        self.table.setRowCount(len(self.df))
        for i in range(len(self.df)):
            for j in range(11):
                val = self.df.iloc[i, j]
                item = QTableWidgetItem(f"{val:.3f}" if isinstance(val, float) else str(val))
                self.table.setItem(i, j, item)

        # Графики
        self._plot_graphs()

        self.btn_csv.setEnabled(True)
        self.btn_video.setEnabled(True)

    def _plot_graphs(self):
        self.canvas_speed.figure.clear()
        ax = self.canvas_speed.figure.add_subplot(111)
        if self.df is not None:
            ax.plot(self.df['time_sec'], self.df['v_left_mps'], label='Left')
            ax.plot(self.df['time_sec'], self.df['v_right_mps'], label='Right')
            ax.set_xlabel('time_sec')
            ax.set_ylabel('v (m/s)')
            ax.legend()
            ax.grid(True)
        self.canvas_speed.draw()

        self.canvas_dist.figure.clear()
        ax = self.canvas_dist.figure.add_subplot(111)
        if self.df is not None:
            ax.plot(self.df['time_sec'], self.df['dist_lr_m'], 'g-')
            ax.set_xlabel('time_sec')
            ax.set_ylabel('dist_lr_m')
            ax.grid(True)
        self.canvas_dist.draw()

    def export_csv(self):
        if self.df is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить CSV", "analysis.csv", "CSV (*.csv)")
        if path:
            self.df.to_csv(path, index=False)

    def export_video(self):
        if self.df is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить видео", "annotated.mp4", "MP4 (*.mp4)")
        if not path:
            return

        # ✅ Копируем уже закрытый файл
        if os.path.exists(self.ann_path):
            import shutil
            shutil.copy(self.ann_path, path)
        else:
            print("⚠️ Аннотированное видео не найдено")

# Запуск
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FightAnalyzer()
    window.show()
    sys.exit(app.exec())