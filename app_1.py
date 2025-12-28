# app.py — финальная версия: move_toward по проекции, zone_left + zone_right,  PDF
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
    QTableWidget, QTableWidgetItem, QTabWidget, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from fpdf import FPDF
from io import BytesIO

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
        self.setWindowTitle("🥋 FightAnalyzer — Анализ тактики единоборств")
        self.resize(1400, 900)
        
        self.video_path = None
        self.candidates = []
        self.left_idx = None
        self.right_idx = None
        self.scale_px_per_m = None
        self.df = None
        self.yolo = None
        
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

        h_layout = QHBoxLayout()
        left_group = QHBoxLayout()
        left_group.addWidget(QLabel("Рост левого (см):"))
        self.height_left = QSpinBox()
        self.height_left.setRange(100, 250)
        self.height_left.setValue(175)
        self.height_left.setFixedWidth(80)
        left_group.addWidget(self.height_left)
        h_layout.addLayout(left_group)
        h_layout.addSpacing(20)

        right_group = QHBoxLayout()
        right_group.addWidget(QLabel("Рост правого (см):"))
        self.height_right = QSpinBox()
        self.height_right.setRange(100, 250)
        self.height_right.setValue(175)
        self.height_right.setFixedWidth(80)
        right_group.addWidget(self.height_right)
        h_layout.addLayout(right_group)
        h_layout.addStretch()

        self.btn_load = QPushButton("📂 Загрузить видео")
        self.btn_load.setMinimumHeight(40)
        self.btn_load.clicked.connect(self.load_video)
        h_layout.addWidget(self.btn_load)
        layout.addLayout(h_layout)

        self.scale_label = QLabel("Масштаб: —")
        self.scale_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        layout.addWidget(self.scale_label)

        self.image_label = QLabel("Изображение не загружено")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(400)
        layout.addWidget(self.image_label)

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

        self.progress = QProgressBar()
        self.progress.setMinimumHeight(30)
        layout.addWidget(self.progress)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.table = QTableWidget(0, 21)  # +1 колонка: zone_right
        self.table.setHorizontalHeaderLabels([
            'frame', 'time_sec', 'dist_lr_m', 'v_left', 'a_left',
            'v_right', 'a_right', 'x_left', 'y_left', 'zone_left',
            'zone_right',  # новая колонка
            'stance_width_left_cm', 'stance_width_right_cm',
            'approach', 'stance_width_change_rate', 'acceleration_profile',
            'displacement_left_m', 'displacement_right_m',
            'move_toward_left_m', 'move_toward_right_m',
            'move_toward_total_m'
        ])
        self.tabs.addTab(self.table, "📊 Данные")

        self.plot_widget = QWidget()
        plot_layout = QVBoxLayout(self.plot_widget)
        self.canvas_speed = FigureCanvas(plt.Figure(figsize=(6, 3)))
        self.canvas_dist = FigureCanvas(plt.Figure(figsize=(6, 3)))
        plot_layout.addWidget(self.canvas_speed)
        plot_layout.addWidget(self.canvas_dist)
        self.tabs.addTab(self.plot_widget, "📈 Графики")

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

        self.btn_pdf = QPushButton("📄 PDF-отчёт")
        self.btn_pdf.setMinimumHeight(40)
        self.btn_pdf.clicked.connect(self.export_pdf)
        self.btn_pdf.setEnabled(False)
        h_layout3.addWidget(self.btn_pdf)

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

        self.candidates = []
        if self.yolo:
            try:
                results = self.yolo(frame, classes=[0], conf=0.3)
                boxes = results[0].boxes.xyxy.cpu().numpy()
                self.candidates = self.filter_human_boxes(boxes, frame.shape)
            except:
                pass

        if not self.candidates:
            self.candidates = self.hog_detect_people(frame)

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

        x1_l, y1_l, x2_l, y2_l, cx_l, cy_l = self.candidates[left_idx]
        x1_r, y1_r, x2_r, y2_r, cx_r, cy_r = self.candidates[right_idx]

        self.cx_l = cx_l
        self.cy_l = cy_l
        self.cx_r = cx_r
        self.cy_r = cy_r

        scale_l = (y2_l - y1_l) / (self.height_left.value() * 0.8 / 100.0)
        scale_r = (y2_r - y1_r) / (self.height_right.value() * 0.8 / 100.0)
        self.scale_px_per_m = (scale_l + scale_r) / 2

        self.scale_label.setText(f"Масштаб: 1 м = {self.scale_px_per_m:.1f} px")

        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w, h = int(cap.get(3)), int(cap.get(4))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        pos_history = {'left': deque(maxlen=5), 'right': deque(maxlen=5)}
        stance_history = deque(maxlen=10)
        data_rows = []

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.ann_path = os.path.join(os.path.dirname(self.video_path), "annotated.mp4")
        out = cv2.VideoWriter(self.ann_path, fourcc, fps, (w, h))

        self.progress.setMaximum(total_frames)
        
        prev_cx_l, prev_cy_l = self.cx_l, self.cy_l
        prev_cx_r, prev_cy_r = self.cx_r, self.cy_r
        prev_dist = None

        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

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

            dt = 1.0 / fps
            pos_history['left'].append((self.cx_l, self.cy_l, frame_idx * dt))
            pos_history['right'].append((self.cx_r, self.cy_r, frame_idx * dt))

            v_l, a_l = self.compute_velocity_acceleration(pos_history['left'])
            v_r, a_r = self.compute_velocity_acceleration(pos_history['right'])

            dist_m = self.pixel_to_meter(math.hypot(self.cx_l - self.cx_r, self.cy_l - self.cy_r))
            zone_l = self.get_zone(self.cx_l, self.cy_l, w, h)
            zone_r = self.get_zone(self.cx_r, self.cy_r, w, h)  # ✅ Новая зона

            stance_width_left_cm = 0.0
            if left_box:
                x1, y1, x2, y2, _, _ = left_box
                width_px = x2 - x1
                width_m = width_px / self.scale_px_per_m
                stance_width_left_cm = width_m * 100

            stance_width_right_cm = 0.0
            if right_box:
                x1, y1, x2, y2, _, _ = right_box
                width_px = x2 - x1
                width_m = width_px / self.scale_px_per_m
                stance_width_right_cm = width_m * 100

            displacement_left_px = math.hypot(self.cx_l - prev_cx_l, self.cy_l - prev_cy_l)
            displacement_right_px = math.hypot(self.cx_r - prev_cx_r, self.cy_r - prev_cy_r)
            displacement_left_m = displacement_left_px / self.scale_px_per_m
            displacement_right_m = displacement_right_px / self.scale_px_per_m

            # ✅ Новая логика: move_toward = проекция смещения на линию к сопернику
            dx = self.cx_r - self.cx_l
            dy = self.cy_r - self.cy_l
            norm = math.hypot(dx, dy)
            if norm > 1:
                ux, uy = dx / norm, dy / norm
            else:
                ux, uy = 0, 0

            dLx = self.cx_l - prev_cx_l
            dLy = self.cy_l - prev_cy_l
            dRx = self.cx_r - prev_cx_r
            dRy = self.cy_r - prev_cy_r

            move_toward_left_m = (dLx * ux + dLy * uy) / self.scale_px_per_m
            move_toward_right_m = -(dRx * ux + dRy * uy) / self.scale_px_per_m

            approach = 1 if (move_toward_left_m + move_toward_right_m) > 0 else 0

            stance_history.append((stance_width_left_cm, frame_idx * dt))
            stance_width_change_rate = 0.0
            if len(stance_history) >= 2:
                widths = np.array([s[0] for s in stance_history])
                times = np.array([s[1] for s in stance_history])
                stance_width_change_rate = np.gradient(widths, times)[-1] if len(widths) > 1 else 0.0

            acc_profile = 0
            acc_abs = max(abs(a_l), abs(a_r))
            if acc_abs > 1.5:
                acc_profile = 2
            elif acc_abs > 0.8:
                acc_profile = 1

            move_toward_total_m = move_toward_left_m + move_toward_right_m

            data_rows.append([
                frame_idx, frame_idx * dt, dist_m,
                v_l, a_l, v_r, a_r,
                self.cx_l, self.cy_l, zone_l, zone_r,  # ✅ Добавлена zone_r
                stance_width_left_cm, stance_width_right_cm,
                approach,
                stance_width_change_rate,
                acc_profile,
                displacement_left_m,
                displacement_right_m,
                move_toward_left_m,
                move_toward_right_m,
                move_toward_total_m
            ])

            ann = frame.copy()
            if left_box:
                x1, y1, x2, y2, _, _ = left_box
                cv2.rectangle(ann, (x1, y1), (x2, y2), COLORS['left_box'], 2)
            if right_box:
                x1, y1, x2, y2, _, _ = right_box
                cv2.rectangle(ann, (x1, y1), (x2, y2), COLORS['right_box'], 2)
            cv2.circle(ann, (int(self.cx_l), int(self.cy_l)), 10, COLORS['left'], -1)
            cv2.circle(ann, (int(self.cx_r), int(self.cy_r)), 10, COLORS['right'], -1)
            cv2.line(ann, (int(self.cx_l), int(self.cy_l)), (int(self.cx_r), int(self.cy_r)), (0,0,0), 2)
            cv2.putText(ann, f"Dist: {dist_m:.2f}m", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(ann, f"L: v{v_l:.2f} a{a_l:.2f}", (int(self.cx_l)-40, int(self.cy_l)-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['left'], 2)
            cv2.putText(ann, f"R: v{v_r:.2f} a{a_r:.2f}", (int(self.cx_r)-40, int(self.cy_r)-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['right'], 2)
            cv2.putText(ann, f"Frame: {frame_idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(ann, f"Width L: {stance_width_left_cm:.0f}cm", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(ann, f"Width R: {stance_width_right_cm:.0f}cm", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(ann, "Ermakov.AV, 2025", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)

            out.write(ann)
            self.progress.setValue(frame_idx + 1)
            QApplication.processEvents()

            prev_cx_l, prev_cy_l = self.cx_l, self.cy_l
            prev_cx_r, prev_cy_r = self.cx_r, self.cy_r
            prev_dist = dist_m

        cap.release()
        out.release()

        self.df = pd.DataFrame(data_rows, columns=[
            'frame', 'time_sec', 'dist_lr_m', 'v_left_mps', 'a_left_mps2',
            'v_right_mps', 'a_right_mps2', 'x_left', 'y_left', 'zone_left', 'zone_right',  # ✅ Добавлена zone_right
            'stance_width_left_cm', 'stance_width_right_cm',
            'approach', 'stance_width_change_rate', 'acceleration_profile',
            'displacement_left_m', 'displacement_right_m',
            'move_toward_left_m', 'move_toward_right_m',
            'move_toward_total_m'
        ])

        self.table.setRowCount(len(self.df))
        for i in range(len(self.df)):
            for j in range(21):  # 21 колонка
                val = self.df.iloc[i, j]
                item = QTableWidgetItem(f"{val:.3f}" if isinstance(val, float) else str(int(val)) if isinstance(val, (int, np.integer)) else str(val))
                self.table.setItem(i, j, item)

        self._plot_graphs()
        self.btn_csv.setEnabled(True)
        self.btn_video.setEnabled(True)
        self.btn_pdf.setEnabled(True)

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

        if os.path.exists(self.ann_path):
            import shutil
            shutil.copy(self.ann_path, path)
        else:
            print("⚠️ Аннотированное видео не найдено")

    def export_pdf(self):
        if self.df is None:
            QMessageBox.warning(self, "Ошибка", "Сначала выполните анализ.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Сохранить PDF", "report.pdf", "PDF (*.pdf)")
        if not path:
            return

        try:
            import os
            script_dir = os.path.dirname(os.path.abspath(__file__))
            font_path = os.path.join(script_dir, "DejaVuSans.ttf")
            if not os.path.exists(font_path):
                raise FileNotFoundError(f"Шрифт не найден: {font_path}")

            from fpdf import FPDF
            pdf = FPDF()
            pdf.add_page()
            pdf.add_font("DejaVu", "", font_path, uni=True)
            pdf.set_font("DejaVu", size=14)
            pdf.cell(200, 10, txt="FightAnalyzer — Автоматический отчёт", ln=True, align='C')
            pdf.set_font("DejaVu", size=10)
            pdf.cell(200, 10, txt="Отчёт собран автоматически и требует критического осмысления и анализа экспертом", ln=True, align='C')
            pdf.ln(5)

            # 1. График скорости
            plt.figure(figsize=(8, 3))
            plt.plot(self.df['time_sec'], self.df['v_left_mps'], label='Левый', color='red')
            plt.plot(self.df['time_sec'], self.df['v_right_mps'], label='Правый', color='blue')
            plt.xlabel('Время (с)'); plt.ylabel('Скорость (м/с)')
            plt.title('Скорость спортсменов')
            plt.legend(); plt.grid(True)
            buf1 = BytesIO()
            plt.savefig(buf1, format='png', bbox_inches='tight')
            plt.close()

            # 2. График дистанции
            plt.figure(figsize=(8, 3))
            plt.plot(self.df['time_sec'], self.df['dist_lr_m'], 'g-')
            plt.xlabel('Время (с)'); plt.ylabel('Дистанция (м)')
            plt.title('Дистанция между спортсменами')
            plt.grid(True)
            buf2 = BytesIO()
            plt.savefig(buf2, format='png', bbox_inches='tight')
            plt.close()

            # 3. График move_toward — оба спортсмена на одном
            plt.figure(figsize=(8, 3))
            plt.plot(self.df['time_sec'], self.df['move_toward_left_m'],
                     label='Левый', color='red', linewidth=1.5, alpha=0.8, marker='.', markersize=2)
            plt.plot(self.df['time_sec'], self.df['move_toward_right_m'],
                     label='Правый', color='blue', linewidth=1.5, alpha=0.8, marker='.', markersize=2)
            plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
            plt.xlabel('Время (с)'); plt.ylabel('Смещение к сопернику (м)')
            plt.title('Тактическое смещение (атака/уход)')
            plt.legend(); plt.grid(True)
            buf3 = BytesIO()
            plt.savefig(buf3, format='png', bbox_inches='tight')
            plt.close()

            # 4. Ширина стойки — оба
            plt.figure(figsize=(8, 3))
            plt.plot(self.df['time_sec'], self.df['stance_width_left_cm'], 'orange', label='Левый')
            plt.plot(self.df['time_sec'], self.df['stance_width_right_cm'], 'purple', label='Правый')
            plt.xlabel('Время (с)'); plt.ylabel('Ширина стойки (см)')
            plt.title('Ширина стойки обоих спортсменов')
            plt.legend(); plt.grid(True)
            buf4 = BytesIO()
            plt.savefig(buf4, format='png', bbox_inches='tight')
            plt.close()

            # 5. Гистограмма скоростей
            plt.figure(figsize=(8, 3))
            plt.hist(self.df['v_left_mps'], bins=20, alpha=0.7, label='Левый', color='red')
            plt.hist(self.df['v_right_mps'], bins=20, alpha=0.7, label='Правый', color='blue')
            plt.xlabel('Скорость (м/с)'); plt.ylabel('Частота')
            plt.title('Распределение скоростей')
            plt.legend(); plt.grid(True)
            buf5 = BytesIO()
            plt.savefig(buf5, format='png', bbox_inches='tight')
            plt.close()

            # 🔍 АНАЛИЗ ДАННЫХ
            total_time = self.df['time_sec'].max()
            third = total_time / 3

            # Макс. ускорения
            a_left_max = self.df['a_left_mps2'].abs().max()
            a_right_max = self.df['a_right_mps2'].abs().max()

            # Атаки: ускорение ≥ 0.8·aₘₐₓ
            attack_left = (self.df['a_left_mps2'].abs() >= 0.8 * a_left_max)
            attack_right = (self.df['a_right_mps2'].abs() >= 0.8 * a_right_max)

            # Зональное время
            zone_time_left = self.df.groupby('zone_left').size() * (1/30)
            zone_time_right = self.df.groupby('zone_right').size() * (1/30)

            # 1. Описательная статистика — ТАБЛИЦА
            pdf.set_font("DejaVu", size=12)
            pdf.cell(200, 10, txt="1. Описательная статистика (дистанция, скорость, ускорение)", ln=True)
            pdf.set_font("DejaVu", size=9)

            stats = self.df[['dist_lr_m', 'v_left_mps', 'v_right_mps', 'a_left_mps2', 'a_right_mps2']].describe().round(2)
            stats = stats.drop('count')  # Удаляем count
            stats_data = stats.reset_index().values.tolist()
            col_widths = [30, 30, 30, 30, 30, 30]
            headers = ["", "dist_lr_m", "v_left_mps", "v_right_mps", "a_left_mps2", "a_right_mps2"]

            for i, h in enumerate(headers):
                pdf.cell(col_widths[i], 6, h, border=1, align='C')
            pdf.ln()
            for row in stats_data:
                pdf.cell(col_widths[0], 6, str(row[0]), border=1, align='L')  # index
                for j in range(1, 6):
                    pdf.cell(col_widths[j], 6, f"{row[j]:.2f}", border=1, align='R')
                pdf.ln()
            pdf.ln(5)

            # 2. Время в зонах и атаки
            pdf.set_font("DejaVu", size=12)
            pdf.cell(200, 10, txt="2. Оценка пространственной активности (зоны считаются относительно кадра)", ln=True)
            pdf.set_font("DejaVu", size=9)
            col_widths = [45, 25, 25, 25]
            headers = ["Зона", "Время", "Левый", "Правый"]
            for i, h in enumerate(headers):
                pdf.cell(col_widths[i], 6, h, border=1, align='C')
            pdf.ln()
            for zone in ZONE_NAMES:
                t_l = zone_time_left.get(zone, 0)
                t_r = zone_time_right.get(zone, 0)
                a_l = int(attack_left[self.df['zone_left'] == zone].sum())
                a_r = int(attack_right[self.df['zone_right'] == zone].sum())
                pdf.cell(col_widths[0], 6, zone, border=1)
                pdf.cell(col_widths[1], 6, f"{t_l:.1f}", border=1, align='R')
                pdf.cell(col_widths[2], 6, str(a_l), border=1, align='C')
                pdf.cell(col_widths[3], 6, str(a_r), border=1, align='C')
                pdf.ln()
            pdf.ln(5)

            # 3. Совпадение ускорения и направления — ЧИСЛА
            left_attack_phase = ((self.df['move_toward_left_m'] > 0) & attack_left).sum()
            left_retreat_phase = ((self.df['move_toward_left_m'] < 0) & attack_left).sum()
            left_defense_phase = ((self.df['move_toward_left_m'] < 0) & (~attack_left)).sum()

            right_attack_phase = ((self.df['move_toward_right_m'] > 0) & attack_right).sum()
            right_retreat_phase = ((self.df['move_toward_right_m'] < 0) & attack_right).sum()
            right_defense_phase = ((self.df['move_toward_right_m'] < 0) & (~attack_right)).sum()

            pdf.set_font("DejaVu", size=12)
            pdf.cell(200, 10, txt="3. Совпадение ускорения и направления маневрирования", ln=True)
            pdf.set_font("DejaVu", size=9)
            col_widths = [45, 30, 30, 30]
            headers = ["Спортсмен", "Атака", "Контр.", "Обор."]
            for i, h in enumerate(headers):
                pdf.cell(col_widths[i], 6, h, border=1, align='C')
            pdf.ln()
            pdf.cell(col_widths[0], 6, "Левый", border=1)
            pdf.cell(col_widths[1], 6, str(int(left_attack_phase)), border=1, align='C')
            pdf.cell(col_widths[2], 6, str(int(left_retreat_phase)), border=1, align='C')
            pdf.cell(col_widths[3], 6, str(int(left_defense_phase)), border=1, align='C')
            pdf.ln()
            pdf.cell(col_widths[0], 6, "Правый", border=1)
            pdf.cell(col_widths[1], 6, str(int(right_attack_phase)), border=1, align='C')
            pdf.cell(col_widths[2], 6, str(int(right_retreat_phase)), border=1, align='C')
            pdf.cell(col_widths[3], 6, str(int(right_defense_phase)), border=1, align='C')
            pdf.ln(5)

            # 4. Плотность боя
            attacks_per_min = {}
            for i, name in enumerate(['1/3', '2/3', '3/3', 'Всего']):
                if i < 3:
                    start = i * third
                    end = (i+1) * third
                    mask = (self.df['time_sec'] >= start) & (self.df['time_sec'] < end)
                    dur = third / 60
                else:
                    mask = slice(None)
                    dur = total_time / 60
                al = attack_left[mask].sum() / dur
                ar = attack_right[mask].sum() / dur
                attacks_per_min[name] = [round(al, 1), round(ar, 1)]

            pdf.set_font("DejaVu", size=12)
            pdf.cell(200, 10, txt="4. Плотность боя (атак/мин)", ln=True)
            pdf.set_font("DejaVu", size=9)
            col_widths = [40, 30, 30, 30]
            headers = ["Период", "Левый", "Правый", "Суммарно"]
            for i, h in enumerate(headers):
                pdf.cell(col_widths[i], 6, h, border=1, align='C')
            pdf.ln()
            for period, (al, ar) in attacks_per_min.items():
                pdf.cell(col_widths[0], 6, period, border=1)
                pdf.cell(col_widths[1], 6, f"{al:.1f}", border=1, align='R')
                pdf.cell(col_widths[2], 6, f"{ar:.1f}", border=1, align='R')
                pdf.cell(col_widths[3], 6, f"{al+ar:.1f}", border=1, align='R')
                pdf.ln()
            pdf.ln(5)

            # 5. Ширина стойки при атаке
            stance_left_attack = self.df.loc[attack_left, 'stance_width_left_cm']
            stance_right_attack = self.df.loc[attack_right, 'stance_width_right_cm']

            stance_left_stats = {
                'min': round(stance_left_attack.min(), 1) if len(stance_left_attack) > 0 else 0,
                'mean': round(stance_left_attack.mean(), 1) if len(stance_left_attack) > 0 else 0,
                'max': round(stance_left_attack.max(), 1) if len(stance_left_attack) > 0 else 0,
                'mode': round(stance_left_attack.mode().iloc[0], 1) if len(stance_left_attack) > 0 and not stance_left_attack.mode().empty else 0,
            }
            stance_right_stats = {
                'min': round(stance_right_attack.min(), 1) if len(stance_right_attack) > 0 else 0,
                'mean': round(stance_right_attack.mean(), 1) if len(stance_right_attack) > 0 else 0,
                'max': round(stance_right_attack.max(), 1) if len(stance_right_attack) > 0 else 0,
                'mode': round(stance_right_attack.mode().iloc[0], 1) if len(stance_right_attack) > 0 and not stance_right_attack.mode().empty else 0,
            }

            pdf.set_font("DejaVu", size=12)
            pdf.cell(200, 10, txt="5. Ширина стойки при атаке (см)", ln=True)
            pdf.set_font("DejaVu", size=9)
            col_widths = [40, 25, 25, 25, 25]
            headers = ["Спортсмен", "min", "mean", "max", "mode"]
            for i, h in enumerate(headers):
                pdf.cell(col_widths[i], 6, h, border=1, align='C')
            pdf.ln()
            pdf.cell(col_widths[0], 6, "Левый", border=1)
            pdf.cell(col_widths[1], 6, f"{stance_left_stats['min']}", border=1, align='R')
            pdf.cell(col_widths[2], 6, f"{stance_left_stats['mean']}", border=1, align='R')
            pdf.cell(col_widths[3], 6, f"{stance_left_stats['max']}", border=1, align='R')
            pdf.cell(col_widths[4], 6, f"{stance_left_stats['mode']}", border=1, align='R')
            pdf.ln()
            pdf.cell(col_widths[0], 6, "Правый", border=1)
            pdf.cell(col_widths[1], 6, f"{stance_right_stats['min']}", border=1, align='R')
            pdf.cell(col_widths[2], 6, f"{stance_right_stats['mean']}", border=1, align='R')
            pdf.cell(col_widths[3], 6, f"{stance_right_stats['max']}", border=1, align='R')
            pdf.cell(col_widths[4], 6, f"{stance_right_stats['mode']}", border=1, align='R')
            pdf.ln(5)

            # 6. Манера ведения боя по третям — для левого и правого
            tactics_left = []
            tactics_right = []
            for i in range(3):
                start = i * third
                end = (i+1) * third
                mask = (self.df['time_sec'] >= start) & (self.df['time_sec'] < end)
                
                att_l = ((self.df.loc[mask, 'move_toward_left_m'] > 0) & attack_left[mask]).sum()
                contr_l = ((self.df.loc[mask, 'move_toward_left_m'] < 0) & attack_left[mask]).sum()
                def_l = ((self.df.loc[mask, 'move_toward_left_m'] < 0) & (~attack_left[mask])).sum()
                total_l = att_l + contr_l + def_l
                if total_l > 0:
                    tactics_left.append([
                        f"{i+1}/3",
                        f"{int(att_l)}",
                        f"{int(contr_l)}",
                        f"{int(def_l)}"
                    ])

                att_r = ((self.df.loc[mask, 'move_toward_right_m'] > 0) & attack_right[mask]).sum()
                contr_r = ((self.df.loc[mask, 'move_toward_right_m'] < 0) & attack_right[mask]).sum()
                def_r = ((self.df.loc[mask, 'move_toward_right_m'] < 0) & (~attack_right[mask])).sum()
                total_r = att_r + contr_r + def_r
                if total_r > 0:
                    tactics_right.append([
                        f"{i+1}/3",
                        f"{int(att_r)}",
                        f"{int(contr_r)}",
                        f"{int(def_r)}"
                    ])

            pdf.set_font("DejaVu", size=12)
            pdf.cell(200, 10, txt="6.1 Атакующие действия в ходе боя — левый", ln=True)
            pdf.set_font("DejaVu", size=9)
            col_widths = [25, 25, 25, 25]
            headers = ["Треть", "Наступ.", "Контр.", "Отход."]
            for i, h in enumerate(headers):
                pdf.cell(col_widths[i], 6, h, border=1, align='C')
            pdf.ln()
            for row in tactics_left:
                for i, cell in enumerate(row):
                    pdf.cell(col_widths[i], 6, str(cell), border=1, align='C')
                pdf.ln()
            pdf.ln(5)

            pdf.set_font("DejaVu", size=12)
            pdf.cell(200, 10, txt="6.2 Атакующие действия в ходе боя — правый", ln=True)
            pdf.set_font("DejaVu", size=9)
            for i, h in enumerate(headers):
                pdf.cell(col_widths[i], 6, h, border=1, align='C')
            pdf.ln()
            for row in tactics_right:
                for i, cell in enumerate(row):
                    pdf.cell(col_widths[i], 6, str(cell), border=1, align='C')
                pdf.ln()
            pdf.ln(5)

            # 7. Текстовое описание — расширено
            v_left_mean = self.df['v_left_mps'].mean()
            v_right_mean = self.df['v_right_mps'].mean()
            move_left_total = self.df['move_toward_left_m'].sum()
            move_right_total = self.df['move_toward_right_m'].sum()
            dist_min = self.df['dist_lr_m'].min()
            dist_max = self.df['dist_lr_m'].max()

            # ✅ Дополнительный анализ для текстового описания
            # Атаки из зоны в зону
            attack_zones_left = self.df[attack_left][['zone_left', 'zone_right']].value_counts().reset_index()
            attack_zones_right = self.df[attack_right][['zone_left', 'zone_right']].value_counts().reset_index()
            if len(attack_zones_left) > 0:
                most_common_attack_left = attack_zones_left.iloc[0]
                from_zone_left = most_common_attack_left['zone_left']
                to_zone_left = most_common_attack_left['zone_right']
            else:
                from_zone_left = "нет"
                to_zone_left = "атак"

            if len(attack_zones_right) > 0:
                most_common_attack_right = attack_zones_right.iloc[0]
                from_zone_right = most_common_attack_right['zone_left']
                to_zone_right = most_common_attack_right['zone_right']
            else:
                from_zone_right = "нет"
                to_zone_right = "атак"

            # Уходы
            retreat_left = (self.df['move_toward_left_m'] < 0)
            retreat_zones_left = self.df[retreat_left][['zone_left', 'zone_right']].value_counts().reset_index()
            if len(retreat_zones_left) > 0:
                most_common_retreat_left = retreat_zones_left.iloc[0]
                from_retreat_left = most_common_retreat_left['zone_left']
                to_retreat_left = most_common_retreat_left['zone_right']
            else:
                from_retreat_left = "нет"
                to_retreat_left = "уход"

            retreat_right = (self.df['move_toward_right_m'] < 0)
            retreat_zones_right = self.df[retreat_right][['zone_left', 'zone_right']].value_counts().reset_index()
            if len(retreat_zones_right) > 0:
                most_common_retreat_right = retreat_zones_right.iloc[0]
                from_retreat_right = most_common_retreat_right['zone_left']
                to_retreat_right = most_common_retreat_right['zone_right']
            else:
                from_retreat_right = "нет"
                to_retreat_right = "уход"

            # Изменение ширины стойки при атаке
            stance_change_left = stance_left_attack.diff().mean()
            stance_change_right = stance_right_attack.diff().mean()

            # Активность по третям
            thirds = [0, total_time/3, 2*total_time/3, total_time]
            activity_left = []
            activity_right = []
            for i in range(3):
                start = thirds[i]
                end = thirds[i+1]
                mask = (self.df['time_sec'] >= start) & (self.df['time_sec'] < end)
                atk_l = attack_left[mask].sum()
                disp_l = self.df.loc[mask, 'move_toward_left_m'].abs().sum()
                atk_r = attack_right[mask].sum()
                disp_r = self.df.loc[mask, 'move_toward_right_m'].abs().sum()
                activity_left.append(atk_l + disp_l)
                activity_right.append(atk_r + disp_r)

            active_third_left = np.argmax(activity_left) + 1
            active_third_right = np.argmax(activity_right) + 1

            # Увеличение ширины стойки в течение времени
            stance_trend_left = np.polyfit(range(len(self.df)), self.df['stance_width_left_cm'], 1)[0]
            stance_trend_right = np.polyfit(range(len(self.df)), self.df['stance_width_right_cm'], 1)[0]

            # Количество атак в активной трети
            mask_active_left = (self.df['time_sec'] >= (active_third_left-1)*third) & (self.df['time_sec'] < active_third_left*third)
            atk_in_active_left = attack_left[mask_active_left].sum()
            mask_active_right = (self.df['time_sec'] >= (active_third_right-1)*third) & (self.df['time_sec'] < active_third_right*third)
            atk_in_active_right = attack_right[mask_active_right].sum()

            # Изменение ширины стойки в активной трети
            stance_change_in_active_left = self.df.loc[mask_active_left, 'stance_width_left_cm'].diff().mean()
            stance_change_in_active_right = self.df.loc[mask_active_right, 'stance_width_right_cm'].diff().mean()

            description = f"""Всегда сверяйте описание с видео и помните, что анализ фокусируется на тактических, а не технических особенностях.

• Средняя скорость: левый — {v_left_mean:.2f} м/с, правый — {v_right_mean:.2f} м/с.
• Макс. ускорение: левый — {a_left_max:.2f} м/с², правый — {a_right_max:.2f} м/с².
• Тактическое поведение: 
  — Левый сделал {'атакующих' if move_left_total > 0 else 'оборонительных'} движений на {abs(move_left_total):.2f} м.
  — Правый сделал {'атакующих' if move_right_total > 0 else 'оборонительных'} движений на {abs(move_right_total):.2f} м.
• Минимальная дистанция: {dist_min:.2f} м — моменты атак.
• Максимальная дистанция: {dist_max:.2f} м — моменты ухода.

• Маневрирование при атаках:
  — Левый чаще атаковал из зоны "{from_zone_left}" в зону "{to_zone_left}".
  — Правый чаще атаковал из зоны "{from_zone_right}" в зону "{to_zone_right}".

• Маневрирование при уходах:
  — Левый чаще уходил из зоны "{from_retreat_left}" в зону "{to_retreat_left}".
  — Правый чаще уходил из зоны "{from_retreat_right}" в зону "{to_retreat_right}".

• Ширина стойки при атаке (уменьшение может свидетельствовать об преимущестенно атакующем стиле, а уменьшение - контратакующем):
  — Левый: {'увеличивалась' if stance_change_left > 0 else 'уменьшалась' if stance_change_left < 0 else 'не изменилась'} на {abs(stance_change_left):.2f} см.
  — Правый: {'увеличивалась' if stance_change_right > 0 else 'уменьшалась' if stance_change_right < 0 else 'не изменилась'} на {abs(stance_change_right):.2f} см.

• Темпо-ритмовая структура боя:
  — Левый был более активен в {active_third_left}-й трети (атаки + смещение) —  {atk_in_active_left} атак.
  — Правый был более активен в {active_third_right}-й трети (атаки + смещение) —  {atk_in_active_right} атак.

• Общая тенденция ширины стойки (увеличение может свидетельствовать об усталости или переходе к обороне):
  — Левый: {'увеличивалась' if stance_trend_left > 0 else 'уменьшалась' if stance_trend_left < 0 else 'не изменилась'} за всё время.
  — Правый: {'увеличивалась' if stance_trend_right > 0 else 'уменьшалась' if stance_trend_right < 0 else 'не изменилась'} за всё время.

• Изменение ширины стойки в активной трети (увеличение может быть признаком контратакующей тактики):
  — Левый: {'увеличивалась' if stance_change_in_active_left > 0 else 'уменьшалась' if stance_change_in_active_left < 0 else 'не изменилась'} на {abs(stance_change_in_active_left):.2f} см.
  — Правый: {'увеличивалась' if stance_change_in_active_right > 0 else 'уменьшалась' if stance_change_in_active_right < 0 else 'не изменилась'} на {abs(stance_change_in_active_right):.2f} см.
"""

            pdf.set_font("DejaVu", size=12)
            pdf.cell(200, 10, txt="7. Текстовое описание", ln=True)
            pdf.set_font("DejaVu", size=10)
            pdf.multi_cell(0, 5, description)
            pdf.ln(5)

            # 8. Графики (5 штук)
            pdf.add_page()
            pdf.set_font("DejaVu", size=12)
            pdf.cell(200, 10, txt="8. Графики", ln=True)
            pdf.ln(2)

            for i, buf in enumerate([buf1, buf2, buf3, buf4, buf5], 1):
                if i in (3, 5):
                    pdf.add_page()
                img_path = f"temp_plot_{i}.png"
                with open(img_path, "wb") as f:
                    f.write(buf.getvalue())
                pdf.image(img_path, x=10, y=None, w=190)
                os.remove(img_path)
                pdf.ln(2)

            pdf.output(path)
            QMessageBox.information(self, "Успех", f"PDF-отчёт сохранён: {path}")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось создать PDF:\n{str(e)}")

# Запуск
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FightAnalyzer()
    window.show()
    sys.exit(app.exec())