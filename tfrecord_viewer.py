# import sys
# import os
# import glob
# import numpy as np
# import tensorflow as tf

# from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
#                              QHBoxLayout, QListWidget, QLabel, QPushButton, 
#                              QMessageBox, QSizePolicy, QSlider, QLineEdit, QComboBox) 
# from PyQt6.QtGui import QImage, QPixmap
# from PyQt6.QtCore import Qt, QThread, pyqtSignal

# # utils.py에서 모델과 추출기 임포트
# from utils import WaymoFrameExtractor, PanopticSegmenter, DepthProEstimator
# from waymo_open_dataset import dataset_pb2

# # --- 백그라운드 데이터 로딩 스레드 ---
# class DataLoaderThread(QThread):
#     finished_signal = pyqtSignal(list)
#     error_signal = pyqtSignal(str)

#     def __init__(self, filepath):
#         super().__init__()
#         self.filepath = filepath

#     def run(self):
#         try:
#             extractor = WaymoFrameExtractor(self.filepath)
#             frames = extractor.get_frame_list()
#             self.finished_signal.emit(frames)
#         except Exception as e:
#             self.error_signal.emit(str(e))

# # --- 백그라운드 딥러닝 추론 스레드 (선택된 모드만 실행) ---
# class InferenceThread(QThread):
#     # 결과 이미지 배열과 모드 인덱스를 함께 반환
#     finished_signal = pyqtSignal(np.ndarray, int)

#     def __init__(self, ps_model, de_model, img_array, mode_idx):
#         super().__init__()
#         self.ps_model = ps_model
#         self.de_model = de_model
#         self.img_array = img_array
#         self.mode_idx = mode_idx

#     def run(self):
#         try:
#             if self.mode_idx == 0:
#                 # 1. 원본 이미지 모드 (추론 없음)
#                 result_array = self.img_array
                
#             elif self.mode_idx == 1:
#                 # 2. Panoptic Segmentation 모드
#                 ps_res = self.ps_model.segment(self.img_array)
#                 ps_map = ps_res['segmentation'].cpu().numpy()
#                 if ps_map.max() > 0:
#                     result_array = (ps_map / ps_map.max() * 255).astype(np.uint8)
#                 else:
#                     result_array = ps_map.astype(np.uint8)
                    
#             elif self.mode_idx == 2:
#                 # 3. Depth Map 모드
#                 dm = self.de_model.get_depth_map(self.img_array)
#                 dm_norm = (dm - dm.min()) / (dm.max() - dm.min() + 1e-8)
#                 result_array = (dm_norm * 255).astype(np.uint8)
                
#             else:
#                 result_array = self.img_array

#             self.finished_signal.emit(result_array, self.mode_idx)
#         except Exception as e:
#             print(f"Inference Error: {e}")


# class DatasetViewer(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Waymo Dataset AI Viewer (PyQt6) - Mode Selection")
#         self.resize(1200, 800) 

#         # 데이터 상태 관리 변수
#         self.tfrecord_files = []
#         self.current_frames = []
#         self.current_frame_idx = 0
#         self.current_img_array = None
        
#         # 이미지 캐싱
#         self.cached_pixmap = None 
        
#         # 스레드
#         self.loader_thread = None
#         self.inference_thread = None

#         # 딥러닝 모델 초기화
#         print("AI 모델 초기화 중... 잠시만 기다려주세요.")
#         self.ps = PanopticSegmenter()
#         self.de = DepthProEstimator()
#         print("AI 모델 초기화 완료.")

#         self.init_ui()
#         self.load_file_list()

#     def init_ui(self):
#         central_widget = QWidget()
#         self.setCentralWidget(central_widget)
#         main_layout = QHBoxLayout(central_widget)

#         # --- 왼쪽 패널: 파일 리스트 ---
#         left_layout = QVBoxLayout()
#         lbl_file_list = QLabel("데이터셋 목록 (./data)")
#         left_layout.addWidget(lbl_file_list)
        
#         self.file_listbox = QListWidget()
#         self.file_listbox.setFixedWidth(250)
#         self.file_listbox.setFocusPolicy(Qt.FocusPolicy.NoFocus) 
#         self.file_listbox.itemSelectionChanged.connect(self.on_file_select)
#         left_layout.addWidget(self.file_listbox)
#         main_layout.addLayout(left_layout)

#         # --- 오른쪽 패널: 컨트롤 및 이미지 ---
#         right_layout = QVBoxLayout()
        
#         # 상단: 파일 이름 표시
#         self.filename_display = QLineEdit()
#         self.filename_display.setReadOnly(True)
#         self.filename_display.setStyleSheet("background-color: #f9f9f9; padding: 5px;")
#         self.filename_display.setPlaceholderText("선택된 파일 이름이 이곳에 표시됩니다.")
#         self.filename_display.setFocusPolicy(Qt.FocusPolicy.NoFocus)
#         right_layout.addWidget(self.filename_display)
        
#         # 뷰 모드 선택 콤보박스 추가
#         mode_layout = QHBoxLayout()
#         lbl_mode = QLabel("뷰 모드:")
#         lbl_mode.setStyleSheet("font-weight: bold;")
#         self.mode_combo = QComboBox()
#         self.mode_combo.addItems(["원본 이미지 (Original)", "Panoptic Segmentation", "Depth Map"])
#         self.mode_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus) # 단축키 충돌 방지
#         self.mode_combo.currentIndexChanged.connect(self.update_display) # 모드 변경 시 화면 즉시 업데이트
#         mode_layout.addWidget(lbl_mode)
#         mode_layout.addWidget(self.mode_combo)
#         mode_layout.addStretch()
#         right_layout.addLayout(mode_layout)
        
#         # 컨트롤 패널
#         v_control_layout = QVBoxLayout()
#         h_btn_layout = QHBoxLayout()
        
#         self.btn_prev = QPushButton("< 이전")
#         self.btn_prev.setFocusPolicy(Qt.FocusPolicy.NoFocus)
#         self.btn_prev.clicked.connect(self.prev_frame)
#         h_btn_layout.addWidget(self.btn_prev)

#         self.lbl_frame_info = QLabel("프레임: 0 / 0")
#         self.lbl_frame_info.setFixedWidth(120)
#         self.lbl_frame_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         h_btn_layout.addWidget(self.lbl_frame_info)

#         self.btn_next = QPushButton("다음 >")
#         self.btn_next.setFocusPolicy(Qt.FocusPolicy.NoFocus)
#         self.btn_next.clicked.connect(self.next_frame)
#         h_btn_layout.addWidget(self.btn_next)
#         h_btn_layout.addStretch()
        
#         self.frame_slider = QSlider(Qt.Orientation.Horizontal)
#         self.frame_slider.setMinimum(0)
#         self.frame_slider.setEnabled(False)
#         self.frame_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
#         self.frame_slider.valueChanged.connect(self.on_slider_moved)
        
#         v_control_layout.addLayout(h_btn_layout)
#         v_control_layout.addWidget(self.frame_slider)
#         right_layout.addLayout(v_control_layout)

#         # 단일 이미지 렌더링 영역
#         self.image_label = QLabel("왼쪽 목록에서 tfrecord 파일을 선택해주세요.")
#         self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         self.image_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
#         self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
#         right_layout.addWidget(self.image_label, stretch=1)

#         main_layout.addLayout(right_layout)

#     # ==========================================
#     # 전역 키보드 단축키 적용
#     # ==========================================
#     def keyPressEvent(self, event):
#         if event.key() == Qt.Key.Key_Left:
#             self.prev_frame()
#         elif event.key() == Qt.Key.Key_Right:
#             self.next_frame()
#         elif event.key() == Qt.Key.Key_Up:
#             self.prev_file()
#         elif event.key() == Qt.Key.Key_Down:
#             self.next_file()
#         # 모드 전환 단축키 (숫자키 1, 2, 3)
#         elif event.key() == Qt.Key.Key_1:
#             self.mode_combo.setCurrentIndex(0)
#         elif event.key() == Qt.Key.Key_2:
#             self.mode_combo.setCurrentIndex(1)
#         elif event.key() == Qt.Key.Key_3:
#             self.mode_combo.setCurrentIndex(2)
#         else:
#             super().keyPressEvent(event)

#     def load_file_list(self):
#         data_dir = "./data/validation"
#         if not os.path.exists(data_dir):
#             QMessageBox.warning(self, "경고", f"'{data_dir}' 폴더가 존재하지 않습니다.")
#             return

#         self.tfrecord_files = glob.glob(os.path.join(data_dir, "*.tfrecord"))
#         self.tfrecord_files.sort()
#         for f in self.tfrecord_files:
#             self.file_listbox.addItem(os.path.basename(f))

#     def on_file_select(self):
#         selected_items = self.file_listbox.selectedItems()
#         if not selected_items:
#             return
        
#         idx = self.file_listbox.row(selected_items[0])
#         selected_file = self.tfrecord_files[idx]
#         self.filename_display.setText(os.path.basename(selected_file))
#         self.show_status_message("TFRecord 프레임 파싱 중... (잠시만 기다려주세요)")
#         self.frame_slider.setEnabled(False)
        
#         self.loader_thread = DataLoaderThread(selected_file)
#         self.loader_thread.finished_signal.connect(self.on_load_finished)
#         self.loader_thread.error_signal.connect(self.on_load_error)
#         self.loader_thread.start()

#     def on_load_finished(self, frames):
#         self.current_frames = frames
#         self.current_frame_idx = 0
        
#         if len(frames) > 0:
#             self.frame_slider.setEnabled(True)
#             self.frame_slider.setMaximum(len(frames) - 1)
#             self.frame_slider.setValue(0)
            
#         self.load_current_frame_data()
#         self.update_display()

#     def on_load_error(self, error_msg):
#         QMessageBox.critical(self, "파싱 오류", f"파일을 읽는 중 오류가 발생했습니다:\n{error_msg}")
#         self.show_status_message("파일 로딩 실패")

#     def load_current_frame_data(self):
#         if not self.current_frames:
#             return

#         frame = self.current_frames[self.current_frame_idx]
#         target_camera = dataset_pb2.CameraName.FRONT
        
#         self.current_img_array = None
#         for img in frame.images:
#             if img.name == target_camera:
#                 self.current_img_array = tf.io.decode_jpeg(img.image).numpy()
#                 break
        
#         self.lbl_frame_info.setText(f"프레임: {self.current_frame_idx + 1} / {len(self.current_frames)}")
#         self.frame_slider.blockSignals(True) 
#         self.frame_slider.setValue(self.current_frame_idx)
#         self.frame_slider.blockSignals(False)

#     def prev_file(self):
#         current_row = self.file_listbox.currentRow()
#         if current_row > 0:
#             self.file_listbox.setCurrentRow(current_row - 1)

#     def next_file(self):
#         current_row = self.file_listbox.currentRow()
#         if current_row < self.file_listbox.count() - 1:
#             self.file_listbox.setCurrentRow(current_row + 1)

#     def prev_frame(self):
#         if self.current_frames and self.current_frame_idx > 0:
#             self.current_frame_idx -= 1
#             self.load_current_frame_data()
#             self.update_display()

#     def next_frame(self):
#         if self.current_frames and self.current_frame_idx < len(self.current_frames) - 1:
#             self.current_frame_idx += 1
#             self.load_current_frame_data()
#             self.update_display()

#     def on_slider_moved(self, value):
#         if self.current_frame_idx != value:
#             self.current_frame_idx = value
#             self.load_current_frame_data()
#             self.update_display()

#     def show_status_message(self, message):
#         self.cached_pixmap = None
#         self.image_label.clear()
#         self.image_label.setText(message)
#         QApplication.processEvents()

#     def update_display(self):
#         if self.current_img_array is None:
#             self.show_status_message("해당 프레임에 전면(FRONT) 카메라 이미지가 없습니다.")
#             return

#         mode_idx = self.mode_combo.currentIndex()
#         mode_text = self.mode_combo.currentText()

#         # 원본 모드가 아니면 로딩 메시지 표시
#         if mode_idx != 0:
#             self.show_status_message(f"[{mode_text}] 모델 추론 중... (잠시만 기다려주세요)")
        
#         # 추론 스레드 실행 (해당 모드만 처리)
#         self.inference_thread = InferenceThread(self.ps, self.de, self.current_img_array, mode_idx)
#         self.inference_thread.finished_signal.connect(self.on_inference_finished)
#         self.inference_thread.start()

#     def on_inference_finished(self, img_array, mode_idx):
#         # 스레드가 끝났을 때 현재 선택된 모드와 결과의 모드가 다르면(사용자가 중간에 모드를 바꿨다면) 무시
#         if self.mode_combo.currentIndex() != mode_idx:
#             return

#         self.cached_pixmap = self.create_pixmap(img_array)
#         self.render_image()

#     def create_pixmap(self, img_array):
#         """Numpy Array를 QPixmap으로 안전하게 변환"""
#         if img_array is None: return None
        
#         img_array = np.require(img_array, np.uint8, 'C')
#         h, w = img_array.shape[:2]
        
#         if len(img_array.shape) == 2:
#             qimg = QImage(img_array.data, w, h, w, QImage.Format.Format_Grayscale8)
#         elif len(img_array.shape) == 3:
#             qimg = QImage(img_array.data, w, h, w * 3, QImage.Format.Format_RGB888)
#         else:
#             return None
#         return QPixmap.fromImage(qimg)

#     def render_image(self):
#         """저장된 픽스맵을 현재 창 크기에 맞게 리사이징하여 세팅"""
#         if self.cached_pixmap:
#             self.image_label.setPixmap(self.cached_pixmap.scaled(
#                 self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

#     def resizeEvent(self, event):
#         super().resizeEvent(event)
#         self.render_image()


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
    
#     font = app.font()
#     font.setFamily("NanumGothic")
#     app.setFont(font)
    
#     viewer = DatasetViewer()
#     viewer.show()
    
#     sys.exit(app.exec())






































import sys
import os
import glob
import numpy as np
import tensorflow as tf

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QListWidget, QLabel, QPushButton, 
                             QMessageBox, QSizePolicy, QSlider, QLineEdit) 
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# Waymo Open Dataset 프로토콜 버퍼
from waymo_open_dataset import dataset_pb2

os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["XDG_SESSION_TYPE"] = "x11"

# --- [통합된 Utils] Waymo 데이터 추출기 ---
class WaymoFrameExtractor:
    def __init__(self, filepath):
        self.filepath = filepath

    def get_frame_list(self):
        """TFRecord 파일에서 프레임 리스트를 추출합니다."""
        dataset = tf.data.TFRecordDataset(self.filepath, compression_type='')
        frames = []
        for data in dataset:
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            frames.append(frame)
        return frames

# --- 백그라운드 데이터 로딩 스레드 ---
class DataLoaderThread(QThread):
    finished_signal = pyqtSignal(list)
    error_signal = pyqtSignal(str)

    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath

    def run(self):
        try:
            extractor = WaymoFrameExtractor(self.filepath)
            frames = extractor.get_frame_list()
            self.finished_signal.emit(frames)
        except Exception as e:
            self.error_signal.emit(str(e))

# --- 메인 뷰어 클래스 ---
class DatasetViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Waymo Dataset Image Viewer")
        self.resize(1100, 750) 

        # 상태 관리 변수
        self.tfrecord_files = []
        self.current_frames = []
        self.current_frame_idx = 0
        self.current_img_array = None
        self.cached_pixmap = None 

        self.init_ui()
        self.load_file_list()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- 왼쪽 패널: 파일 리스트 ---
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("데이터셋 목록 (./data/validation)"))
        
        self.file_listbox = QListWidget()
        self.file_listbox.setFixedWidth(250)
        self.file_listbox.setFocusPolicy(Qt.FocusPolicy.NoFocus) 
        self.file_listbox.itemSelectionChanged.connect(self.on_file_select)
        left_layout.addWidget(self.file_listbox)
        main_layout.addLayout(left_layout)

        # --- 오른쪽 패널: 이미지 및 컨트롤 ---
        right_layout = QVBoxLayout()
        
        # 상단: 파일 이름 표시
        self.filename_display = QLineEdit()
        self.filename_display.setReadOnly(True)
        self.filename_display.setStyleSheet("background-color: #f9f9f9; padding: 5px; font-weight: bold;")
        self.filename_display.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        right_layout.addWidget(self.filename_display)
        
        # 컨트롤 패널 (이전/다음 버튼 및 슬라이더)
        control_panel = QVBoxLayout()
        h_btn_layout = QHBoxLayout()
        
        self.btn_prev = QPushButton("◀ 이전 프레임")
        self.btn_prev.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_prev.clicked.connect(self.prev_frame)
        h_btn_layout.addWidget(self.btn_prev)

        self.lbl_frame_info = QLabel("프레임: 0 / 0")
        self.lbl_frame_info.setFixedWidth(150)
        self.lbl_frame_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        h_btn_layout.addWidget(self.lbl_frame_info)

        self.btn_next = QPushButton("다음 프레임 ▶")
        self.btn_next.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_next.clicked.connect(self.next_frame)
        h_btn_layout.addWidget(self.btn_next)
        
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.frame_slider.valueChanged.connect(self.on_slider_moved)
        
        control_panel.addLayout(h_btn_layout)
        control_panel.addWidget(self.frame_slider)
        right_layout.addLayout(control_panel)

        # 이미지 렌더링 영역
        self.image_label = QLabel("왼쪽 목록에서 파일을 선택하세요.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #222; border: 2px solid #444; color: white;")
        self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        right_layout.addWidget(self.image_label, stretch=1)

        main_layout.addLayout(right_layout)

    def keyPressEvent(self, event):
        """방향키 단축키 지원"""
        if event.key() == Qt.Key.Key_Left:
            self.prev_frame()
        elif event.key() == Qt.Key.Key_Right:
            self.next_frame()
        elif event.key() == Qt.Key.Key_Up:
            self.change_file(-1)
        elif event.key() == Qt.Key.Key_Down:
            self.change_file(1)
        else:
            super().keyPressEvent(event)

    def load_file_list(self):
        data_dir = "./data/validation"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
            self.image_label.setText(f"'{data_dir}' 폴더에 tfrecord 파일을 넣어주세요.")
            return

        self.tfrecord_files = glob.glob(os.path.join(data_dir, "*.tfrecord"))
        self.tfrecord_files.sort()
        self.file_listbox.clear()
        for f in self.tfrecord_files:
            self.file_listbox.addItem(os.path.basename(f))

    def on_file_select(self):
        selected_items = self.file_listbox.selectedItems()
        if not selected_items: return
        
        idx = self.file_listbox.row(selected_items[0])
        selected_file = self.tfrecord_files[idx]
        self.filename_display.setText(os.path.basename(selected_file))
        self.image_label.setText("데이터 로딩 중...")
        
        self.loader_thread = DataLoaderThread(selected_file)
        self.loader_thread.finished_signal.connect(self.on_load_finished)
        self.loader_thread.error_signal.connect(self.on_load_error)
        self.loader_thread.start()

    def on_load_finished(self, frames):
        self.current_frames = frames
        self.current_frame_idx = 0
        if frames:
            self.frame_slider.setEnabled(True)
            self.frame_slider.setMaximum(len(frames) - 1)
            self.frame_slider.setValue(0)
            self.update_frame()

    def on_load_error(self, error_msg):
        QMessageBox.critical(self, "오류", f"파일 로드 실패: {error_msg}")

    def update_frame(self):
        if not self.current_frames: return

        frame = self.current_frames[self.current_frame_idx]
        # FRONT 카메라 이미지 추출
        target_camera = dataset_pb2.CameraName.FRONT
        
        self.current_img_array = None
        for img in frame.images:
            if img.name == target_camera:
                self.current_img_array = tf.io.decode_jpeg(img.image).numpy()
                break
        
        self.lbl_frame_info.setText(f"프레임: {self.current_frame_idx + 1} / {len(self.current_frames)}")
        
        # 슬라이더 동기화
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_frame_idx)
        self.frame_slider.blockSignals(False)

        if self.current_img_array is not None:
            self.cached_pixmap = self.create_pixmap(self.current_img_array)
            self.render_image()
        else:
            self.image_label.setText("이 프레임에 FRONT 카메라 데이터가 없습니다.")

    def create_pixmap(self, img_array):
        h, w, c = img_array.shape
        # Waymo 이미지는 보통 RGB이므로 RGB888 포맷 사용
        qimg = QImage(img_array.data, w, h, w * c, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def render_image(self):
        if self.cached_pixmap:
            scaled_pixmap = self.cached_pixmap.scaled(
                self.image_label.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

    def prev_frame(self):
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.update_frame()

    def next_frame(self):
        if self.current_frame_idx < len(self.current_frames) - 1:
            self.current_frame_idx += 1
            self.update_frame()

    def on_slider_moved(self, value):
        self.current_frame_idx = value
        self.update_frame()

    def change_file(self, delta):
        current_row = self.file_listbox.currentRow()
        new_row = current_row + delta
        if 0 <= new_row < self.file_listbox.count():
            self.file_listbox.setCurrentRow(new_row)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.render_image()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 폰트 설정 (시스템에 따라 변경 가능)
    font = app.font()
    font.setFamily("Malgun Gothic") 
    app.setFont(font)
    
    viewer = DatasetViewer()
    viewer.show()
    sys.exit(app.exec())
