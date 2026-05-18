import sys
import os
import glob
import math
import numpy as np
import tensorflow as tf

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QLabel, QPushButton, QMessageBox, QSizePolicy,
    QSlider, QLineEdit, QCheckBox, QComboBox
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from PIL import Image, ImageDraw, ImageFont

# Waymo Open Dataset 프로토콜 버퍼
from waymo_open_dataset import dataset_pb2, label_pb2

# os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["XDG_SESSION_TYPE"] = "x11"


# -----------------------------------------------------------------------------
# 좌표계 / label helper
# -----------------------------------------------------------------------------
# Waymo vehicle/camera 좌표계: x=전방, y=좌측, z=위쪽
# OpenCV optical 좌표계: x=우측, y=아래, z=전방
VEHICLE_TO_OPTICAL = np.array([
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [1,  0,  0, 0],
    [0,  0,  0, 1],
], dtype=np.float64)

LABEL_TYPE_NAME = {
    label_pb2.Label.TYPE_UNKNOWN: "UNKNOWN",
    label_pb2.Label.TYPE_VEHICLE: "VEHICLE",
    label_pb2.Label.TYPE_PEDESTRIAN: "PEDESTRIAN",
    label_pb2.Label.TYPE_SIGN: "SIGN",
    label_pb2.Label.TYPE_CYCLIST: "CYCLIST",
}

# RGB 색상. PIL ImageDraw에서 사용한다.
LABEL_COLORS = {
    label_pb2.Label.TYPE_VEHICLE: (255, 80, 80),
    label_pb2.Label.TYPE_PEDESTRIAN: (80, 255, 80),
    label_pb2.Label.TYPE_SIGN: (255, 220, 40),
    label_pb2.Label.TYPE_CYCLIST: (80, 180, 255),
    label_pb2.Label.TYPE_UNKNOWN: (220, 220, 220),
}

FILTER_TO_TYPE = {
    "ALL": None,
    "VEHICLE": label_pb2.Label.TYPE_VEHICLE,
    "PEDESTRIAN": label_pb2.Label.TYPE_PEDESTRIAN,
    "SIGN": label_pb2.Label.TYPE_SIGN,
    "CYCLIST": label_pb2.Label.TYPE_CYCLIST,
}

# 3D box corner index 연결 순서
BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom rectangle
    (4, 5), (5, 6), (6, 7), (7, 4),  # top rectangle
    (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
]


def get_front_camera_calibration(frame):
    """FRONT camera calibration protobuf를 반환한다."""
    for calib in frame.context.camera_calibrations:
        if calib.name == dataset_pb2.CameraName.FRONT:
            return calib
    return None


def get_box_corners_vehicle(label):
    """
    Waymo 3D laser label box를 vehicle 좌표계의 8개 corner로 변환.

    Waymo box 정의:
      - center_x/y/z: box 중심
      - length: vehicle x 방향 길이
      - width: vehicle y 방향 폭
      - height: vehicle z 방향 높이
      - heading: z축 기준 yaw 회전
    """
    box = label.box
    l = float(box.length)
    w = float(box.width)
    h = float(box.height)
    cx = float(box.center_x)
    cy = float(box.center_y)
    cz = float(box.center_z)
    heading = float(box.heading)

    # heading 적용 전 local corner. 아래 4개, 위 4개 순서.
    x = l / 2.0
    y = w / 2.0
    z = h / 2.0
    corners = np.array([
        [ x,  y, -z],
        [ x, -y, -z],
        [-x, -y, -z],
        [-x,  y, -z],
        [ x,  y,  z],
        [ x, -y,  z],
        [-x, -y,  z],
        [-x,  y,  z],
    ], dtype=np.float64)

    c = math.cos(heading)
    s = math.sin(heading)
    R = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    corners = corners @ R.T
    corners += np.array([cx, cy, cz], dtype=np.float64)
    return corners


def project_vehicle_points_to_front_image(points_vehicle, camera_calib, image_shape):
    """
    Vehicle frame 3D points를 FRONT 이미지 픽셀로 투영한다.

    Returns:
        valid: (N,) bool, 카메라 앞쪽이고 이미지 내부인 점
        pixels: (N, 2) float, 각 점의 u/v
        depth: (N,) float, optical z depth
        in_front: (N,) bool, 카메라 앞쪽 여부
    """
    H, W = image_shape[:2]
    intrinsic = camera_calib.intrinsic
    fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    cam_to_vehicle = np.array(camera_calib.extrinsic.transform).reshape(4, 4)
    vehicle_to_cam = np.linalg.inv(cam_to_vehicle)
    vehicle_to_optical = VEHICLE_TO_OPTICAL @ vehicle_to_cam

    pts_h = np.concatenate(
        [points_vehicle.astype(np.float64), np.ones((points_vehicle.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    pts_opt = (vehicle_to_optical @ pts_h.T).T[:, :3]
    depth = pts_opt[:, 2]
    in_front = depth > 1e-3

    pixels = np.zeros((points_vehicle.shape[0], 2), dtype=np.float64)
    if np.any(in_front):
        uvw = (K @ pts_opt[in_front].T).T
        uv = uvw[:, :2] / uvw[:, 2:3]
        pixels[in_front] = uv

    valid = (
        in_front
        & (pixels[:, 0] >= 0) & (pixels[:, 0] < W)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < H)
    )
    return valid, pixels, depth, in_front


def draw_text_with_background(draw, xy, text, fill, font):
    """가독성을 위해 검은 배경 박스를 깔고 텍스트를 그린다."""
    x, y = xy
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((x, y), text, font=font)
    else:
        w, h = draw.textsize(text, font=font)
        bbox = (x, y, x + w, y + h)
    pad = 3
    bg = (0, 0, 0)
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=bg)
    draw.text((x, y), text, fill=fill, font=font)


def draw_waymo_3d_boxes_on_image(img_array, frame, type_filter=None):
    """
    FRONT RGB 이미지 위에 Waymo laser 3D bounding box, class label, track id를 그린다.

    Args:
        img_array: H x W x 3 RGB uint8
        frame: Waymo Frame protobuf
        type_filter: None이면 전체, 아니면 label_pb2.Label.TYPE_* 하나

    Returns:
        annotated_img: H x W x 3 RGB uint8
        visible_count: 실제로 이미지에 그려진 box 개수
        total_count: 필터를 통과한 laser label 개수
    """
    camera_calib = get_front_camera_calibration(frame)
    if camera_calib is None:
        return img_array, 0, 0

    pil_img = Image.fromarray(img_array.copy())
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    H, W = img_array.shape[:2]
    visible_count = 0
    total_count = 0

    for label in frame.laser_labels:
        if type_filter is not None and label.type != type_filter:
            continue
        total_count += 1

        corners = get_box_corners_vehicle(label)
        valid, pixels, depth, in_front = project_vehicle_points_to_front_image(
            corners, camera_calib, (H, W)
        )

        # 카메라 앞쪽에 corner가 거의 없으면 FRONT 이미지에 그리지 않는다.
        if int(np.count_nonzero(in_front)) < 2:
            continue

        color = LABEL_COLORS.get(label.type, LABEL_COLORS[label_pb2.Label.TYPE_UNKNOWN])
        line_width = 3 if label.type == label_pb2.Label.TYPE_SIGN else 2

        # box edge 그리기. 두 endpoint가 모두 카메라 앞쪽이면 PIL이 화면 밖 좌표도 알아서 clip한다.
        any_edge_drawn = False
        for a, b in BOX_EDGES:
            if not (in_front[a] and in_front[b]):
                continue
            p1 = (int(round(pixels[a, 0])), int(round(pixels[a, 1])))
            p2 = (int(round(pixels[b, 0])), int(round(pixels[b, 1])))
            draw.line([p1, p2], fill=color, width=line_width)
            any_edge_drawn = True

        if not any_edge_drawn:
            continue

        # label 텍스트 위치: 이미지 내부에 들어온 corner 중 가장 위쪽/왼쪽.
        if np.any(valid):
            uv = pixels[valid]
        else:
            # endpoint는 앞에 있으나 화면 밖이면 전체 projection 좌표 기준으로 대략 위치 표시
            uv = pixels[in_front]
        if uv.shape[0] == 0:
            continue

        u_min = int(np.clip(np.min(uv[:, 0]), 0, W - 1))
        v_min = int(np.clip(np.min(uv[:, 1]) - 18, 0, H - 1))
        label_name = LABEL_TYPE_NAME.get(label.type, f"TYPE_{int(label.type)}")
        track_id = label.id if getattr(label, "id", "") else "no-id"
        text = f"{label_name} | id={track_id}"
        draw_text_with_background(draw, (u_min, v_min), text, color, font)
        visible_count += 1

    return np.asarray(pil_img), visible_count, total_count


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
        self.setWindowTitle("Waymo Dataset Viewer - 3D Box / Label / Track ID")
        self.resize(1280, 820)

        # 상태 관리 변수
        self.tfrecord_files = []
        self.current_frames = []
        self.current_frame_idx = 0
        self.current_img_array = None
        self.current_annotated_array = None
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
        self.file_listbox.setFixedWidth(280)
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
        self.lbl_frame_info.setFixedWidth(170)
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

        # 3D box 표시 옵션
        box_control_layout = QHBoxLayout()
        self.chk_show_3d_box = QCheckBox("3D Bounding Box / Label / Track ID 표시")
        self.chk_show_3d_box.setChecked(True)
        self.chk_show_3d_box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.chk_show_3d_box.stateChanged.connect(self.update_frame)
        box_control_layout.addWidget(self.chk_show_3d_box)

        box_control_layout.addWidget(QLabel("Label filter:"))
        self.combo_label_filter = QComboBox()
        self.combo_label_filter.addItems(["ALL", "SIGN", "VEHICLE", "PEDESTRIAN", "CYCLIST"])
        self.combo_label_filter.setCurrentText("ALL")
        self.combo_label_filter.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.combo_label_filter.currentTextChanged.connect(self.update_frame)
        box_control_layout.addWidget(self.combo_label_filter)

        self.lbl_box_info = QLabel("boxes: 0 / 0")
        self.lbl_box_info.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        box_control_layout.addWidget(self.lbl_box_info, stretch=1)
        right_layout.addLayout(box_control_layout)

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
        elif event.key() == Qt.Key.Key_B:
            self.chk_show_3d_box.setChecked(not self.chk_show_3d_box.isChecked())
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
        if not selected_items:
            return

        idx = self.file_listbox.row(selected_items[0])
        selected_file = self.tfrecord_files[idx]
        self.filename_display.setText(os.path.basename(selected_file))
        self.image_label.setText("데이터 로딩 중...")
        self.frame_slider.setEnabled(False)
        self.lbl_box_info.setText("boxes: - / -")

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
        else:
            self.image_label.setText("프레임이 없습니다.")

    def on_load_error(self, error_msg):
        QMessageBox.critical(self, "오류", f"파일 로드 실패: {error_msg}")

    def update_frame(self):
        if not self.current_frames:
            return

        frame = self.current_frames[self.current_frame_idx]
        target_camera = dataset_pb2.CameraName.FRONT

        # FRONT 카메라 이미지 추출
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

        if self.current_img_array is None:
            self.image_label.setText("이 프레임에 FRONT 카메라 데이터가 없습니다.")
            self.lbl_box_info.setText("boxes: 0 / 0")
            return

        # 3D bbox / label / track id overlay
        if self.chk_show_3d_box.isChecked():
            filter_name = self.combo_label_filter.currentText()
            type_filter = FILTER_TO_TYPE.get(filter_name, None)
            annotated, visible_count, total_count = draw_waymo_3d_boxes_on_image(
                self.current_img_array, frame, type_filter=type_filter
            )
            self.current_annotated_array = annotated
            self.lbl_box_info.setText(f"visible boxes: {visible_count} / filtered labels: {total_count}")
        else:
            self.current_annotated_array = self.current_img_array
            self.lbl_box_info.setText(f"laser labels: {len(frame.laser_labels)}")

        self.cached_pixmap = self.create_pixmap(self.current_annotated_array)
        self.render_image()

    def create_pixmap(self, img_array):
        h, w, c = img_array.shape
        # QImage가 numpy 메모리를 참조하므로 copy()로 수명 문제를 방지한다.
        qimg = QImage(img_array.data, w, h, w * c, QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    def render_image(self):
        if self.cached_pixmap:
            scaled_pixmap = self.cached_pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
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
