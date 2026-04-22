# ================================================================================
# Waymo Open Dataset → NeRF/3DGS 데이터셋 변환 스크립트
# ================================================================================
# [회의록 기반 요구사항]
#   1. Mask (Panoptic Segmentation) 추출 ─ Mask2Former 사용
#   2. Depth Map (Edge 보존) 추출 ─ DepthPro 사용
#   3. Camera Extrinsic Parameters 추출
#   4. Camera Intrinsic Parameters 추출 (3D 재구성 단계에서 필수)
#
# [추출 대상 프레임]
#   - 시작 프레임: 6
#   - 프레임 간격: 6 (즉, 6, 12, 18, ..., 72)
#   - 총 12프레임
#
# [Waymo Camera Parameter vs 표준 NeRF/3DGS 차이점]
#   ── Intrinsic ──
#     Waymo: 1D 배열 [f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3]
#     표준 : 3x3 K 행렬 + 별도의 왜곡 계수
#
#   ── Extrinsic ──
#     Waymo: 4x4 행렬 (Camera → Vehicle 변환)
#     표준 NeRF/3DGS: 4x4 행렬 (Camera → World 변환, "cam2world")
#     변환식: cam2world = vehicle_pose(vehicle→world) @ cam2vehicle
#
#   ── 좌표계 ──
#     Waymo 카메라: x-전방, y-좌측, z-상단 (vehicle-aligned)
#     OpenCV (표준): z-전방, x-우측, y-하단
#     NeRF (OpenGL): -z-전방, x-우측, y-상단
#
#   본 스크립트에서는 Waymo 원본과 표준 cam2world 변환본을 함께 저장합니다.
# ================================================================================

import os                                                       # 디렉토리 생성, 경로 처리
import sys                                                      # depth_pro 모듈 경로 추가용
import tensorflow as tf                                         # TFRecord 파일 읽기 + JPEG 디코딩
import torch                                                    # 딥러닝 추론 디바이스 관리
import numpy as np                                              # 수치 연산
from PIL import Image                                           # 이미지 변환/저장
from waymo_open_dataset import dataset_pb2                      # Waymo 데이터셋 protobuf 정의
from transformers import (                                      # Mask2Former 관련 모듈
    Mask2FormerImageProcessor,
    Mask2FormerForUniversalSegmentation,
)

# DepthPro 패키지 경로 추가 (로컬 클론된 ml-depth-pro 디렉토리)
sys.path.append('ml-depth-pro')
import depth_pro                                                # DepthPro 모델 import


# ================================================================================
# [Class] WaymoFrameExtractor
#   Waymo Open Dataset의 TFRecord 파일에서 프레임 단위로 데이터 추출
# ================================================================================
class WaymoFrameExtractor:
    """Waymo TFRecord 파일에서 프레임 객체를 파싱하고 카메라 데이터를 추출하는 클래스"""

    def __init__(self, tfrecord_path: str):
        """
        Args:
            tfrecord_path (str): Waymo TFRecord 파일의 경로
        """
        # 인스턴스 변수에 파일 경로 저장
        self.tfrecord_path = tfrecord_path

    def get_frame_list(self) -> list:
        """
        TFRecord 파일을 한 번 순회하여 모든 Frame 객체를 메모리에 로드.

        Returns:
            list: dataset_pb2.Frame 객체들의 리스트 (시간 순서)
        """
        # TensorFlow의 TFRecordDataset 객체 생성 (Waymo는 압축 없음)
        dataset = tf.data.TFRecordDataset(self.tfrecord_path, compression_type='')

        # 결과를 담을 빈 리스트 초기화
        frame_list = []

        # 데이터셋의 각 직렬화된 레코드를 순회
        for data in dataset:
            # 빈 Frame protobuf 객체 생성
            frame = dataset_pb2.Frame()
            # bytes 데이터를 Frame 객체로 역직렬화(파싱)
            frame.ParseFromString(bytearray(data.numpy()))
            # 리스트에 추가
            frame_list.append(frame)

        return frame_list

    def get_camera_data(self, frame: dataset_pb2.Frame, target_camera: int) -> tuple:
        """
        프레임에서 특정 카메라의 이미지/외부 파라미터/내부 파라미터/해상도를 일괄 추출.

        Args:
            frame (dataset_pb2.Frame): 단일 프레임 객체
            target_camera (int): 추출할 카메라 이름 (예: dataset_pb2.CameraName.FRONT)

        Returns:
            tuple: (img_array, extrinsic_matrix, intrinsic_params, width, height)
                - img_array (np.ndarray): (H, W, 3) RGB uint8 이미지
                - extrinsic_matrix (np.ndarray): 4x4 (Camera → Vehicle 변환)
                - intrinsic_params (np.ndarray): (9,) [f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3]
                - width (int): 이미지 가로 픽셀 수
                - height (int): 이미지 세로 픽셀 수
        """
        # 반환값 초기화 (해당 카메라가 없을 경우 None 유지)
        img_array = None
        extrinsic_matrix = None
        intrinsic_params = None
        width = None
        height = None

        # frame.images 리스트에서 타겟 카메라 이미지 검색
        for img in frame.images:
            if img.name == target_camera:
                # JPEG 바이너리를 NumPy 배열(H,W,3)로 디코딩
                img_array = tf.io.decode_jpeg(img.image).numpy()
                # 첫 매칭에서 종료 (불필요한 순회 방지)
                break

        # frame.context.camera_calibrations에서 동일 카메라의 캘리브레이션 검색
        for cal in frame.context.camera_calibrations:
            if cal.name == target_camera:
                # Extrinsic: 길이 16의 1D 리스트 → 4x4 행렬로 reshape
                # 의미: Camera 좌표계의 점을 Vehicle 좌표계로 변환하는 행렬
                extrinsic_matrix = np.array(cal.extrinsic.transform).reshape(4, 4)

                # Intrinsic: Waymo 형식 1D 배열 [f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3]
                intrinsic_params = np.array(cal.intrinsic)

                # 이미지 해상도 (왜곡 보정/리사이즈 시 필요)
                width = cal.width
                height = cal.height
                break

        return img_array, extrinsic_matrix, intrinsic_params, width, height

    def get_vehicle_pose(self, frame: dataset_pb2.Frame) -> np.ndarray:
        """
        해당 프레임에서 차량(Vehicle)의 World 좌표계 포즈(4x4 행렬) 추출.
        의미: Vehicle 좌표계의 점을 World 좌표계로 변환하는 행렬.

        Args:
            frame (dataset_pb2.Frame): 단일 프레임 객체

        Returns:
            np.ndarray: 4x4 변환 행렬 (Vehicle → World)
        """
        # frame.pose.transform: 길이 16의 1D 리스트 → 4x4 행렬로 reshape
        return np.array(frame.pose.transform).reshape(4, 4)


# ================================================================================
# [Class] PanopticSegmenter
#   Mask2Former 기반 Panoptic Segmentation 수행
# ================================================================================
class PanopticSegmenter:
    """Mask2Former (Cityscapes-Panoptic) 기반 Panoptic Segmentation 클래스"""

    def __init__(self, device_id: str = None):
        """
        Args:
            device_id (str): 사용할 디바이스 (None이면 자동 선택)
        """
        # 사전 학습된 모델 ID (도시 환경 → 자율주행 데이터에 적합)
        model_id = "facebook/mask2former-swin-large-cityscapes-panoptic"

        # GPU 사용 가능 시 cuda, 아니면 cpu 자동 선택
        self.device = torch.device(
            device_id if device_id else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # 이미지 전처리 프로세서 로드 (정규화, 리사이즈 등 담당)
        self.processor = Mask2FormerImageProcessor.from_pretrained(model_id)

        # 모델 가중치 로드 후 디바이스로 이동
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id)
        self.model.to(self.device)
        # 평가 모드 (Dropout, BN 등 비활성화)
        self.model.eval()

    def segment(self, image: np.ndarray) -> dict:
        """
        NumPy 이미지를 받아 Panoptic Segmentation을 수행.

        Args:
            image (np.ndarray): 분석할 (H, W, 3) RGB 이미지

        Returns:
            dict:
                - 'segmentation' (torch.Tensor): (H, W) 각 픽셀의 segment ID
                - 'segments_info' (list): segment ID에 대응하는 클래스/score 등 메타데이터
        """
        # 이미지 전처리 → 텐서 변환 → 디바이스 이동
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # 추론 (gradient 계산 비활성화로 메모리 절약)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 후처리: 모델 출력 해상도 → 원본 이미지 해상도로 복원
        # NumPy의 image.shape는 (H, W, C) 형태이므로 [:2]는 (H, W)
        target_sizes = [image.shape[:2]]
        results = self.processor.post_process_panoptic_segmentation(
            outputs, target_sizes=target_sizes
        )[0]

        return results

    def get_class_name(self, label_id: int) -> str:
        """
        모델의 label_id를 사람이 읽을 수 있는 클래스 이름으로 변환.
        예: 6 → 'traffic light'

        Args:
            label_id (int): 클래스 ID

        Returns:
            str: 클래스 이름
        """
        # config.id2label 딕셔너리에서 조회 (없으면 unknown으로 처리)
        return self.model.config.id2label.get(label_id, f"unknown_{label_id}")


# ================================================================================
# [Class] DepthProEstimator
#   DepthPro 기반 Metric Depth Estimation 수행
# ================================================================================
class DepthProEstimator:
    """DepthPro 기반 단안(monocular) Metric Depth Estimation 클래스"""

    def __init__(self, device_id: str = None):
        """
        Args:
            device_id (str): 사용할 디바이스 (None이면 자동 선택)
        """
        # GPU 사용 가능 시 cuda, 아니면 cpu 자동 선택
        self.device = torch.device(
            device_id if device_id else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # 도시 환경 기준 최대 깊이 50m로 클리핑 (먼 거리는 의미가 적음)
        self.max_depth = 50.0

        # DepthPro 모델 + 입력 transform을 한 번에 로드
        self.model, self.transform = depth_pro.create_model_and_transforms()
        # 디바이스 이동 + 평가 모드 설정
        self.model.to(self.device)
        self.model.eval()

    def get_depth_map(self, image_input: np.ndarray) -> np.ndarray:
        """
        NumPy 이미지를 받아 Metric Depth Map(단위: m)을 반환.

        Args:
            image_input (np.ndarray): (H, W, 3) RGB 이미지 (uint8 또는 float)

        Returns:
            np.ndarray: (H, W) float, 0~max_depth(50m) 범위로 클리핑된 깊이 맵
        """
        # focal length(f_px)가 None이면 DepthPro가 EXIF 또는 자체 추정으로 처리
        f_px = None

        # 입력 dtype에 따라 uint8로 통일 (depth_pro의 transform이 PIL Image를 기대)
        if image_input.dtype in [np.float32, np.float64]:
            # 0~1 정규화된 float이라고 가정하고 0~255로 스케일
            img_array_uint8 = (image_input * 255).astype(np.uint8)
        else:
            # 이미 uint8이거나 그에 준하는 경우 그대로 캐스팅
            img_array_uint8 = image_input.astype(np.uint8)

        # NumPy → PIL Image 변환
        image = Image.fromarray(img_array_uint8)

        # 모델 입력 형태로 전처리 후 디바이스 이동
        input_tensor = self.transform(image).to(self.device)

        # 추론 (gradient 비활성화)
        with torch.no_grad():
            prediction = self.model.infer(input_tensor, f_px=f_px)

        # 결과 텐서를 CPU로 이동 후 NumPy 변환
        depth_map_npy = prediction["depth"].cpu().numpy()

        # 0 ~ max_depth(50m) 범위로 클리핑 (음수/극단값 제거)
        dm_clipped = np.clip(depth_map_npy, a_min=0.0, a_max=self.max_depth)

        return dm_clipped


# ================================================================================
# [Helper Functions]
# ================================================================================

def waymo_intrinsic_to_K_matrix(intrinsic_params: np.ndarray) -> tuple:
    """
    Waymo 형식 1D Intrinsic을 표준 3x3 K 행렬 + 왜곡 계수로 분리.

    Waymo 형식: [f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3]
    표준 NeRF/3DGS 형식: 3x3 K 행렬 + OpenCV 호환 왜곡 계수 [k1, k2, p1, p2, k3]

    Args:
        intrinsic_params (np.ndarray): Waymo의 길이 9 1D 배열

    Returns:
        tuple:
            - K (np.ndarray): 3x3 카메라 행렬
            - distortion (np.ndarray): (5,) 왜곡 계수 [k1, k2, p1, p2, k3]
    """
    # 초점 거리(focal length)와 주점(principal point) 추출
    f_u, f_v, c_u, c_v = intrinsic_params[:4]

    # 표준 3x3 K 행렬 구성
    K = np.array([
        [f_u, 0.0, c_u],
        [0.0, f_v, c_v],
        [0.0, 0.0, 1.0],
    ])

    # 왜곡 계수 (OpenCV 표준 순서: k1, k2, p1, p2, k3)
    distortion = intrinsic_params[4:9]

    return K, distortion


def visualize_depth(depth_map: np.ndarray, max_depth: float = 50.0) -> np.ndarray:
    """
    Float depth map(단위: m)을 시각화용 0~255 uint8 이미지로 변환.
    가까운 곳 → 검은색(0), 먼 곳 → 흰색(255)

    Args:
        depth_map (np.ndarray): (H, W) float depth (단위: m)
        max_depth (float): 시각화 최대 깊이 (이상은 흰색으로 포화)

    Returns:
        np.ndarray: (H, W) uint8
    """
    # 안전을 위해 한 번 더 클리핑
    dm_clipped = np.clip(depth_map, 0.0, max_depth)
    # 0~max_depth → 0~255 선형 변환
    dm_vis = (dm_clipped / max_depth * 255).astype(np.uint8)
    return dm_vis


def visualize_panoptic(seg_map: np.ndarray) -> np.ndarray:
    """
    Panoptic segmentation map(정수 ID)을 시각화용 0~255 uint8로 변환.
    (육안 확인용일 뿐 실제 분석에는 .npy 파일을 사용해야 함)

    Args:
        seg_map (np.ndarray): (H, W) 정수 ID 맵

    Returns:
        np.ndarray: (H, W) uint8
    """
    # 모든 픽셀이 0이면 그대로 반환 (0으로 나눔 방지)
    if seg_map.max() > 0:
        ps_vis = (seg_map.astype(np.float32) / seg_map.max() * 255).astype(np.uint8)
    else:
        ps_vis = seg_map.astype(np.uint8)
    return ps_vis


def save_image(img_array: np.ndarray, save_path: str):
    """
    NumPy uint8 배열을 JPG로 저장.

    Args:
        img_array (np.ndarray): 저장할 이미지 배열 (uint8)
        save_path (str): 저장 경로
    """
    # NumPy → PIL Image 변환
    img = Image.fromarray(img_array)
    # JPEG 형식으로 디스크에 저장
    img.save(save_path, format="JPEG")


def save_extrinsic_text(save_path: str,
                        frame_idx: int,
                        camera_name: str,
                        extrinsic_cam2vehicle: np.ndarray,
                        vehicle_pose: np.ndarray,
                        extrinsic_cam2world: np.ndarray):
    """
    카메라 Extrinsic 파라미터를 사람이 읽기 쉬운 텍스트 파일로 저장.

    Args:
        save_path (str): 저장 경로 (.txt)
        frame_idx (int): 프레임 인덱스
        camera_name (str): 카메라 이름 (예: 'FRONT')
        extrinsic_cam2vehicle (np.ndarray): 4x4 (Camera → Vehicle)
        vehicle_pose (np.ndarray): 4x4 (Vehicle → World)
        extrinsic_cam2world (np.ndarray): 4x4 (Camera → World, NeRF/3DGS 표준)
    """
    # 텍스트 파일 쓰기 (UTF-8로 한국어 주석 포함)
    with open(save_path, 'w', encoding='utf-8') as f:
        # 데이터 종류 식별을 위한 헤더
        f.write("# ============================================================\n")
        f.write("# DATA TYPE  : Camera Extrinsic Parameters\n")
        f.write("# SOURCE     : Waymo Open Dataset\n")
        f.write(f"# FRAME INDEX: {frame_idx}\n")
        f.write(f"# CAMERA NAME: {camera_name}\n")
        f.write("# ============================================================\n")
        f.write("# [좌표계 설명]\n")
        f.write("#   Waymo 카메라 좌표계: x-전방, y-좌측, z-상단\n")
        f.write("#   Vehicle 좌표계    : x-전방, y-좌측, z-상단\n")
        f.write("#   World 좌표계      : ENU 기반 글로벌 좌표\n")
        f.write("# ============================================================\n\n")

        # 1) Waymo 원본: Camera → Vehicle
        f.write("[1] Extrinsic: Camera → Vehicle (4x4)\n")
        f.write("# Waymo 원본 (cal.extrinsic.transform). 카메라 좌표를 차량 좌표로 변환.\n")
        np.savetxt(f, extrinsic_cam2vehicle, fmt='%.10f')
        f.write("\n")

        # 2) Waymo 차량 포즈: Vehicle → World
        f.write("[2] Vehicle Pose: Vehicle → World (4x4)\n")
        f.write("# Waymo 원본 (frame.pose.transform). 차량 좌표를 글로벌 좌표로 변환.\n")
        np.savetxt(f, vehicle_pose, fmt='%.10f')
        f.write("\n")

        # 3) NeRF/3DGS 표준: Camera → World
        f.write("[3] Extrinsic: Camera → World (4x4)  ← NeRF/3DGS 표준 (cam2world)\n")
        f.write("# = Vehicle Pose @ (Camera → Vehicle)\n")
        f.write("# NeRF/3DGS에서 곧바로 사용하는 형식.\n")
        np.savetxt(f, extrinsic_cam2world, fmt='%.10f')


def save_intrinsic_text(save_path: str,
                        frame_idx: int,
                        camera_name: str,
                        intrinsic_raw: np.ndarray,
                        K_matrix: np.ndarray,
                        distortion: np.ndarray,
                        width: int,
                        height: int):
    """
    카메라 Intrinsic 파라미터를 사람이 읽기 쉬운 텍스트 파일로 저장.

    Args:
        save_path (str): 저장 경로 (.txt)
        frame_idx (int): 프레임 인덱스
        camera_name (str): 카메라 이름
        intrinsic_raw (np.ndarray): Waymo 원본 1D 배열 (9,)
        K_matrix (np.ndarray): 표준 3x3 K 행렬
        distortion (np.ndarray): 왜곡 계수 (5,)
        width (int): 이미지 가로 해상도
        height (int): 이미지 세로 해상도
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        # 데이터 종류 식별을 위한 헤더
        f.write("# ============================================================\n")
        f.write("# DATA TYPE  : Camera Intrinsic Parameters\n")
        f.write("# SOURCE     : Waymo Open Dataset\n")
        f.write(f"# FRAME INDEX: {frame_idx}\n")
        f.write(f"# CAMERA NAME: {camera_name}\n")
        f.write("# ============================================================\n\n")

        # 1) 이미지 해상도
        f.write("[1] Image Resolution\n")
        f.write(f"width : {width}\n")
        f.write(f"height: {height}\n\n")

        # 2) Waymo 원본 형식
        f.write("[2] Waymo Raw Intrinsic (1D, 9 elements)\n")
        f.write("# 형식: [f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3]\n")
        f.write(f"f_u: {intrinsic_raw[0]:.10f}\n")
        f.write(f"f_v: {intrinsic_raw[1]:.10f}\n")
        f.write(f"c_u: {intrinsic_raw[2]:.10f}\n")
        f.write(f"c_v: {intrinsic_raw[3]:.10f}\n")
        f.write(f"k1 : {intrinsic_raw[4]:.10f}\n")
        f.write(f"k2 : {intrinsic_raw[5]:.10f}\n")
        f.write(f"p1 : {intrinsic_raw[6]:.10f}\n")
        f.write(f"p2 : {intrinsic_raw[7]:.10f}\n")
        f.write(f"k3 : {intrinsic_raw[8]:.10f}\n\n")

        # 3) 표준 3x3 K 행렬 (NeRF/3DGS에서 곧바로 사용)
        f.write("[3] Standard 3x3 K Matrix  ← NeRF/3DGS 표준\n")
        f.write("# K = [[f_u,   0, c_u],\n")
        f.write("#      [  0, f_v, c_v],\n")
        f.write("#      [  0,   0,   1]]\n")
        np.savetxt(f, K_matrix, fmt='%.10f')
        f.write("\n")

        # 4) 왜곡 계수 (OpenCV 호환)
        f.write("[4] Distortion Coefficients (OpenCV 형식)\n")
        f.write("# 형식: [k1, k2, p1, p2, k3]\n")
        np.savetxt(f, distortion.reshape(1, -1), fmt='%.10f')


def save_panoptic_info(save_path: str,
                       frame_idx: int,
                       camera_name: str,
                       segments_info: list,
                       segmenter: PanopticSegmenter):
    """
    Panoptic Segmentation의 segments_info 메타데이터를 텍스트로 저장.
    (.npy 파일의 픽셀 ID와 매칭하여 클래스/점수 확인용)

    Args:
        save_path (str): 저장 경로 (.txt)
        frame_idx (int): 프레임 인덱스
        camera_name (str): 카메라 이름
        segments_info (list): Mask2Former의 segments_info 리스트
        segmenter (PanopticSegmenter): 클래스 이름 변환에 사용할 segmenter
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        # 데이터 종류 식별 헤더
        f.write("# ============================================================\n")
        f.write("# DATA TYPE  : Panoptic Segmentation Info (segments_info)\n")
        f.write("# MODEL      : Mask2Former (Cityscapes-Panoptic)\n")
        f.write(f"# FRAME INDEX: {frame_idx}\n")
        f.write(f"# CAMERA NAME: {camera_name}\n")
        f.write("# ============================================================\n")
        f.write("# 같은 디렉토리의 .npy 파일에 저장된 픽셀 ID와 매칭됩니다.\n")
        f.write("# 예: .npy의 픽셀 값이 3이면, 아래 표에서 segment_id=3 행 참조\n")
        f.write("# ============================================================\n\n")

        # 표 헤더
        f.write(f"{'segment_id':<12}{'label_id':<10}{'class_name':<25}{'score':<10}{'was_fused':<10}\n")
        f.write("-" * 70 + "\n")

        # 각 segment 정보를 한 줄씩 출력
        for seg in segments_info:
            seg_id = seg['id']
            label_id = seg['label_id']
            # 모델의 id2label로 사람이 읽을 수 있는 이름 변환
            class_name = segmenter.get_class_name(label_id)
            score = seg.get('score', 0.0)
            was_fused = seg.get('was_fused', False)
            f.write(f"{seg_id:<12}{label_id:<10}{class_name:<25}{score:<10.4f}{str(was_fused):<10}\n")


# ================================================================================
# [Main] 실행부
# ================================================================================
if __name__ == "__main__":

    # ────────────────────────── 설정값 ──────────────────────────
    # 처리할 TFRecord 파일 경로
    TFRECORD_PATH = '../blender/data/validation/individual_files_validation_segment-4575389405178805994_4900_000_4920_000_with_camera_labels.tfrecord'

    # 추출 시작 프레임 인덱스
    START_FRAME = 6
    # 프레임 간격 (n번째마다 추출)
    FRAME_STEP = 6
    # 추출할 총 프레임 수
    NUM_FRAMES = 12

    # 추출 대상 카메라 (정면 카메라)
    TARGET_CAMERA = dataset_pb2.CameraName.FRONT
    # 카메라 이름 문자열 ('FRONT' 등) — 파일 헤더 기록용
    CAMERA_NAME_STR = dataset_pb2.CameraName.Name.Name(TARGET_CAMERA)

    # 출력 루트 디렉토리
    OUTPUT_ROOT = './output'

    # ────────────────────────── 출력 폴더 생성 ──────────────────────────
    # 데이터 종류별로 하위 디렉토리 분리
    OUTPUT_DIRS = {
        'image':     os.path.join(OUTPUT_ROOT, 'image'),       # 원본 이미지
        'depth':     os.path.join(OUTPUT_ROOT, 'depth'),       # Depth Map (.npy + .jpg)
        'panoptic':  os.path.join(OUTPUT_ROOT, 'panoptic'),    # Panoptic (.npy + .jpg + info .txt)
        'extrinsic': os.path.join(OUTPUT_ROOT, 'extrinsic'),   # Camera Extrinsic (.txt)
        'intrinsic': os.path.join(OUTPUT_ROOT, 'intrinsic'),   # Camera Intrinsic (.txt)
    }
    # 폴더가 없으면 생성 (이미 있으면 무시)
    for dir_path in OUTPUT_DIRS.values():
        os.makedirs(dir_path, exist_ok=True)

    # ────────────────────────── 추출기/모델 초기화 ──────────────────────────
    print(f"[1/4] TFRecord 로드 중: {TFRECORD_PATH}")
    wfe = WaymoFrameExtractor(TFRECORD_PATH)
    # 전체 프레임 리스트 메모리에 로드
    frame_list = wfe.get_frame_list()
    print(f"      → 총 {len(frame_list)}개 프레임 로드 완료")

    print("[2/4] PanopticSegmenter 초기화 중... (Mask2Former)")
    ps = PanopticSegmenter()

    print("[3/4] DepthProEstimator 초기화 중... (DepthPro)")
    de = DepthProEstimator()

    # ────────────────────────── 추출할 프레임 인덱스 계산 ──────────────────────────
    # [START_FRAME, START_FRAME + STEP, ..., 총 NUM_FRAMES개]
    frame_indices = [START_FRAME + i * FRAME_STEP for i in range(NUM_FRAMES)]

    print(f"[4/4] 추출 대상 프레임: {frame_indices}")
    print(f"      → 카메라: {CAMERA_NAME_STR}\n")

    # ────────────────────────── 각 프레임에 대해 데이터 추출 및 저장 ──────────────────────────
    for i, frame_idx in enumerate(frame_indices):
        # 프레임 인덱스가 전체 프레임 수를 벗어나면 건너뛰기
        if frame_idx >= len(frame_list):
            print(f"  [SKIP] frame_{frame_idx} - 범위 초과 (총 {len(frame_list)}개)")
            continue

        print(f"  [{i + 1:2d}/{len(frame_indices)}] frame_{frame_idx:04d} 처리 중...")

        # 파일명 prefix (zero-padding으로 정렬 용이)
        prefix = f"frame_{frame_idx:04d}"
        # 현재 처리할 프레임 객체
        frame = frame_list[frame_idx]

        # ─── (1) 이미지 + 카메라 파라미터 일괄 추출 ───
        img_array, extrinsic_cam2vehicle, intrinsic_raw, width, height = \
            wfe.get_camera_data(frame, TARGET_CAMERA)

        # 차량 World 포즈 (Vehicle → World)
        vehicle_pose = wfe.get_vehicle_pose(frame)

        # NeRF/3DGS 표준 cam2world 계산: vehicle_pose @ cam2vehicle
        extrinsic_cam2world = vehicle_pose @ extrinsic_cam2vehicle

        # Waymo 1D Intrinsic → 표준 3x3 K + 왜곡 계수
        K_matrix, distortion = waymo_intrinsic_to_K_matrix(intrinsic_raw)

        # ─── (2) 원본 이미지 저장 (참조용) ───
        save_image(
            img_array,
            os.path.join(OUTPUT_DIRS['image'], f"{prefix}.jpg"),
        )

        # ─── (3) Depth Map 추출 및 저장 ───
        depth_map = de.get_depth_map(img_array)
        # Raw float depth (단위: m, 0~50) → .npy로 정밀 저장
        np.save(
            os.path.join(OUTPUT_DIRS['depth'], f"{prefix}.npy"),
            depth_map,
        )
        # 시각화용 .jpg 저장
        depth_vis = visualize_depth(depth_map, max_depth=de.max_depth)
        save_image(
            depth_vis,
            os.path.join(OUTPUT_DIRS['depth'], f"{prefix}.jpg"),
        )

        # ─── (4) Panoptic Segmentation 추출 및 저장 ───
        ps_res = ps.segment(img_array)
        # segmentation 텐서 → CPU NumPy (int32)
        seg_map = ps_res['segmentation'].cpu().numpy().astype(np.int32)
        segments_info = ps_res['segments_info']

        # Raw segment ID 맵 → .npy로 정밀 저장 (분석용)
        np.save(
            os.path.join(OUTPUT_DIRS['panoptic'], f"{prefix}.npy"),
            seg_map,
        )
        # 시각화용 .jpg 저장
        ps_vis = visualize_panoptic(seg_map)
        save_image(
            ps_vis,
            os.path.join(OUTPUT_DIRS['panoptic'], f"{prefix}.jpg"),
        )
        # segments_info 메타데이터 → .txt 저장
        save_panoptic_info(
            os.path.join(OUTPUT_DIRS['panoptic'], f"{prefix}_info.txt"),
            frame_idx, CAMERA_NAME_STR, segments_info, ps,
        )

        # ─── (5) Extrinsic 파라미터 저장 ───
        save_extrinsic_text(
            os.path.join(OUTPUT_DIRS['extrinsic'], f"{prefix}.txt"),
            frame_idx, CAMERA_NAME_STR,
            extrinsic_cam2vehicle, vehicle_pose, extrinsic_cam2world,
        )

        # ─── (6) Intrinsic 파라미터 저장 ───
        save_intrinsic_text(
            os.path.join(OUTPUT_DIRS['intrinsic'], f"{prefix}.txt"),
            frame_idx, CAMERA_NAME_STR,
            intrinsic_raw, K_matrix, distortion, width, height,
        )

    # ────────────────────────── 완료 메시지 ──────────────────────────
    print("\n✓ 모든 프레임 추출 완료!")
    print(f"✓ 출력 위치: {OUTPUT_ROOT}/")
    print(f"  ├── image/      : 원본 이미지 (.jpg)")
    print(f"  ├── depth/      : Depth Map (.npy = float meter, .jpg = 시각화)")
    print(f"  ├── panoptic/   : Panoptic (.npy = ID, .jpg = 시각화, _info.txt = 클래스 매핑)")
    print(f"  ├── extrinsic/  : Camera Extrinsic (.txt) [Waymo 원본 + cam2world 변환본]")
    print(f"  └── intrinsic/  : Camera Intrinsic (.txt) [Waymo 원본 + 표준 K 행렬]")