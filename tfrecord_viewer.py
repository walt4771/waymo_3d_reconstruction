"""
Waymo RGB-D 기반 3D Reconstruction Baseline
============================================
파이프라인:
  1. LiDAR → 각 카메라에 투영 → sparse depth map 생성 (per camera)
  2. depth map + RGB image + intrinsic → 카메라 frame 3D 포인트 unproject
  3. camera extrinsic + vehicle pose 적용 → world frame 누적
  4. PLY 저장 + Open3D 시각화

핵심 가정:
  - Waymo는 dense depth map을 제공하지 않음 → LiDAR 투영으로 sparse depth 생성
  - Distortion은 무시 (pinhole only). 작은 오차로 나타날 텐데 그 자체도 관찰 대상.

Waymo 카메라 좌표 규약 (헷갈리기 쉬움, 주의!):
  - +x = 광축(forward, 깊이 방향)
  - +y = 이미지 LEFT
  - +z = 이미지 UP
  투영:   u = cu - fu * y/x,   v = cv - fv * z/x
  역투영: x = d,  y = -(u-cu)*d/fu,  z = -(v-cv)*d/fv
"""

import os                                                # OS 환경변수 접근 (TF/CUDA 설정용)
# ↓ TF가 import 되기 "전에" 설정해야 효과 있음. import 후엔 늦음.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"                # 모든 GPU 숨김 → TF가 CPU만 사용 (초기화 오버헤드 제거)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"                 # TF C++ 백엔드 로그 억제 (0=다, 1=info, 2=warn, 3=error만)

from pathlib import Path                                 # 경로를 객체로 다루기 (문자열 / 연산자로 합쳐짐)
import time                                              # 구간별 실행 시간 측정용 (time.time())
import numpy as np                                       # 포인트 클라우드/행렬 연산의 메인 라이브러리
import cv2                                               # OpenCV - JPEG 디코딩이 TF보다 훨씬 빠름
import tensorflow as tf                                  # Waymo가 TFRecord 포맷이라 TF가 필수
import open3d as o3d                                     # 3D 시각화 + PLY/PCD I/O
from tqdm import tqdm                                    # 루프 진행률 바

from waymo_open_dataset import dataset_pb2               # Waymo의 protobuf 정의 (Frame, CameraName 등)
from waymo_open_dataset.utils import frame_utils         # range image → point cloud 변환 헬퍼

tf.config.set_visible_devices([], "GPU")                 # Python 레벨에서도 GPU 끔 (이중 안전장치)


# ============================================================
# 설정
# ============================================================
TFRECORD_PATH = "./data/validation/individual_files_validation_segment-1071392229495085036_1844_790_1864_790_with_camera_labels.tfrecord"
                                                          # ↑ 처리할 Waymo segment 파일 경로 (1 segment ≈ 약 200 frame)
OUTPUT_DIR = Path("./output")                            # 출력물 저장 폴더 (PLY, npz 캐시 등)
OUTPUT_DIR.mkdir(exist_ok=True)                          # 폴더 없으면 생성, 있으면 그냥 통과

MAX_FRAMES        = 198                                  # 처리할 frame 수 (segment 전체)
TARGET_CAMERAS    = [
    dataset_pb2.CameraName.FRONT,                        # 일단 FRONT 카메라 하나만 (빠른 디버깅용)
]                                                        # CameraName enum: FRONT=1, FRONT_LEFT=2, FRONT_RIGHT=3, SIDE_LEFT=4, SIDE_RIGHT=5
PER_FRAME_VOXEL   = 0                                    # frame당 voxel 다운샘플 크기 (m). 0이면 비활성
FINAL_VOXEL       = 0.10                                 # 모든 frame 누적 후 최종 다운샘플 크기 (m)
MIN_DEPTH         = 0.5                                  # 너무 가까운 LiDAR 포인트 (차체 자체 등) 제외 임계값 (m)
MAX_DEPTH         = 80.0                                 # 너무 먼 노이즈성 포인트 제외 (m). Waymo LiDAR 유효 범위 ~75m
CACHE_NPZ         = True                                 # True면 누적 결과를 npz로 저장 → 다음 실행 시 frame 루프 건너뜀


# ============================================================
# Numpy voxel downsample (Open3D 객체 생성 회피로 빠름)
# ============================================================
def voxel_down_np(points, colors, voxel):
    """
    포인트 클라우드를 voxel 단위로 다운샘플링 (각 voxel당 대표 포인트 1개).

    Args:
        points (np.ndarray): shape (N, 3), float, XYZ 좌표.
        colors (np.ndarray): shape (N, 3), float, RGB (0~1 정규화 가정).
        voxel (float): voxel 한 변 크기 (m). 0 이하면 no-op.

    Returns:
        (np.ndarray, np.ndarray):
            points_down: shape (M, 3), M ≤ N, 다운샘플된 XYZ.
            colors_down: shape (M, 3), 동일하게 줄어든 RGB.
    """
    if voxel <= 0 or len(points) == 0:                   # voxel=0이거나 입력이 비었으면 그대로 반환 (no-op)
        return points, colors
    idx = np.floor(points / voxel).astype(np.int64)      # 각 포인트를 voxel 격자 인덱스로 변환 (정수)
    idx -= idx.min(axis=0)                               # 인덱스를 0부터 시작하도록 평행이동 (음수 인덱스 방지)
    sizes = idx.max(axis=0) + 1                          # 각 축 voxel 개수 (해싱 시 multiplier로 사용)
    # ↓ (x_idx, y_idx, z_idx) 3차원 인덱스를 하나의 int64 키로 패킹 → np.unique 한 번에 처리 가능
    key = idx[:, 0] * (sizes[1] * sizes[2]) + idx[:, 1] * sizes[2] + idx[:, 2]
    _, first = np.unique(key, return_index=True)         # 각 unique 키에서 가장 먼저 등장한 포인트의 인덱스 (대표 1개씩)
    return points[first], colors[first]                  # 대표 포인트와 그에 매칭되는 색만 반환


# ============================================================
# Calibration 한 번만 파싱
# ============================================================
class CamCalib:
    """
    카메라 캘리브레이션 보관 (intrinsic + extrinsic).

    Attributes:
        name (int): CameraName enum 값 (1~5).
        fu, fv (float): 초점거리 (pixel).
        cu, cv (float): 주점(principal point) 픽셀 좌표.
        H, W (int): 이미지 height, width.
        extrinsic (np.ndarray): shape (4, 4), camera → vehicle 변환.
        inv_extrinsic (np.ndarray): shape (4, 4), vehicle → camera 변환.
    """
    __slots__ = ("name", "fu", "fv", "cu", "cv", "H", "W", "extrinsic", "inv_extrinsic")
                                                          # ↑ __slots__: 인스턴스 dict 안 만들어 메모리/속도 약간 이득

    def __init__(self, cam_calib):
        """
        Args:
            cam_calib: Waymo protobuf의 CameraCalibration 메시지 객체.
        """
        self.name = cam_calib.name                       # CameraName enum 값 (1~5)
        intr = cam_calib.intrinsic                       # Waymo 내부 파라미터 9개 배열: [fu, fv, cu, cv, k1, k2, p1, p2, k3]
        self.fu, self.fv, self.cu, self.cv = intr[0], intr[1], intr[2], intr[3]
                                                          # ↑ 초점거리(fu,fv) + 주점(cu,cv). distortion(k1~k3) 무시
        self.W = cam_calib.width                         # 이미지 가로 픽셀 수
        self.H = cam_calib.height                        # 이미지 세로 픽셀 수
        # camera → vehicle (4x4 row-major)
        self.extrinsic = np.array(cam_calib.extrinsic.transform, dtype=np.float64).reshape(4, 4)
                                                          # ↑ 16개 float를 4x4 변환행렬로 reshape
        self.inv_extrinsic = np.linalg.inv(self.extrinsic)  # vehicle → camera (depth map 만들 때 LiDAR 변환에 사용)


def parse_calibrations(frame):
    """
    Frame에서 모든 카메라의 캘리브레이션 정보 추출.
    캘리브는 segment 내내 동일 → 첫 frame에서 1번만 호출하면 됨.

    Args:
        frame: Waymo Frame protobuf 메시지.

    Returns:
        dict: {camera_name (int): CamCalib 객체} 매핑.
    """
    return {c.name: CamCalib(c) for c in frame.context.camera_calibrations}
                                                          # ↑ frame.context.camera_calibrations: 5개 카메라 캘리브 정보 list


# ============================================================
# Depth map ↔ Unproject
# ============================================================
def build_depth_map(points_vehicle, calib):
    """
    LiDAR 포인트(vehicle frame)를 카메라 이미지 평면에 투영해 sparse depth map 생성.
    Z-buffer로 픽셀당 가장 가까운 depth만 유지.

    Args:
        points_vehicle (np.ndarray): shape (N, 3), vehicle 좌표계의 LiDAR 포인트.
        calib (CamCalib): 대상 카메라의 캘리브레이션.

    Returns:
        np.ndarray: shape (H, W), float32, depth map.
                    0인 픽셀 = LiDAR hit 없음. 0보다 크면 해당 픽셀의 depth (m).
    """
    N = points_vehicle.shape[0]                          # 입력 LiDAR 포인트 개수
    pts_h = np.concatenate([points_vehicle, np.ones((N, 1), dtype=np.float32)], axis=1)
                                                          # ↑ (N,3) → (N,4) homogeneous 좌표 (행렬곱으로 affine 변환하려고)
    pts_cam = (calib.inv_extrinsic @ pts_h.T).T[:, :3]   # vehicle → camera 변환. (4,4)@(4,N)=(4,N) → transpose → (N,3)만 추출

    x = pts_cam[:, 0]                                    # camera frame x = depth (광축 = 앞쪽)
    y = pts_cam[:, 1]                                    # camera frame y = 이미지 left (Waymo 규약 주의)
    z = pts_cam[:, 2]                                    # camera frame z = 이미지 up

    front = (x > MIN_DEPTH) & (x < MAX_DEPTH)            # 너무 가깝거나 멀거나 카메라 뒤(x≤0)인 포인트 제외
    x, y, z = x[front], y[front], z[front]               # 마스킹된 포인트만 남김
    if x.size == 0:                                      # 유효 포인트 0개면 빈 depth map 반환 (edge case)
        return np.zeros((calib.H, calib.W), dtype=np.float32)

    # Waymo pinhole projection (좌표 규약 주의: y/z 앞에 마이너스)
    u = calib.cu - calib.fu * y / x                      # u(가로 픽셀) = 주점 - 초점거리 * (y/x). y>0(left)이면 u 작아짐(left)
    v = calib.cv - calib.fv * z / x                      # v(세로 픽셀) = 주점 - 초점거리 * (z/x). z>0(up)이면 v 작아짐(top)

    u_int = np.round(u).astype(np.int32)                 # 실수 픽셀 좌표 → 정수로 반올림 (depth map 인덱싱용)
    v_int = np.round(v).astype(np.int32)                 # 동일
    in_img = (u_int >= 0) & (u_int < calib.W) & (v_int >= 0) & (v_int < calib.H)
                                                          # ↑ 이미지 경계 안에 들어오는 픽셀만 유효
    u_int, v_int, depth = u_int[in_img], v_int[in_img], x[in_img]  # 유효 마스크 적용

    depth_map = np.zeros((calib.H, calib.W), dtype=np.float32)  # 빈 depth map 초기화 (0 = no data)
    # 먼 것부터 채우고 → 가까운 것이 덮어쓰도록 (간이 z-buffer)
    order = np.argsort(-depth)                           # depth 내림차순 인덱스 (먼 것부터)
    depth_map[v_int[order], u_int[order]] = depth[order] # 같은 픽셀에 여러 포인트 들어오면 마지막(=가장 가까운)이 살아남음
    return depth_map


def unproject_depth_map(depth_map, rgb_image, calib):
    """
    sparse depth map + RGB image → 카메라 frame의 3D 포인트와 색.
    LiDAR hit 있는 픽셀만 unproject.

    Args:
        depth_map (np.ndarray): shape (H, W), float32. 0 = no data, >0 = depth (m).
        rgb_image (np.ndarray): shape (H, W, 3), uint8, 0~255 RGB.
        calib (CamCalib): 해당 카메라의 캘리브레이션.

    Returns:
        (np.ndarray, np.ndarray):
            pts_cam: shape (M, 3), float32, 카메라 좌표계의 3D 포인트.
            colors:  shape (M, 3), float32, 0~1 정규화된 RGB.
    """
    valid = depth_map > 0                                # 0이 아닌 픽셀 = LiDAR hit 있는 픽셀
    v_idx, u_idx = np.where(valid)                       # 2D mask → 유효 픽셀의 (행, 열) 인덱스 배열
    d = depth_map[v_idx, u_idx]                          # 해당 픽셀의 depth 값들 (1D array)

    # Waymo pinhole unprojection (projection의 역연산)
    x_cam = d                                            # x_cam은 정의상 depth 그 자체
    y_cam = -(u_idx - calib.cu) * d / calib.fu           # 픽셀 u에서 주점 빼고 depth/fu 곱함. 부호 -는 Waymo 규약
    z_cam = -(v_idx - calib.cv) * d / calib.fv           # 동일. 부호 빼먹으면 상하/좌우 반전 → 디버깅 1순위
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1).astype(np.float32)
                                                          # ↑ 세 1D 배열을 합쳐 (N, 3) 카메라 frame 포인트 클라우드

    colors = rgb_image[v_idx, u_idx, :3].astype(np.float32) / 255.0
                                                          # ↑ 같은 픽셀에서 RGB 추출 → 0~1 정규화 (Open3D 컬러 규격)
    return pts_cam, colors


def cam_to_world(pts_cam, calib, vehicle_pose):
    """
    카메라 좌표계의 포인트를 world 좌표계로 변환 (camera → vehicle → world).

    Args:
        pts_cam (np.ndarray): shape (N, 3), 카메라 좌표계 포인트.
        calib (CamCalib): camera→vehicle 변환을 위한 extrinsic 포함.
        vehicle_pose (np.ndarray): shape (4, 4), vehicle → world 변환 (frame.pose.transform).

    Returns:
        np.ndarray: shape (N, 3), world 좌표계 포인트.
    """
    cam_to_world_mat = (vehicle_pose @ calib.extrinsic).astype(np.float32)
                                                          # ↑ 두 변환행렬 합성 (cam→vehicle→world를 한 행렬로). float32로 다운캐스트
    N = pts_cam.shape[0]                                 # 포인트 개수
    pts_h = np.concatenate([pts_cam, np.ones((N, 1), dtype=np.float32)], axis=1)
                                                          # ↑ homogeneous 좌표화 (N, 3) → (N, 4)
    return (cam_to_world_mat @ pts_h.T).T[:, :3]         # 합성 행렬 한 번 곱하고 마지막 1 제거


# ============================================================
# Waymo helpers
# ============================================================
def parse_frame(data):
    """
    TFRecord의 raw bytes를 Waymo Frame protobuf 메시지로 역직렬화.

    Args:
        data (tf.Tensor): TFRecord에서 한 레코드를 꺼낸 tensor (string).

    Returns:
        dataset_pb2.Frame: 파싱된 Frame 메시지.
    """
    frame = dataset_pb2.Frame()                          # 빈 Frame protobuf 객체 생성
    frame.ParseFromString(bytearray(data.numpy()))       # TFRecord의 raw bytes → Frame 메시지로 역직렬화
    return frame


def extract_lidar_vehicle(frame):
    """
    Frame의 모든 LiDAR(5개)의 first return을 vehicle 좌표계로 합쳐 반환.

    Args:
        frame (dataset_pb2.Frame): Waymo Frame 메시지.

    Returns:
        np.ndarray: shape (N_total, 3), float32, vehicle 좌표계 LiDAR 포인트.
    """
    ri, cp, _, ri_top_pose = frame_utils.parse_range_image_and_camera_projection(frame)
                                                          # ↑ range image / camera projection / segmentation / top pose 분리
                                                          # ↑ ri: {laser_name: [return1, return2]} range image dict
                                                          # ↑ cp: 카메라 투영 정보 (사용 안 함, 우리가 직접 투영하기 때문)
                                                          # ↑ ri_top_pose: TOP LiDAR의 ego-motion compensation에 필요
    points, _ = frame_utils.convert_range_image_to_point_cloud(
        frame, ri, cp, ri_top_pose, ri_index=0           # ri_index=0 → first return만 (second return 무시, 속도 2배)
    )                                                     # 반환: points는 5개 LiDAR별 (Ni, 3) array의 list
    return np.concatenate(points, axis=0).astype(np.float32)
                                                          # ↑ 5개 LiDAR 합쳐서 (N_total, 3) 단일 array로


def get_camera_image(frame, camera_name):
    """
    Frame에서 지정한 카메라의 JPEG 이미지를 디코딩해 RGB ndarray로 반환.

    Args:
        frame (dataset_pb2.Frame): Waymo Frame 메시지.
        camera_name (int): CameraName enum 값 (1~5).

    Returns:
        np.ndarray or None: shape (H, W, 3), uint8, RGB 이미지. 못 찾으면 None.
    """
    for image in frame.images:                           # frame.images는 5개 카메라 이미지 list
        if image.name == camera_name:                    # 원하는 카메라 찾으면
            arr = np.frombuffer(image.image, dtype=np.uint8)  # JPEG 바이트를 numpy uint8 배열로 (디코딩 전 raw bytes)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)    # OpenCV로 JPEG 디코딩 → BGR ndarray (H, W, 3)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR → RGB 변환 (Open3D/일반 컨벤션)
    return None                                          # 못 찾으면 None


# ============================================================
# Main
# ============================================================
def run():
    """
    메인 파이프라인 실행:
      TFRecord 로드 → frame 순회 → 카메라별 depth map/unproject → world 누적
      → 다운샘플 → PLY 저장 → Open3D 시각화.
    캐시 npz가 있으면 frame 루프는 건너뛰고 시각화만 다시 수행.
    """
    cache_path = OUTPUT_DIR / f"cache_rgbd_{MAX_FRAMES}.npz"  # 캐시 npz 경로

    if CACHE_NPZ and cache_path.exists():                # 캐시 옵션 켜져있고 파일이 있으면
        print(f"[cache hit] {cache_path}")               # 캐시 사용 알림
        d = np.load(cache_path)                          # npz 로드
        all_points, all_colors, trajectory = d["points"], d["colors"], d["traj"]  # 키별로 꺼냄
    else:                                                # 캐시 없으면 처음부터 처리
        dataset = tf.data.TFRecordDataset(TFRECORD_PATH, compression_type="")
                                                          # ↑ TFRecord를 stream으로 읽음 (전체를 메모리에 안 올림)
        all_points, all_colors, trajectory = [], [], []  # frame별 결과를 누적할 리스트
        calibs = None                                    # 캘리브는 첫 frame에서만 파싱 (lazy init)
        t0 = time.time()                                 # 루프 시작 시각 기록

        pbar = tqdm(total=MAX_FRAMES, desc="frames")     # 진행률 바 (전체 MAX_FRAMES 개수 기준)
        for i, data in enumerate(dataset):               # TFRecord에서 frame 하나씩 꺼냄
            if i >= MAX_FRAMES:                          # 지정한 개수만큼만 처리하고 중단
                break
            frame = parse_frame(data)                    # raw bytes → Frame protobuf
            if calibs is None:                           # 첫 frame에서만 캘리브 파싱
                calibs = parse_calibrations(frame)       # {camera_name: CamCalib} 채움

            # 1) LiDAR (vehicle frame) — depth map의 소스
            pts_v = extract_lidar_vehicle(frame)         # 모든 LiDAR의 first return을 vehicle 좌표계로

            # 2) Vehicle pose — frame마다 바뀜
            pose = np.array(frame.pose.transform, dtype=np.float64).reshape(4, 4)
                                                          # ↑ frame.pose.transform: 16개 float (row-major). vehicle → world 변환

            frame_pts, frame_cols = [], []               # 이 frame에서 카메라들이 만들어낸 3D 포인트/색을 임시 저장
            for cam_name in TARGET_CAMERAS:              # 설정된 카메라들 순회
                calib = calibs.get(cam_name)             # 해당 카메라 캘리브 찾기
                if calib is None:                        # 없으면 (예외 케이스) 건너뜀
                    continue

                # 3) sparse depth map (LiDAR → camera projection)
                depth_map = build_depth_map(pts_v, calib)  # vehicle LiDAR → camera 투영 → depth map
                if not depth_map.any():                  # depth map이 전부 0이면 (유효 hit 없음) 건너뜀
                    continue

                # 4) RGB
                rgb = get_camera_image(frame, cam_name)  # 동일 카메라의 JPEG 이미지 디코딩
                if rgb is None:                          # 이미지 없으면 건너뜀
                    continue

                # 5) depth + RGB → camera frame 3D points
                pts_cam, cols = unproject_depth_map(depth_map, rgb, calib)
                                                          # ↑ sparse depth의 유효 픽셀들만 3D 포인트로 unproject + RGB 매칭
                if pts_cam.shape[0] == 0:                # 그래도 유효 포인트 없으면 건너뜀
                    continue

                # 6) camera → world
                pts_w = cam_to_world(pts_cam, calib, pose)  # camera → vehicle → world 합성 변환
                frame_pts.append(pts_w)                  # 이 카메라가 만든 포인트 추가
                frame_cols.append(cols)                  # 색도 추가

            if not frame_pts:                            # 이번 frame에서 아무 카메라도 결과 못 만들었으면
                pbar.update(1)                           # 진행률만 올리고
                continue                                  # 다음 frame으로
            pts_w = np.concatenate(frame_pts, axis=0)    # 카메라별 결과를 한 frame 단위로 합침
            cols = np.concatenate(frame_cols, axis=0)    # 색도 동일하게
            pts_w, cols = voxel_down_np(pts_w, cols, PER_FRAME_VOXEL)
                                                          # ↑ frame 단위 다운샘플 (메모리 절약). 0이면 no-op

            all_points.append(pts_w)                     # 전체 누적 리스트에 추가
            all_colors.append(cols)                      # 색도 추가
            trajectory.append(pose[:3, 3].astype(np.float32))  # vehicle 위치(translation 부분만) 궤적에 추가

            pbar.update(1)                               # 진행률 +1
            pbar.set_postfix(pts=f"{pts_w.shape[0]:>6d}")  # 진행률 옆에 현재 frame 포인트 수 표시
        pbar.close()                                     # 진행률 바 종료

        all_points = np.concatenate(all_points, axis=0)  # 모든 frame을 합쳐 (N_total, 3) 단일 배열로
        all_colors = np.concatenate(all_colors, axis=0)  # 색도 동일
        trajectory = np.array(trajectory)                # frame별 vehicle 위치 → (MAX_FRAMES, 3) 배열
        print(f"frame loop: {time.time() - t0:.1f}s")    # 루프 총 소요 시간 출력

        if CACHE_NPZ:                                    # 캐시 저장 옵션 켜져 있으면
            np.savez(cache_path, points=all_points, colors=all_colors, traj=trajectory)
            print(f"cached: {cache_path}")               # 캐시 저장 완료 알림

    print(f"\nAccumulated points: {all_points.shape[0]:,}")  # 누적 포인트 총 개수 (천 단위 콤마)
    print(f"World X: [{all_points[:,0].min():.1f}, {all_points[:,0].max():.1f}]")  # world 좌표계 X 범위
    print(f"World Y: [{all_points[:,1].min():.1f}, {all_points[:,1].max():.1f}]")  # Y 범위
    print(f"World Z: [{all_points[:,2].min():.1f}, {all_points[:,2].max():.1f}]")  # Z 범위 (지면 부근이면 거의 평평)

    # 최종 다운샘플
    t1 = time.time()                                     # 다운샘플 시작 시각
    pts_f, col_f = voxel_down_np(all_points, all_colors, FINAL_VOXEL)
                                                          # ↑ 모든 frame 누적된 포인트 클라우드를 통째로 voxel down → 중복 제거
    print(f"final voxel({FINAL_VOXEL}m): {pts_f.shape[0]:,}  [{time.time()-t1:.1f}s]")  # 결과 통계

    pcd = o3d.geometry.PointCloud()                      # Open3D 포인트 클라우드 객체 생성
    pcd.points = o3d.utility.Vector3dVector(pts_f.astype(np.float64))   # XYZ 설정 (Open3D는 float64 요구)
    pcd.colors = o3d.utility.Vector3dVector(col_f.astype(np.float64))   # RGB 설정 (0~1 범위)
    ply_path = OUTPUT_DIR / f"recon_rgbd_{MAX_FRAMES}.ply"  # 출력 PLY 경로
    o3d.io.write_point_cloud(str(ply_path), pcd)         # PLY로 저장 (MeshLab/CloudCompare 등에서 열림)
    print(f"saved: {ply_path}")                          # 저장 완료 알림

    traj_pcd = o3d.geometry.PointCloud()                 # 궤적도 별도 포인트 클라우드로
    traj_pcd.points = o3d.utility.Vector3dVector(trajectory.astype(np.float64))  # 궤적 XYZ 설정
    traj_pcd.paint_uniform_color([1.0, 0.0, 0.0])        # 빨간색으로 통일 (재구성 결과와 시각적 구분)
    o3d.io.write_point_cloud(str(OUTPUT_DIR / f"traj_rgbd_{MAX_FRAMES}.ply"), traj_pcd)  # 궤적 PLY 저장

    o3d.visualization.draw_geometries(                   # 인터랙티브 3D 뷰어 띄우기 (마우스로 회전/줌)
        [pcd, traj_pcd], window_name=f"Waymo RGB-D recon - {MAX_FRAMES} frames"
    )


if __name__ == "__main__":                               # 스크립트로 직접 실행 시에만 run() 호출 (import 시엔 X)
    run()
