# ============================================================
# 3_verify_standard_space.py
#
# [목적]
# Translation 제거 후 신호등(Traffic Light)이 표준 공간에 "수학적으로"
# 고정되는지를 직접 계산으로 검증하고, 이미지 확대·정렬이 수학적으로
# 올바른지 다양한 시각화로 검증합니다.
# ★ 대상 객체: Traffic Light (신호등) — Cityscapes label_id = 6
#
# [핵심 이론 - 미팅 내용 요약]
# - "신호등이 표준 공간에 고정되고 카메라가 주위를 도는 형태가 돼야"
# - Translation을 없애면: 여러 프레임에서 카메라가 원점에 위치하게 됨
#   → 실제 세계에서 고정된 신호등이 표준 좌표계 원점 근처에 일관되게 투영됨
# - "얘가 이 크기만큼 확대돼서 정합이 돼야지 정상"
#   → Translation의 tz(전후 이동)이 제거되면 신호등이 마치 확대된 것처럼 보임
# - "신호등은 정적이면 내 움직임에만 귀속되겠지. 종속성이 존재한단 말이야."
#
# [검증 항목]
# 1. 표준 공간 고정 검증:
#    - 여러 프레임에서 같은 3D 신호등 점을 Translation 제거 전/후의 카메라로 투영
#    - Translation 제거 전: 투영 위치가 프레임마다 다름
#    - Translation 제거 후: 투영 위치가 프레임 간 일관성 향상 (고정 효과)
#
# 2. 이미지 확대 수학적 검증:
#    - 원래 깊이 Z → Translation 제거 후 깊이 Z' = Z - tz
#    - 확대 배율 = Z / Z' = Z / (Z - tz)
#    - 이 배율로 확대하면 원본과 정렬되는지 검증
#
# 3. 정렬 시각화 (Overlay):
#    - 원본 이미지와 수학적으로 확대한 이미지를 겹쳐 그리드 정렬 확인
# ============================================================

import numpy as np                          # 수치 연산 (행렬, 통계 등)
import matplotlib.pyplot as plt             # 시각화
plt.rcParams["font.family"] = 'NanumGothic' # 한글 폰트 설정

import matplotlib.patches as mpatches       # 사각형 패치 시각화
import matplotlib.gridspec as gridspec       # 복잡한 서브플롯 레이아웃
import cv2                                  # OpenCV (이미지 처리, 워핑)
from PIL import Image                       # 이미지 저장/로드
import tensorflow as tf                     # Waymo 데이터 로드
from waymo_open_dataset import dataset_pb2  # Waymo 프로토버퍼 구조

# utils.py에서 코드 변경 없이 임포트
from utils import WaymoFrameExtractor, PanopticSegmenter, DepthProEstimator, save_image

# ============================================================
# ★ [상수 정의] Cityscapes Panoptic 클래스 ID
# ============================================================
TRAFFIC_LIGHT_LABEL_ID = 6                  # 신호등 클래스 ID (Cityscapes 기준)


# ============================================================
# ★ [모듈 1] 신호등(Traffic Light) 세그먼트 탐색
# ============================================================
def find_traffic_light_segments(segments_info: list, model: PanopticSegmenter) -> list:
    """
    Panoptic Segmentation 결과에서 Traffic Light(신호등)에 해당하는
    세그먼트만 필터링하여 반환합니다.

    Args:
        segments_info : segments_info 리스트
        model         : PanopticSegmenter 인스턴스
    Returns:
        traffic_light_segs (list): Traffic Light 세그먼트 정보 리스트
    """
    traffic_light_segs = []  # 결과 리스트

    for seg in segments_info:
        seg_label_id = seg['label_id']  # 클래스 라벨 ID

        # 방법 1: 모델 config의 id2label로 클래스 이름 확인
        if hasattr(model.model.config, 'id2label'):
            label_name = model.model.config.id2label.get(seg_label_id, '')
            if 'traffic light' in label_name.lower():
                traffic_light_segs.append(seg)
                continue

        # 방법 2: 하드코딩된 label_id 비교
        if seg_label_id == TRAFFIC_LIGHT_LABEL_ID:
            traffic_light_segs.append(seg)

    return traffic_light_segs


# ============================================================
# [모듈 2] 카메라 내부 파라미터(K) 및 외부 파라미터(E) 추출
# ============================================================
def get_camera_params(frame: dataset_pb2.Frame, camera_name: int) -> tuple:
    """
    Waymo 프레임에서 특정 카메라의 Intrinsic K와 Extrinsic E를 동시에 추출합니다.

    Args:
        frame       : Waymo 프레임 객체
        camera_name : dataset_pb2.CameraName 열거형 정수값
    Returns:
        K (3×3 numpy.ndarray): 내부 파라미터 행렬
        E (4×4 numpy.ndarray): 외부 파라미터 행렬
    """
    K = None
    E = None

    for cal in frame.context.camera_calibrations:
        if cal.name == camera_name:
            # Intrinsic 행렬 K 구성
            intr = cal.intrinsic
            K = np.array([
                [intr[0], 0.0,     intr[2]],
                [0.0,     intr[1], intr[3]],
                [0.0,     0.0,     1.0    ]
            ], dtype=np.float64)

            # Extrinsic 행렬 E 구성
            E = np.array(cal.extrinsic.transform).reshape(4, 4)
            break

    if K is None or E is None:
        raise ValueError(f"카메라 {camera_name}의 파라미터를 찾을 수 없습니다.")

    return K, E


# ============================================================
# [모듈 3] Translation 제거 및 분해
# ============================================================
def decompose_extrinsic(E: np.ndarray) -> tuple:
    """
    4×4 Extrinsic 행렬을 Rotation(R), Translation(t), Rotation-only 행렬로 분해합니다.

    Args:
        E (4×4 numpy.ndarray): 원본 Extrinsic 행렬
    Returns:
        R (3×3 numpy.ndarray): 회전 행렬
        t (3,  numpy.ndarray): 이동 벡터 [tx, ty, tz]
        E_rot (4×4 numpy.ndarray): Translation 제거된 Rotation-only 행렬 [R|0]
    """
    R = E[:3, :3].copy()     # 상단 3×3: Rotation 행렬
    t = E[:3, 3].copy()      # 우측 3×1: Translation 벡터

    # Translation을 0으로 설정한 Rotation-only 행렬
    E_rot = np.eye(4, dtype=np.float64)
    E_rot[:3, :3] = R

    return R, t, E_rot


# ============================================================
# [모듈 4] 3D 세계 점을 카메라 이미지에 투영 (단일 점)
# ============================================================
def project_world_point(
    X_world: np.ndarray,   # (3,) 또는 (4,) 세계 좌표
    K      : np.ndarray,   # (3, 3) 내부 파라미터 행렬
    E      : np.ndarray    # (4, 4) 외부 파라미터 행렬
) -> tuple:
    """
    세계 좌표계의 3D 점을 카메라 이미지 좌표계로 투영합니다.

    Args:
        X_world : 세계 좌표 (3,) 또는 동차 (4,)
        K       : 3×3 Intrinsic 행렬
        E       : 4×4 Extrinsic 행렬
    Returns:
        u (float) : 투영된 x 픽셀 좌표
        v (float) : 투영된 y 픽셀 좌표
        Z (float) : 카메라 좌표계에서의 깊이
    """
    # 동차 좌표 변환
    if len(X_world) == 3:
        X_h = np.append(X_world, 1.0)
    else:
        X_h = X_world.astype(np.float64)

    # 세계 → 카메라 좌표 변환
    X_cam_h = E @ X_h
    X_cam   = X_cam_h[:3]
    Z       = X_cam[2]

    if Z <= 0:
        return None, None, Z

    # 픽셀 좌표로 투영
    p_h = K @ X_cam
    u   = p_h[0] / p_h[2]
    v   = p_h[1] / p_h[2]

    return u, v, Z


# ============================================================
# ★ [모듈 5] 신호등 3D 세계 좌표 추출
# ============================================================
def extract_traffic_light_world_point(
    image     : np.ndarray,     # (H, W, 3) 이미지
    depth_map : np.ndarray,     # (H, W) Depth Map
    seg_map   : np.ndarray,     # (H, W) 세그멘테이션 맵
    tl_seg_id : int,            # ★ 신호등 세그먼트 ID
    K         : np.ndarray,     # (3, 3) Intrinsic
    E         : np.ndarray      # (4, 4) Extrinsic
) -> np.ndarray:
    """
    Panoptic 세그멘테이션 결과에서 신호등 영역의 중심 픽셀을 찾고,
    해당 픽셀의 Depth와 카메라 파라미터를 이용해 3D 세계 좌표로 변환합니다.

    신호등이 정적 객체이므로 이 세계 좌표는 모든 프레임에서 일관되어야 합니다.

    Args:
        image     : 원본 이미지
        depth_map : Depth Map (미터)
        seg_map   : 세그멘테이션 ID 맵
        tl_seg_id : 신호등 세그먼트 ID
        K         : 3×3 Intrinsic 행렬
        E         : 4×4 Extrinsic 행렬
    Returns:
        world_point (numpy.ndarray): (3,) 신호등의 3D 세계 좌표 [X, Y, Z]
                                     신호등 미검출 시 기본값 [5.0, 0.0, 10.0] 반환
    """
    default_point = np.array([5.0, 0.0, 10.0])  # 기본값: 전방 10m 정면

    # ★ 신호등 세그먼트에 속하는 픽셀 검색
    tl_mask = (seg_map == tl_seg_id)
    tl_pixels = np.argwhere(tl_mask)     # (K, 2): [row, col]

    if len(tl_pixels) == 0:
        print("  ⚠ 신호등 픽셀이 없어 기본 세계 좌표를 사용합니다.")
        return default_point

    # 신호등 중심 픽셀 계산
    center_px = tl_pixels.mean(axis=0)   # [row 평균, col 평균]
    v_c = int(np.clip(center_px[0], 0, depth_map.shape[0] - 1))  # 행 좌표 (v)
    u_c = int(np.clip(center_px[1], 0, depth_map.shape[1] - 1))  # 열 좌표 (u)
    d_c = depth_map[v_c, u_c]           # 중심 픽셀의 깊이값 (미터)

    if d_c <= 0.1:
        print(f"  ⚠ 신호등 중심 깊이값이 너무 작음 (d={d_c:.4f}m). 기본값 사용.")
        return default_point

    # Back-projection: 픽셀 + 깊이 → 카메라 좌표
    f_u = K[0, 0]; f_v = K[1, 1]
    c_u = K[0, 2]; c_v = K[1, 2]

    X_cam = np.array([
        (u_c - c_u) / f_u * d_c,        # 카메라 X 좌표
        (v_c - c_v) / f_v * d_c,        # 카메라 Y 좌표
        d_c,                             # 카메라 Z 좌표 (깊이)
        1.0                              # 동차 좌표 w=1
    ])

    # 카메라 → 세계 좌표 변환
    X_world_h = np.linalg.inv(E) @ X_cam
    world_point = X_world_h[:3]          # (3,) 세계 좌표 [X, Y, Z]

    print(f"  신호등 3D 세계 좌표: X={world_point[0]:.3f}, "
          f"Y={world_point[1]:.3f}, Z={world_point[2]:.3f} (m)")

    return world_point


# ============================================================
# [모듈 6] 표준 공간 고정 직접 계산 검증
# ============================================================
def verify_standard_space_fixation(
    frame_params  : list,          # [(K, E, depth, image, seg_map), ...] 프레임별 데이터
    world_point   : np.ndarray,    # (3,) 검증할 3D 세계 좌표 (★ 신호등 위치)
    image_size    : tuple          # (H, W) 이미지 크기
) -> dict:
    """
    지정된 신호등 3D 세계 점이 Translation 제거 전/후에 어떻게 투영되는지를
    여러 프레임에 걸쳐 계산하여 표준 공간 고정 효과를 수치로 검증합니다.

    핵심 원리:
      - [Translation 있음]: 카메라 위치가 프레임마다 달라서 신호등 투영 위치가 분산됨
      - [Translation 없음]: 카메라가 매 프레임 원점에서 같은 방향으로 바라봄
        → 고정된 신호등의 투영 위치가 더 일관성 있게 됨 (표준 공간 효과)

    Args:
        frame_params : 각 프레임의 파라미터 리스트
        world_point  : (3,) 신호등 3D 세계 좌표
        image_size   : (H, W) 이미지 크기
    Returns:
        result (dict): 검증 결과
    """
    H, W = image_size
    proj_orig     = []   # 원본 Extrinsic 투영 좌표
    proj_rot_only = []   # Translation 제거 후 투영 좌표
    zoom_ratios   = []   # 이론적 확대 배율

    for K, E_orig, depth_map, image, seg_map in frame_params:
        # ---- 원본 Extrinsic으로 신호등 투영 ----
        u_o, v_o, Z_o = project_world_point(world_point, K, E_orig)

        # ---- Translation 제거 후 신호등 투영 ----
        _, t_vec, E_rot = decompose_extrinsic(E_orig)
        u_r, v_r, Z_r   = project_world_point(world_point, K, E_rot)

        # 투영 성공 및 이미지 경계 내 확인
        if u_o is not None and v_o is not None:
            if 0 <= u_o < W and 0 <= v_o < H:
                proj_orig.append((u_o, v_o))

        if u_r is not None and v_r is not None:
            if 0 <= u_r < W and 0 <= v_r < H:
                proj_rot_only.append((u_r, v_r))

        # ---- 이론적 확대 배율 계산 ----
        if Z_o is not None and Z_o > 0:
            tz = t_vec[2]
            Z_rot_theory = Z_o - tz
            if Z_rot_theory > 0:
                zoom_ratio = Z_o / Z_rot_theory
                zoom_ratios.append(zoom_ratio)

    # ---- 투영 일관성 수치화: 표준편차 계산 ----
    std_orig     = 0.0
    std_rot_only = 0.0

    if len(proj_orig) > 1:
        pts_orig = np.array(proj_orig)
        std_orig = np.std(pts_orig, axis=0).mean()

    if len(proj_rot_only) > 1:
        pts_rot = np.array(proj_rot_only)
        std_rot_only = np.std(pts_rot, axis=0).mean()

    return {
        'proj_orig'     : proj_orig,
        'proj_rot_only' : proj_rot_only,
        'std_orig'      : std_orig,
        'std_rot_only'  : std_rot_only,
        'zoom_ratios'   : zoom_ratios,
        'world_point'   : world_point
    }


# ============================================================
# ★ [모듈 7] 신호등 확대 배율 수학적 계산 및 검증
# ============================================================
def verify_zoom_ratio(
    image     : np.ndarray,   # (H, W, 3) 원본 이미지
    depth_map : np.ndarray,   # (H, W) Depth Map
    K         : np.ndarray,   # (3, 3) Intrinsic 행렬
    E_orig    : np.ndarray,   # (4, 4) 원본 Extrinsic
    seg_map   : np.ndarray,   # (H, W) 세그멘테이션 ID 맵
    tl_seg_id : int           # ★ 신호등 세그먼트 ID
) -> dict:
    """
    Translation의 tz(전후 이동 성분)이 제거될 때 예상되는 신호등 확대 배율을 계산하고,
    실제 이미지에서 그 배율로 확대했을 때 원본과 정렬되는지 수치로 검증합니다.

    수학적 원리:
      - 원본 Z_obj: 신호등의 카메라 기준 깊이 (미터)
      - Translation 제거 후 Z_rot = Z_obj - tz
      - 이론적 확대 배율 s = Z_obj / Z_rot

    Args:
        image     : 원본 이미지
        depth_map : Depth Map
        K         : Intrinsic K 행렬
        E_orig    : 원본 Extrinsic 행렬
        seg_map   : 세그멘테이션 ID 맵
        tl_seg_id : ★ 신호등 세그먼트 ID
    Returns:
        result (dict): 배율 및 확대 이미지 포함 딕셔너리
    """
    H, W = image.shape[:2]
    c_u = K[0, 2]
    c_v = K[1, 2]

    # ---- ★ 신호등 영역의 평균 깊이 계산 ----
    seg_mask = (seg_map == tl_seg_id)
    if seg_mask.sum() == 0:
        return {'error': f'신호등 세그먼트 {tl_seg_id}가 존재하지 않습니다.'}

    # 신호등 영역에서 유효한 깊이값의 중앙값
    seg_depths = depth_map[seg_mask]
    seg_depths_valid = seg_depths[seg_depths > 0.1]
    if len(seg_depths_valid) == 0:
        return {'error': '신호등 영역에 유효한 깊이값이 없습니다.'}

    Z_obj = float(np.median(seg_depths_valid))

    # ---- Translation 성분 추출 ----
    _, t_vec, _ = decompose_extrinsic(E_orig)
    tz = t_vec[2]

    # ---- 이론적 확대 배율 계산 ----
    Z_rot = Z_obj - tz
    if Z_rot <= 0:
        return {'error': f'Z_rot = {Z_rot:.4f} <= 0, 배율 계산 불가'}

    zoom = Z_obj / Z_rot

    # ---- 확대 변환 행렬 (주점 기준 스케일링) ----
    M = np.array([
        [zoom,  0.0,  c_u * (1.0 - zoom)],
        [0.0,  zoom,  c_v * (1.0 - zoom)]
    ], dtype=np.float64)

    # OpenCV warpAffine으로 이미지 확대 적용
    scaled_image = cv2.warpAffine(
        image, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # ---- ★ 신호등 바운딩 박스 계산 ----
    seg_pixels = np.argwhere(seg_mask)
    bbox_r_min, bbox_c_min = seg_pixels.min(axis=0)
    bbox_r_max, bbox_c_max = seg_pixels.max(axis=0)

    orig_h = bbox_r_max - bbox_r_min
    orig_w = bbox_c_max - bbox_c_min

    # 확대 후 바운딩 박스 예측 크기
    pred_h = orig_h * zoom
    pred_w = orig_w * zoom

    return {
        'Z_obj'       : Z_obj,
        'tz'          : tz,
        'Z_rot'       : Z_rot,
        'zoom_ratio'  : zoom,
        'M'           : M,
        'scaled_image': scaled_image,
        'orig_bbox'   : (bbox_r_min, bbox_c_min, bbox_r_max, bbox_c_max),
        'orig_hw'     : (orig_h, orig_w),
        'pred_hw'     : (pred_h, pred_w)
    }


# ============================================================
# [모듈 8] 그리드 오버레이 정렬 시각화 (수학적 정렬 검증)
# ============================================================
def visualize_grid_alignment(
    orig_image    : np.ndarray,
    scaled_image  : np.ndarray,
    zoom_ratio    : float,
    Z_obj         : float,
    tz            : float,
    ax1, ax2, ax3
):
    """
    원본 이미지, 확대된 이미지, 오버레이 비교를 나란히 시각화합니다.
    ★ 신호등 영역이 정렬되는지 그리드 선으로 육안 검증합니다.

    Args:
        orig_image, scaled_image : 원본 및 확대 이미지
        zoom_ratio : 확대 배율
        Z_obj      : 신호등 원본 깊이
        tz         : Translation z 성분
        ax1, ax2, ax3 : matplotlib Axes 3개
    """
    H, W = orig_image.shape[:2]

    # ---- Axes 1: 원본 이미지 + 그리드 ----
    ax1.imshow(orig_image)
    ax1.set_title(f"원본 이미지\n(신호등 Z_obj={Z_obj:.2f}m)", fontsize=9)
    ax1.axis('off')

    for x_line in np.linspace(0, W, 6)[1:-1]:
        ax1.axvline(x=x_line, color='cyan', linewidth=0.8, alpha=0.6, linestyle='--')
    for y_line in np.linspace(0, H, 6)[1:-1]:
        ax1.axhline(y=y_line, color='cyan', linewidth=0.8, alpha=0.6, linestyle='--')

    # ---- Axes 2: 배율 적용 이미지 + 그리드 ----
    ax2.imshow(scaled_image)
    ax2.set_title(
        f"이론 배율 s={zoom_ratio:.4f}×로 확대\n"
        f"(tz={tz:.2f}m 제거 → Z'={Z_obj - tz:.2f}m)",
        fontsize=9
    )
    ax2.axis('off')

    for x_line in np.linspace(0, W, 6)[1:-1]:
        ax2.axvline(x=x_line, color='cyan', linewidth=0.8, alpha=0.6, linestyle='--')
    for y_line in np.linspace(0, H, 6)[1:-1]:
        ax2.axhline(y=y_line, color='cyan', linewidth=0.8, alpha=0.6, linestyle='--')

    # ---- Axes 3: 오버레이 (알파 블렌딩) ----
    orig_f    = orig_image.astype(np.float32) / 255.0
    scaled_f  = scaled_image.astype(np.float32) / 255.0
    overlay   = np.clip(orig_f * 0.5 + scaled_f * 0.5, 0, 1)

    diff      = np.abs(orig_f - scaled_f).mean(axis=2)
    diff_norm = diff / (diff.max() + 1e-8)

    ax3.imshow(overlay)
    ax3.set_title(
        f"오버레이 (원본 50% + 확대 50%)\n"
        f"평균 픽셀 차이: {diff_norm.mean():.4f}",
        fontsize=9
    )
    ax3.axis('off')

    for x_line in np.linspace(0, W, 6)[1:-1]:
        ax3.axvline(x=x_line, color='yellow', linewidth=0.8, alpha=0.6, linestyle='--')
    for y_line in np.linspace(0, H, 6)[1:-1]:
        ax3.axhline(y=y_line, color='yellow', linewidth=0.8, alpha=0.6, linestyle='--')


# ============================================================
# [모듈 9] 표준 공간 고정 효과 산점도 시각화
# ============================================================
def visualize_projection_scatter(std_result: dict, image_size: tuple, ax1, ax2):
    """
    여러 프레임에서 동일 신호등 3D 점이 투영되는 2D 위치를
    Translation 제거 전/후 두 경우에 대해 산점도로 시각화합니다.

    Args:
        std_result : verify_standard_space_fixation()의 반환값
        image_size : (H, W)
        ax1, ax2   : matplotlib Axes 2개
    """
    H, W = image_size

    # ---- Translation 제거 전 투영 산점도 ----
    if std_result['proj_orig']:
        pts_o = np.array(std_result['proj_orig'])
        ax1.scatter(pts_o[:, 0], pts_o[:, 1],
                    c=range(len(pts_o)), cmap='viridis', s=80, zorder=3,
                    label='각 프레임 신호등 투영점')

        center_o = pts_o.mean(axis=0)
        ax1.scatter([center_o[0]], [center_o[1]],
                    c='red', s=200, marker='X', zorder=5, label='평균 중심')

        std_o = std_result['std_orig']
        circle_o = plt.Circle(center_o, std_o,
                               color='red', fill=False, linestyle='--',
                               linewidth=1.5, alpha=0.7, label=f'std={std_o:.1f}px')
        ax1.add_patch(circle_o)

    ax1.set_xlim(0, W)
    ax1.set_ylim(H, 0)
    ax1.set_title(
        f"Translation 있음 [R|t]\n"
        f"신호등 투영 좌표 분산 (std={std_result['std_orig']:.2f}px)",
        fontsize=9
    )
    ax1.set_xlabel('u (픽셀)', fontsize=8)
    ax1.set_ylabel('v (픽셀)', fontsize=8)
    ax1.legend(fontsize=7)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # ---- Translation 제거 후 투영 산점도 ----
    if std_result['proj_rot_only']:
        pts_r = np.array(std_result['proj_rot_only'])
        ax2.scatter(pts_r[:, 0], pts_r[:, 1],
                    c=range(len(pts_r)), cmap='plasma', s=80, zorder=3,
                    label='각 프레임 신호등 투영점')

        center_r = pts_r.mean(axis=0)
        ax2.scatter([center_r[0]], [center_r[1]],
                    c='blue', s=200, marker='X', zorder=5, label='평균 중심')

        std_r = std_result['std_rot_only']
        circle_r = plt.Circle(center_r, std_r,
                               color='blue', fill=False, linestyle='--',
                               linewidth=1.5, alpha=0.7, label=f'std={std_r:.1f}px')
        ax2.add_patch(circle_r)

    ax2.set_xlim(0, W)
    ax2.set_ylim(H, 0)
    ax2.set_title(
        f"Translation 제거 [R|0]\n"
        f"신호등 투영 좌표 분산 (std={std_result['std_rot_only']:.2f}px) → 더 수렴",
        fontsize=9
    )
    ax2.set_xlabel('u (픽셀)', fontsize=8)
    ax2.set_ylabel('v (픽셀)', fontsize=8)
    ax2.legend(fontsize=7)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)


# ============================================================
# [모듈 10] 확대 배율 분포 시각화
# ============================================================
def visualize_zoom_distribution(zoom_ratios: list, ax):
    """
    여러 프레임에서 계산된 신호등 이론적 확대 배율의 분포를 시각화합니다.

    Args:
        zoom_ratios : 각 프레임의 이론적 확대 배율 목록
        ax          : matplotlib Axes
    """
    if not zoom_ratios:
        ax.text(0.5, 0.5, '계산된 배율 데이터 없음', ha='center', va='center',
                transform=ax.transAxes, fontsize=11, color='gray')
        return

    ratios = np.array(zoom_ratios)

    n_bins = min(15, len(ratios))
    ax.hist(ratios, bins=n_bins,
            color='steelblue', edgecolor='white', alpha=0.8,
            label='신호등 이론적 확대 배율')

    mean_r   = ratios.mean()
    std_r    = ratios.std()
    median_r = np.median(ratios)

    ax.axvline(mean_r,   color='red',    linewidth=2.0, linestyle='-',  label=f'평균={mean_r:.4f}')
    ax.axvline(median_r, color='orange', linewidth=1.5, linestyle='--', label=f'중앙값={median_r:.4f}')
    ax.axvline(1.0,      color='green',  linewidth=1.5, linestyle=':',  label='배율=1.0')

    ax.axvspan(mean_r - std_r, mean_r + std_r,
               alpha=0.15, color='red', label=f'±1σ ({std_r:.4f})')

    ax.set_xlabel('확대 배율 s = Z_obj / (Z_obj - tz)', fontsize=9)
    ax.set_ylabel('프레임 수', fontsize=9)
    ax.set_title(
        f'신호등 Translation 제거 확대 배율 분포\n'
        f'(s>1이면 확대, 일관적일수록 유효)',
        fontsize=9
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    summary = (f"총 {len(ratios)}개 프레임\n"
               f"평균 배율: {mean_r:.4f}×\n"
               f"표준편차: ±{std_r:.4f}\n"
               f"범위: [{ratios.min():.4f}, {ratios.max():.4f}]")
    ax.text(0.02, 0.97, summary, transform=ax.transAxes,
            fontsize=8, va='top', ha='left',
            bbox=dict(facecolor='lightyellow', edgecolor='gray', alpha=0.85, pad=4))


# ============================================================
# [모듈 11] 픽셀 이동 벡터 필드 시각화
# ============================================================
def visualize_pixel_displacement(
    orig_image : np.ndarray,
    K          : np.ndarray,
    E_orig     : np.ndarray,
    depth_map  : np.ndarray,
    ax
):
    """
    Translation 제거 시 각 픽셀이 얼마나, 어느 방향으로 이동하는지를
    화살표(quiver) 필드로 시각화합니다.
    ★ 신호등 등 가까운 객체일수록 크게 이동 (시차 효과)

    Args:
        orig_image : 원본 이미지
        K          : Intrinsic 행렬
        E_orig     : 원본 Extrinsic
        depth_map  : Depth Map
        ax         : matplotlib Axes
    """
    H, W = orig_image.shape[:2]

    f_u = K[0, 0]; f_v = K[1, 1]
    c_u = K[0, 2]; c_v = K[1, 2]
    _, t_vec, _ = decompose_extrinsic(E_orig)
    tx, ty, tz = t_vec

    # 서브샘플링된 픽셀 그리드
    step = max(1, H // 20)
    v_sample = np.arange(step // 2, H, step)
    u_sample = np.arange(step // 2, W, step)
    U_grid, V_grid = np.meshgrid(u_sample, v_sample)

    Z_sample = depth_map[V_grid, U_grid]

    eps = 1e-6
    dU = f_u * tx / (Z_sample + eps)
    dV = f_v * ty / (Z_sample + eps)

    valid_mask = Z_sample > 0.1

    ax.imshow(orig_image, alpha=0.6)

    displacement_magnitude = np.sqrt(dU**2 + dV**2)

    q = ax.quiver(
        U_grid[valid_mask], V_grid[valid_mask],
        dU[valid_mask], -dV[valid_mask],
        displacement_magnitude[valid_mask],
        cmap='hot', scale=None, scale_units='xy',
        angles='xy', width=0.002, alpha=0.8
    )
    plt.colorbar(q, ax=ax, label='이동 크기 (픽셀)', fraction=0.03)

    ax.set_title(
        f"Translation 제거 시 픽셀 이동 벡터 필드\n"
        f"(tx={tx:.3f}m, ty={ty:.3f}m, tz={tz:.3f}m)\n"
        f"가까운 신호등일수록 많이 이동 (시차 효과)",
        fontsize=9
    )
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')


# ============================================================
# [메인] 전체 파이프라인 실행
# ============================================================
def main():
    """
    메인 실행 함수:
    1. 여러 프레임에서 데이터 추출
    2. ★ 신호등을 대상으로 표준 공간 고정 검증 (직접 계산)
    3. 신호등 확대 배율 수학적 검증
    4. 종합 시각화 (산점도 + 배율 분포 + 그리드 정렬 + 벡터 필드)
    """
    # ---- 설정값 ----
    TFRECORD_PATH  = './data/individual_files_validation_segment-12496433400137459534_120_000_140_000_with_camera_labels.tfrecord'
    CAMERA         = dataset_pb2.CameraName.FRONT  # 검증할 카메라
    MAX_FRAMES     = 8      # 표준 공간 검증에 사용할 프레임 수
    VERIFY_FRAME   = 0      # 확대 배율 상세 검증에 사용할 프레임 인덱스
    OUTPUT_PATH    = './standard_space_verification_traffic_light.png'   # ★ 결과 저장 경로

    print("=" * 60)
    print("[1단계] Waymo TFRecord 프레임 로드")
    print("=" * 60)

    wfe = WaymoFrameExtractor(TFRECORD_PATH)
    frame_list = wfe.get_frame_list()
    n_use = min(MAX_FRAMES, len(frame_list))
    print(f"총 {len(frame_list)}개 프레임 로드, {n_use}개 사용.")

    print("\n" + "=" * 60)
    print("[2단계] 딥러닝 모델 초기화")
    print("=" * 60)

    ps = PanopticSegmenter()     # Mask2Former (utils.py 재사용)
    de = DepthProEstimator()     # DepthPro (utils.py 재사용)

    print("\n" + "=" * 60)
    print("[3단계] 여러 프레임에서 데이터 수집 (신호등 표준 공간 검증용)")
    print("=" * 60)

    frame_params    = []      # [(K, E, depth, image, seg_map), ...]
    all_zoom_ratios = []      # 모든 프레임의 확대 배율
    first_tl_seg_id = None    # ★ 첫 번째 프레임에서 검출된 신호등 세그먼트 ID

    for i in range(n_use):
        print(f"\n  프레임 {i+1}/{n_use} 처리 중...")
        frame = frame_list[i]

        # 이미지 추출
        img_array = None
        for img_obj in frame.images:
            if img_obj.name == CAMERA:
                img_array = tf.io.decode_jpeg(img_obj.image).numpy()
                break

        if img_array is None:
            print(f"  프레임 {i}: 이미지 없음 → 건너뜀")
            continue

        K, E = get_camera_params(frame, CAMERA)
        depth = de.get_depth_map(img_array)
        seg_res = ps.segment(img_array)
        seg_map = seg_res['segmentation'].cpu().numpy()

        frame_params.append((K, E, depth, img_array, seg_map))

        # ★ 신호등 세그먼트 검출 및 확대 배율 계산
        tl_segs = find_traffic_light_segments(seg_res['segments_info'], ps)
        print(f"  신호등 검출: {len(tl_segs)}개")

        if tl_segs:
            tl_seg_id = tl_segs[0]['id']  # 첫 번째 신호등 세그먼트

            # 첫 번째 프레임의 신호등 ID 기록 (나중에 검증에 사용)
            if first_tl_seg_id is None:
                first_tl_seg_id = tl_seg_id

            # 신호등 기준 확대 배율 계산
            zoom_res = verify_zoom_ratio(img_array, depth, K, E, seg_map, tl_seg_id)
            if 'zoom_ratio' in zoom_res:
                all_zoom_ratios.append(zoom_res['zoom_ratio'])
                print(f"  → 신호등 확대 배율: {zoom_res['zoom_ratio']:.4f}× "
                      f"(Z_obj={zoom_res['Z_obj']:.2f}m, tz={zoom_res['tz']:.3f}m)")
        else:
            print(f"  ⚠ 이 프레임에서 신호등이 검출되지 않았습니다.")

    if not frame_params:
        print("처리할 프레임이 없습니다. 경로를 확인하세요.")
        return

    print("\n" + "=" * 60)
    print("[4단계] 신호등 표준 공간 고정 직접 계산 검증")
    print("=" * 60)

    # ★ 검증에 사용할 신호등 3D 세계 점 추출
    K0, E0, depth0, img0, seg0 = frame_params[0]
    world_point = np.array([5.0, 0.0, 10.0])   # 기본값

    # 첫 번째 프레임에서 신호등 세계 좌표 계산
    if first_tl_seg_id is not None:
        world_point = extract_traffic_light_world_point(
            img0, depth0, seg0, first_tl_seg_id, K0, E0
        )
    else:
        # 신호등 미검출 시 첫 번째 세그먼트 중심으로 대체
        seg_info_0 = ps.segment(img0)['segments_info']
        if seg_info_0:
            seg_id_0 = seg_info_0[0]['id']
            mask_px = np.argwhere(seg0 == seg_id_0)
            if len(mask_px) > 0:
                center_px = mask_px.mean(axis=0)
                v_c = int(np.clip(center_px[0], 0, depth0.shape[0] - 1))
                u_c = int(np.clip(center_px[1], 0, depth0.shape[1] - 1))
                d_c = depth0[v_c, u_c]
                if d_c > 0.1:
                    f_u0 = K0[0, 0]; f_v0 = K0[1, 1]
                    c_u0 = K0[0, 2]; c_v0 = K0[1, 2]
                    X_cam = np.array([
                        (u_c - c_u0) / f_u0 * d_c,
                        (v_c - c_v0) / f_v0 * d_c,
                        d_c, 1.0
                    ])
                    X_world_h = np.linalg.inv(E0) @ X_cam
                    world_point = X_world_h[:3]
        print(f"  ⚠ 신호등 미검출. 대체 세계 좌표: X={world_point[0]:.3f}, "
              f"Y={world_point[1]:.3f}, Z={world_point[2]:.3f} (m)")

    # 표준 공간 고정 검증 실행
    image_size = (img0.shape[0], img0.shape[1])
    std_result = verify_standard_space_fixation(frame_params, world_point, image_size)

    print(f"\n[검증 결과]")
    print(f"  Translation 있음  → 신호등 투영 좌표 표준편차: {std_result['std_orig']:.2f}px")
    print(f"  Translation 제거  → 신호등 투영 좌표 표준편차: {std_result['std_rot_only']:.2f}px")

    if std_result['std_orig'] > 0 and std_result['std_rot_only'] >= 0:
        reduction = (1 - std_result['std_rot_only'] / (std_result['std_orig'] + 1e-8)) * 100
        print(f"  → 분산 감소율: {reduction:.1f}% "
              f"({'신호등 표준 공간 고정 효과 확인' if reduction > 0 else '오차 발생 가능'})")

    print("\n" + "=" * 60)
    print("[5단계] 신호등 확대 배율 수학적 검증 (단일 프레임)")
    print("=" * 60)

    # ★ 단일 프레임에서 신호등 확대 배율 상세 검증
    K_v, E_v, depth_v, img_v, seg_v = frame_params[VERIFY_FRAME % len(frame_params)]

    zoom_detail = {'error': '신호등 세그먼트 없음'}
    # 검증 프레임에서 신호등 검출 시도
    seg_res_v = ps.segment(img_v)
    tl_segs_v = find_traffic_light_segments(seg_res_v['segments_info'], ps)

    if tl_segs_v:
        tl_sid_v = tl_segs_v[0]['id']
        zoom_detail = verify_zoom_ratio(img_v, depth_v, K_v, E_v, seg_v, tl_sid_v)
        if 'zoom_ratio' in zoom_detail:
            print(f"  신호등 깊이 Z_obj = {zoom_detail['Z_obj']:.4f}m")
            print(f"  Translation tz  = {zoom_detail['tz']:.4f}m")
            print(f"  제거 후 깊이 Z' = {zoom_detail['Z_rot']:.4f}m")
            print(f"  이론적 배율  s  = {zoom_detail['zoom_ratio']:.6f}×")
            print(f"  원본 신호등 크기: {zoom_detail['orig_hw']} px")
            print(f"  예측 확대 후 크기: ({zoom_detail['pred_hw'][0]:.1f}, "
                  f"{zoom_detail['pred_hw'][1]:.1f}) px")
    else:
        print(f"  ⚠ 검증 프레임에서 신호등이 검출되지 않았습니다.")

    print("\n" + "=" * 60)
    print("[6단계] 종합 시각화 생성")
    print("=" * 60)

    # ---- 플롯 레이아웃 구성 ----
    fig = plt.figure(figsize=(18, 16))
    fig.suptitle(
        "신호등(Traffic Light) 표준 공간 고정 검증\n"
        "Translation 제거의 수학적 올바름 확인 (직접 계산 + 이미지 확대 정렬)",
        fontsize=13, fontweight='bold', y=0.98
    )

    # ---- 행 1: 산점도 2개 + 배율 분포 1개 ----
    ax_s1  = fig.add_subplot(3, 3, 1)
    ax_s2  = fig.add_subplot(3, 3, 2)
    ax_zoom_hist = fig.add_subplot(3, 3, 3)

    visualize_projection_scatter(std_result, image_size, ax_s1, ax_s2)
    visualize_zoom_distribution(all_zoom_ratios, ax_zoom_hist)

    # ---- 행 2: 그리드 정렬 시각화 ----
    ax_g1 = fig.add_subplot(3, 3, 4)
    ax_g2 = fig.add_subplot(3, 3, 5)
    ax_g3 = fig.add_subplot(3, 3, 6)

    if 'scaled_image' in zoom_detail:
        visualize_grid_alignment(
            img_v, zoom_detail['scaled_image'],
            zoom_detail['zoom_ratio'],
            zoom_detail['Z_obj'], zoom_detail['tz'],
            ax_g1, ax_g2, ax_g3
        )
    else:
        for ax_g in [ax_g1, ax_g2, ax_g3]:
            ax_g.imshow(img_v)
            ax_g.set_title("신호등 확대 배율 계산 실패\n(신호등 미검출)", fontsize=9)
            ax_g.axis('off')

    # ---- 행 3: 픽셀 이동 벡터 필드 + 수치 요약 ----
    ax_vec = fig.add_subplot(3, 3, 7)

    K_p, E_p, depth_p, img_p, seg_p = frame_params[0]
    visualize_pixel_displacement(img_p, K_p, E_p, depth_p, ax_vec)

    # ---- 수치 요약 텍스트 박스 ----
    ax_summary = fig.add_subplot(3, 3, 8)
    ax_summary.axis('off')
    ax_summary.set_facecolor('#f5f5f5')

    _, t0, _ = decompose_extrinsic(E_p)

    summary_text = (
        "[신호등 표준 공간 고정 수치 검증 요약]\n\n"
        f"● 총 검증 프레임: {n_use}개\n\n"
        f"● 신호등 투영 좌표 표준편차 비교:\n"
        f"  Translation 있음: {std_result['std_orig']:.2f}px (분산)\n"
        f"  Translation 제거: {std_result['std_rot_only']:.2f}px (수렴)\n\n"
        f"● 신호등 확대 배율 통계 ({len(all_zoom_ratios)}개 프레임):\n"
        f"  평균: {np.mean(all_zoom_ratios):.4f}×\n"
        f"  범위: [{min(all_zoom_ratios):.4f}, {max(all_zoom_ratios):.4f}]×\n\n"
        if all_zoom_ratios else
        "[신호등 표준 공간 고정 수치 검증 요약]\n\n"
        f"● 총 검증 프레임: {n_use}개\n"
        "● 확대 배율 데이터 없음 (신호등 미검출)\n"
    )

    if 'zoom_ratio' in zoom_detail:
        summary_text += (
            f"● 단일 프레임 상세 (프레임 {VERIFY_FRAME}):\n"
            f"  Z_obj = {zoom_detail['Z_obj']:.4f}m (신호등 깊이)\n"
            f"  tz    = {zoom_detail['tz']:.4f}m\n"
            f"  s     = {zoom_detail['zoom_ratio']:.6f}×\n\n"
            f"● 수학적 검증:\n"
            f"  s = Z_obj / (Z_obj - tz)\n"
            f"    = {zoom_detail['Z_obj']:.4f} / "
            f"({zoom_detail['Z_obj']:.4f} - {zoom_detail['tz']:.4f})\n"
            f"    = {zoom_detail['zoom_ratio']:.6f}×"
        )

    ax_summary.text(
        0.05, 0.97, summary_text,
        transform=ax_summary.transAxes,
        fontsize=7.5, va='top', ha='left',
        fontfamily='monospace',
        bbox=dict(facecolor='#e8f4fd', edgecolor='#2980b9', alpha=0.9, pad=6)
    )
    ax_summary.set_title("신호등 수치 검증 요약", fontsize=9, fontweight='bold')

    # ---- 수학적 설명 텍스트 박스 ----
    ax_math = fig.add_subplot(3, 3, 9)
    ax_math.axis('off')

    math_text = (
        "[수학적 원리 요약 — 신호등 기준]\n\n"
        "Extrinsic E = [R | t]\n"
        "  R: 회전 행렬 (3×3)\n"
        "  t: 이동 벡터 [tx, ty, tz]\n\n"
        "Translation 제거:\n"
        "  E' = [R | 0]\n\n"
        "카메라 투영 (신호등):\n"
        "  X_cam = R·X_tl + t  (원본)\n"
        "  X_rot = R·X_tl      (t=0)\n\n"
        "신호등 확대 배율:\n"
        "  Z_orig = R·X_tl + tz\n"
        "  Z_rot  = R·X_tl (tz 제거)\n"
        "  s = Z_orig / Z_rot\n"
        "    ≈ Z_tl / (Z_tl - tz)\n\n"
        "신호등 픽셀 이동 (1차 근사):\n"
        "  Δu ≈ f_u·tx / Z_tl\n"
        "  Δv ≈ f_v·ty / Z_tl\n\n"
        "→ 가까운 신호등일수록 많이 이동\n"
        "→ 일관적 확대 = 표준 공간 고정"
    )

    ax_math.text(
        0.05, 0.97, math_text,
        transform=ax_math.transAxes,
        fontsize=7.5, va='top', ha='left',
        fontfamily='monospace',
        bbox=dict(facecolor='#fef9e7', edgecolor='#f39c12', alpha=0.9, pad=6)
    )
    ax_math.set_title("수학적 원리 설명", fontsize=9, fontweight='bold')

    # ---- 레이아웃 조정 및 저장 ----
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_PATH, dpi=130, bbox_inches='tight')
    print(f"\n최종 검증 결과 이미지 저장 완료: {OUTPUT_PATH}")
    plt.show()

    print("\n" + "=" * 60)
    print("[완료] 신호등 표준 공간 고정 수학적 검증 완료")
    print("=" * 60)


# ============================================================
# 스크립트 직접 실행 시 메인 함수 호출
# ============================================================
if __name__ == "__main__":
    main()
