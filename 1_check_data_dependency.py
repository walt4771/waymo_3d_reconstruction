# ============================================================
# 1_check_data_dependency_front_only.py
#
# [목적]
# FRONT 카메라 단일 뷰에서
# Mask(파노픽 세그멘테이션), Depth Map, Camera Extrinsic Parameter
# 세 데이터 사이의 종속성을 수치 및 시각적으로 검증합니다.
# ★ 대상 객체: Traffic Light (신호등) — Cityscapes label_id = 6
#
# [원본 대비 변경 사항]
# - 두 카메라(FRONT + FRONT_LEFT) 비교 → FRONT 단일 카메라로 단순화
# - compare_backprojection() 제거
#   → analyze_single_camera_backprojection() 으로 대체
#   → 단일 카메라 내에서 Depth 분포·3D 점군을 분석
# - 플롯 레이아웃: 3×4 → 2×4 (카메라 비교 행 제거)
# - 유지: 모든 모듈 함수(1~7, 9, 10), 실험 목적, 클래스 임포트
#
# [핵심 이론 - 미팅 내용 요약]
# - Depth는 카메라의 Extrinsic/Intrinsic 파라미터에 의해 완전히 결정됩니다.
#   → "카메라 파라미터를 알고 있는 한, 이 값은 절대적이야. 데이터 종속성이야."
# - Mask는 Depth와 Camera Params가 있어야 3D 공간의 위치를 특정할 수 있습니다.
# - "신호등이라는 이 데이터는 굉장히 개방돼 있는 걸 전제로 한단 말이야."
#   → 신호등은 정적(static) 객체이므로 카메라 포즈에만 종속됩니다.
#
# [검증 항목]
# 1. FRONT 카메라의 신호등 Mask 영역 내 Depth 분포 확인
# 2. Mask × Depth × Extrinsic 조합으로 신호등의 3D 점군 및 중심(centroid) 추출
# 3. Translation 성분 제거 후 Rotation-only 정렬 결과 비교
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'NanumGothic'

import matplotlib.patches as mpatches
from PIL import Image
import tensorflow as tf
from waymo_open_dataset import dataset_pb2

from utils import WaymoFrameExtractor, PanopticSegmenter, DepthProEstimator, save_image

# ============================================================
# ★ [상수 정의] Cityscapes Panoptic 클래스 ID
# ============================================================
TRAFFIC_LIGHT_LABEL_ID = 6  # 신호등 클래스 ID (Cityscapes 기준)
#   0=road, 1=sidewalk, 2=building, 3=wall, 4=fence, 5=pole,
#   6=traffic light, 7=traffic sign, 8=vegetation, ...


# ============================================================
# [모듈 1] 신호등(Traffic Light) 세그먼트 탐색
# ============================================================
def find_traffic_light_segments(segments_info: list, model: PanopticSegmenter) -> list:
    """
    Panoptic Segmentation 결과에서 Traffic Light(신호등)에 해당하는
    세그먼트만 필터링하여 반환합니다.

    Args:
        segments_info : Panoptic Segmentation 결과의 segments_info 리스트
        model         : PanopticSegmenter 인스턴스 (id2label 매핑 확인용)
    Returns:
        traffic_light_segs (list): Traffic Light 세그먼트 정보 딕셔너리 리스트
    """
    traffic_light_segs = []

    for seg in segments_info:
        seg_label_id = seg['label_id']

        # 방법 1: 모델의 id2label 매핑으로 클래스 이름 직접 확인
        if hasattr(model.model.config, 'id2label'):
            label_name = model.model.config.id2label.get(seg_label_id, '')
            if 'traffic light' in label_name.lower():
                traffic_light_segs.append(seg)
                continue

        # 방법 2: 하드코딩된 Cityscapes label_id 비교 (폴백)
        if seg_label_id == TRAFFIC_LIGHT_LABEL_ID:
            traffic_light_segs.append(seg)

    return traffic_light_segs


# ============================================================
# [모듈 2] 카메라 내부 파라미터(Intrinsic) 추출
# ============================================================
def get_camera_intrinsics(frame: dataset_pb2.Frame, camera_name: int) -> np.ndarray:
    """
    Waymo 프레임에서 특정 카메라의 3×3 내부 파라미터 행렬 K를 반환합니다.

    Waymo intrinsic 배열 순서: [f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3]

    Args:
        frame      : Waymo 프레임 객체
        camera_name: dataset_pb2.CameraName 열거형 정수값
    Returns:
        K (3×3 numpy.ndarray): 카메라 내부 파라미터 행렬
    Raises:
        ValueError: 해당 카메라 캘리브레이션 정보가 없을 때
    """
    for cal in frame.context.camera_calibrations:
        if cal.name == camera_name:
            intr = cal.intrinsic
            K = np.array([
                [intr[0], 0.0,     intr[2]],
                [0.0,     intr[1], intr[3]],
                [0.0,     0.0,     1.0    ]
            ], dtype=np.float64)
            return K

    raise ValueError(f"camera_name={camera_name} 에 대한 캘리브레이션 정보가 없습니다.")


# ============================================================
# [모듈 3] 카메라 이미지 추출
# ============================================================
def extract_camera_image(frame: dataset_pb2.Frame, camera_name: int) -> np.ndarray:
    """
    Waymo 프레임에서 특정 카메라의 이미지를 JPEG 디코딩하여 numpy 배열로 반환합니다.

    Args:
        frame      : Waymo 프레임 객체
        camera_name: dataset_pb2.CameraName 열거형 정수값
    Returns:
        image_array (numpy.ndarray): RGB 이미지 배열 (H, W, 3), dtype=uint8
    Raises:
        ValueError: 해당 카메라 이미지가 없을 때
    """
    for img in frame.images:
        if img.name == camera_name:
            return tf.io.decode_jpeg(img.image).numpy()

    raise ValueError(f"camera_name={camera_name} 에 대한 이미지가 없습니다.")


# ============================================================
# [모듈 4] 픽셀 → 3D 세계 좌표 역투영 (Back-projection)
# ============================================================
def backproject_to_world(
    pixel_coords: np.ndarray,
    depth_values: np.ndarray,
    K: np.ndarray,
    extrinsic: np.ndarray
) -> np.ndarray:
    """
    픽셀 좌표와 Depth 값을 3D 세계(차량) 좌표계로 역투영합니다.

    변환 과정:
        1. 픽셀 좌표 → 정규화 카메라 좌표: K^{-1} * [u, v, 1]^T
        2. 정규화 좌표 × Depth → 3D 카메라 좌표: X_cam
        3. 카메라 좌표 → 세계 좌표: X_world = E^{-1} * X_cam

    Args:
        pixel_coords : (N, 2) 픽셀 좌표 배열 [(u, v), ...]
        depth_values : (N,) 깊이값 배열 (미터 단위)
        K            : (3, 3) 내부 파라미터 행렬
        extrinsic    : (4, 4) 외부 파라미터 행렬 (세계 → 카메라)
    Returns:
        world_coords (numpy.ndarray): (N, 3) 세계 좌표 배열
    """
    N = pixel_coords.shape[0]
    ones = np.ones((N, 1), dtype=np.float64)

    # Step 1: 동차 좌표 변환 [u, v] → [u, v, 1]
    pixels_h = np.hstack([pixel_coords, ones])           # (N, 3)

    # Step 2: K^{-1} 적용 → 정규화 카메라 좌표
    K_inv = np.linalg.inv(K)
    normalized = (K_inv @ pixels_h.T).T                  # (N, 3)

    # Step 3: Depth 곱 → 3D 카메라 좌표
    pts_cam = normalized * depth_values.reshape(-1, 1)   # (N, 3)

    # Step 4: E^{-1} 적용 → 3D 세계 좌표
    pts_cam_h = np.hstack([pts_cam, ones])               # (N, 4)
    E_inv = np.linalg.inv(extrinsic)
    pts_world_h = (E_inv @ pts_cam_h.T).T                # (N, 4)

    # 동차 w로 나눠 실제 3D 좌표 추출
    world_coords = pts_world_h[:, :3] / pts_world_h[:, 3:4]  # (N, 3)

    return world_coords


# ============================================================
# [모듈 5] Mask 영역의 대표 픽셀 샘플링
# ============================================================
def sample_pixels_from_mask(
    segmentation_map: np.ndarray,
    target_segment_id: int,
    max_samples: int = 200
) -> np.ndarray:
    """
    세그멘테이션 맵에서 특정 세그먼트 ID(신호등)에 해당하는 픽셀 좌표를 샘플링합니다.

    Args:
        segmentation_map  : (H, W) 세그멘테이션 ID 맵
        target_segment_id : 추출할 세그먼트의 ID 값
        max_samples       : 랜덤 샘플링할 최대 픽셀 수
    Returns:
        sampled_pixels (numpy.ndarray): (M, 2) 픽셀 좌표 배열 [(u, v), ...]
    """
    mask_indices = np.argwhere(segmentation_map == target_segment_id)  # [row, col]

    if len(mask_indices) == 0:
        return np.empty((0, 2), dtype=np.float64)

    if len(mask_indices) > max_samples:
        chosen = np.random.choice(len(mask_indices), max_samples, replace=False)
        mask_indices = mask_indices[chosen]

    # [row, col] → [u, v] (col=u, row=v)
    return mask_indices[:, [1, 0]].astype(np.float64)


# ============================================================
# [모듈 6] 단일 카메라에서 데이터 일괄 추출 (신호등 타겟)
# ============================================================
def extract_single_camera_data(
    frame: dataset_pb2.Frame,
    camera_name: int,
    ps_model: PanopticSegmenter,
    de_model: DepthProEstimator
) -> dict:
    """
    하나의 카메라에 대해 Image / Intrinsic K / Extrinsic E / Depth Map / Mask 를
    일괄 추출합니다. ★ 신호등(Traffic Light) 세그먼트 정보도 함께 반환합니다.

    Args:
        frame       : Waymo 프레임 객체
        camera_name : dataset_pb2.CameraName 열거형 정수값
        ps_model    : 초기화된 PanopticSegmenter 인스턴스
        de_model    : 초기화된 DepthProEstimator 인스턴스
    Returns:
        result (dict):
            'image'              : (H, W, 3) numpy 배열
            'K'                  : (3, 3) 내부 파라미터 행렬
            'E'                  : (4, 4) 외부 파라미터 행렬
            'depth'              : (H, W) 깊이 맵 (미터, float)
            'segmentation'       : (H, W) 세그멘테이션 ID 맵 (int)
            'segments_info'      : 전체 세그먼트 메타데이터 리스트
            'traffic_light_segs' : 신호등 세그먼트만 필터링한 리스트
            'camera_name'        : 카메라 이름 정수값
    """
    print(f"  [카메라 {camera_name}] 이미지 추출 중...")
    image = extract_camera_image(frame, camera_name)

    print(f"  [카메라 {camera_name}] 내부 파라미터(K) 추출 중...")
    K = get_camera_intrinsics(frame, camera_name)

    print(f"  [카메라 {camera_name}] 외부 파라미터(E) 추출 중...")
    all_extrinsics = {}
    for cal in frame.context.camera_calibrations:
        mat = np.array(cal.extrinsic.transform).reshape(4, 4)
        all_extrinsics[cal.name] = mat
    E = all_extrinsics[camera_name]

    print(f"  [카메라 {camera_name}] Panoptic Segmentation 수행 중...")
    seg_result = ps_model.segment(image)
    seg_map = seg_result['segmentation'].cpu().numpy()

    tl_segs = find_traffic_light_segments(seg_result['segments_info'], ps_model)
    print(f"  [카메라 {camera_name}] 신호등 세그먼트 {len(tl_segs)}개 검출됨")
    for idx, tl in enumerate(tl_segs):
        n_pixels = int((seg_map == tl['id']).sum())
        print(f"    → 신호등 #{idx}: segment_id={tl['id']}, "
              f"label_id={tl['label_id']}, 픽셀 수={n_pixels}")

    print(f"  [카메라 {camera_name}] Depth Estimation 수행 중...")
    depth = de_model.get_depth_map(image)

    return {
        'image'              : image,
        'K'                  : K,
        'E'                  : E,
        'depth'              : depth,
        'segmentation'       : seg_map,
        'segments_info'      : seg_result['segments_info'],
        'traffic_light_segs' : tl_segs,
        'camera_name'        : camera_name,
    }


# ============================================================
# [모듈 7] 단일 카메라 데이터 시각화 (신호등 강조)
# ============================================================
def visualize_single_camera(data: dict, ax_row: list, cam_label: str):
    """
    하나의 카메라에 대한 Image / Depth / Mask 를 한 행(row)에 시각화합니다.
    ★ 신호등 영역을 바운딩 박스로 강조 표시합니다.

    Args:
        data      : extract_single_camera_data()의 반환값 딕셔너리
        ax_row    : matplotlib Axes 객체 3개의 리스트 [ax_img, ax_depth, ax_mask]
        cam_label : 카메라 이름 문자열 (예: "FRONT")
    """
    image   = data['image']
    depth   = data['depth']
    seg     = data['segmentation']
    tl_segs = data['traffic_light_segs']

    # ---- 컬럼 0: 원본 이미지 + 신호등 바운딩 박스 ----
    ax_row[0].imshow(image)
    ax_row[0].set_title(f'{cam_label}\nRGB Image (신호등 강조)', fontsize=9)
    ax_row[0].axis('off')

    for tl in tl_segs:
        tl_mask   = (seg == tl['id'])
        tl_pixels = np.argwhere(tl_mask)
        if len(tl_pixels) > 0:
            r_min, c_min = tl_pixels.min(axis=0)
            r_max, c_max = tl_pixels.max(axis=0)
            rect = mpatches.Rectangle(
                (c_min, r_min), c_max - c_min, r_max - r_min,
                linewidth=2, edgecolor='yellow', facecolor='none',
                linestyle='-', label='Traffic Light'
            )
            ax_row[0].add_patch(rect)
            ax_row[0].text(
                c_min, r_min - 5, 'Traffic Light',
                color='yellow', fontsize=7, fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.6, pad=1)
            )

    # ---- 컬럼 1: Depth Map ----
    depth_vis = np.clip(depth, 0, 50.0)
    depth_vis = 1.0 - (depth_vis / 50.0)    # 반전: 가까울수록 밝음
    ax_row[1].imshow(depth_vis, cmap='hot')
    ax_row[1].set_title(f'{cam_label}\nDepth Map\n(밝을수록 가까움)', fontsize=9)
    ax_row[1].axis('off')

    # ---- 컬럼 2: Panoptic Segmentation Mask ----
    seg_vis = seg.astype(np.float32)
    if seg_vis.max() > 0:
        seg_vis = seg_vis / seg_vis.max() * 255
    ax_row[2].imshow(seg_vis.astype(np.uint8), cmap='tab20')
    n_tl = len(tl_segs)
    ax_row[2].set_title(
        f'{cam_label}\nPanoptic Mask\n'
        f'(전체: {len(data["segments_info"])}, 신호등: {n_tl})',
        fontsize=9
    )
    ax_row[2].axis('off')

    for tl in tl_segs:
        tl_mask   = (seg == tl['id'])
        tl_pixels = np.argwhere(tl_mask)
        if len(tl_pixels) > 0:
            r_min, c_min = tl_pixels.min(axis=0)
            r_max, c_max = tl_pixels.max(axis=0)
            rect_m = mpatches.Rectangle(
                (c_min, r_min), c_max - c_min, r_max - r_min,
                linewidth=2, edgecolor='lime', facecolor='none', linestyle='--'
            )
            ax_row[2].add_patch(rect_m)


# ============================================================
# [모듈 8-single] ★ 단일 카메라 Back-projection 분석
# (원본의 compare_backprojection 대체)
# ============================================================
def analyze_single_camera_backprojection(data: dict, cam_label: str) -> dict:
    """
    단일 카메라에서 신호등 세그먼트를 추출하고 Back-projection하여
    3D 세계 좌표 분포와 Depth 통계를 분석합니다.

    두 카메라 비교 대신, 단일 카메라 내에서
    ① Translation 포함 Extrinsic(E_full)과
    ② Translation 제거 Extrinsic(E_rot_only, t=0)
    두 가지로 역투영한 결과를 비교하여
    Translation이 3D 좌표에 미치는 영향을 수치로 보여줍니다.

    Args:
        data      : extract_single_camera_data()의 반환값 딕셔너리
        cam_label : 카메라 이름 레이블 (로그용)
    Returns:
        result (dict):
            'world_pts_full'    : (N, 3) E 원본으로 역투영한 3D 점군
            'world_pts_rot'     : (N, 3) Translation 제거 E로 역투영한 3D 점군
            'centroid_full'     : (3,) 원본 3D 중심
            'centroid_rot'      : (3,) Rotation-only 3D 중심
            'centroid_shift'    : 두 중심의 유클리드 거리 (m)
            'depth_stats'       : 신호등 Mask 내 Depth 기술 통계 딕셔너리
            'pixel_count'       : 샘플링된 픽셀 수
    """
    result = {}
    tl_segs = data['traffic_light_segs']

    if len(tl_segs) == 0:
        print(f"  [{cam_label}] ⚠ 신호등이 검출되지 않았습니다.")
        return result

    # 첫 번째 신호등 세그먼트 사용
    seg_id = tl_segs[0]['id']
    pixels = sample_pixels_from_mask(data['segmentation'], seg_id, max_samples=300)

    if len(pixels) == 0:
        print(f"  [{cam_label}] ⚠ 신호등 세그먼트 ID={seg_id}에 픽셀이 없습니다.")
        return result

    # 픽셀 인덱스 클리핑
    u = np.clip(pixels[:, 0].astype(int), 0, data['depth'].shape[1] - 1)
    v = np.clip(pixels[:, 1].astype(int), 0, data['depth'].shape[0] - 1)
    depths = data['depth'][v, u]

    # ── Depth 기술 통계 ──────────────────────────────────────
    result['depth_stats'] = {
        'mean'   : float(depths.mean()),
        'std'    : float(depths.std()),
        'min'    : float(depths.min()),
        'max'    : float(depths.max()),
        'median' : float(np.median(depths)),
    }
    result['pixel_count'] = len(pixels)

    print(f"\n  [{cam_label}] 신호등 seg_id={seg_id}, 샘플={len(pixels)}개")
    print(f"    Depth 통계: mean={result['depth_stats']['mean']:.2f}m, "
          f"std={result['depth_stats']['std']:.2f}m, "
          f"min={result['depth_stats']['min']:.2f}m, "
          f"max={result['depth_stats']['max']:.2f}m")

    E = data['E']

    # ── ① 원본 Extrinsic(E_full)으로 역투영 ─────────────────
    world_full = backproject_to_world(pixels, depths, data['K'], E)
    centroid_full = world_full.mean(axis=0)
    result['world_pts_full'] = world_full
    result['centroid_full']  = centroid_full
    print(f"    [E_full] 3D 중심: X={centroid_full[0]:.3f}, "
          f"Y={centroid_full[1]:.3f}, Z={centroid_full[2]:.3f} m")

    # ── ② Translation 제거 Extrinsic(E_rot_only)으로 역투영 ──
    # 교수님 지시: "거리를 없애주고 싶어 — t 성분만 0으로"
    # E_rot_only: R 성분 유지, t 열을 0으로 설정
    E_rot_only = E.copy()
    E_rot_only[0, 3] = 0.0   # tx = 0
    E_rot_only[1, 3] = 0.0   # ty = 0
    E_rot_only[2, 3] = 0.0   # tz = 0

    world_rot = backproject_to_world(pixels, depths, data['K'], E_rot_only)
    centroid_rot = world_rot.mean(axis=0)
    result['world_pts_rot'] = world_rot
    result['centroid_rot']  = centroid_rot
    print(f"    [E_rot_only] 3D 중심: X={centroid_rot[0]:.3f}, "
          f"Y={centroid_rot[1]:.3f}, Z={centroid_rot[2]:.3f} m")

    # ── 두 중심 사이의 이동 거리 = Translation 성분의 영향 ───
    shift = float(np.linalg.norm(centroid_full - centroid_rot))
    result['centroid_shift'] = shift
    print(f"    Translation 제거로 인한 중심 이동 거리: {shift:.4f}m")
    print(f"    → 이 값이 바로 Extrinsic의 Translation이 Depth·좌표에 미친 영향량입니다.")

    return result


# ============================================================
# [모듈 9] 데이터 종속성 요약 시각화 (종속성 다이어그램)
# ============================================================
def visualize_dependency_diagram(ax):
    """
    Mask, Depth, Camera Extrinsic 세 데이터의 종속 관계를 다이어그램으로 시각화합니다.
    ★ 신호등을 중심 객체로 다이어그램을 구성합니다.
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_facecolor('#f8f9fa')
    ax.set_title(
        "데이터 종속성 다이어그램\n(신호등: Mask ↔ Depth ↔ Camera Extrinsic)",
        fontsize=11, fontweight='bold', pad=10
    )

    nodes = {
        'extrinsic' : (5.0, 5.5, 'Camera\nExtrinsic\n[R | t]', '#3498db'),
        'intrinsic' : (2.0, 5.5, 'Camera\nIntrinsic\n[K]',     '#2ecc71'),
        'depth'     : (3.5, 3.5, 'Depth Map\n(미터 단위)',      '#e74c3c'),
        'mask'      : (6.5, 3.5, 'Panoptic Mask\n(신호등)',     '#9b59b6'),
        '3d_obj'    : (5.0, 1.5, '3D 신호등\n세계 좌표\n(절대적)', '#f39c12'),
    }

    for key, (x, y, text, color) in nodes.items():
        rect = mpatches.FancyBboxPatch(
            (x - 0.9, y - 0.6), 1.8, 1.2,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor='white', linewidth=2, alpha=0.85
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=8, fontweight='bold', color='white')

    edges = [
        ('extrinsic', 'depth',  '위치·방향\n결정'),
        ('intrinsic', 'depth',  '투영\n관계 결정'),
        ('depth',     '3d_obj', 'Back-\nProjection'),
        ('mask',      '3d_obj', '신호등 영역\n지정'),
        ('extrinsic', 'mask',   '카메라 시점\n결정'),
    ]

    for src_key, dst_key, edge_label in edges:
        sx, sy = nodes[src_key][0], nodes[src_key][1]
        dx, dy = nodes[dst_key][0], nodes[dst_key][1]
        vec = np.array([dx - sx, dy - sy])
        unit_vec = vec / np.linalg.norm(vec)
        arrow_start = np.array([sx, sy]) + unit_vec * 0.65
        arrow_end   = np.array([dx, dy]) - unit_vec * 0.65
        ax.annotate('', xy=arrow_end, xytext=arrow_start,
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5))
        mid_x = (arrow_start[0] + arrow_end[0]) / 2
        mid_y = (arrow_start[1] + arrow_end[1]) / 2
        ax.text(mid_x, mid_y, edge_label, ha='center', va='center',
                fontsize=7, color='#333333',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1.5))

    ax.text(
        5.0, 0.4,
        '※ 신호등은 정적 객체 → 카메라 파라미터에 절대 종속\n'
        '   → 파라미터 없이는 Depth도 Mask도 3D 의미 없음\n'
        '   → "신호등은 카메라 포즈에 종속된단 말이야"',
        ha='center', va='center', fontsize=8, style='italic', color='#c0392b',
        bbox=dict(facecolor='#ffeaa7', edgecolor='#e17055', alpha=0.9, pad=4)
    )


# ============================================================
# [모듈 10] 3D Back-projection 결과 시각화
# (★ 단일 카메라: E_full vs E_rot_only 비교 산점도)
# ============================================================
def visualize_backprojection_3d(analysis_result: dict, cam_label: str, ax):
    """
    단일 카메라에서 E_full / E_rot_only 두 가지 역투영 결과를
    하나의 3D 산점도에 겹쳐 표시합니다.

    두 점군의 오프셋이 Translation 성분의 영향을 시각적으로 보여줍니다.

    Args:
        analysis_result : analyze_single_camera_backprojection()의 반환값
        cam_label       : 카메라 이름 레이블
        ax              : matplotlib 3D Axes 객체
    """
    if not analysis_result:
        ax.set_title("신호등 미검출\n(Back-projection 불가)", fontsize=9)
        return

    # E_full 점군 (파란색)
    if 'world_pts_full' in analysis_result:
        pts = analysis_result['world_pts_full']
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c='steelblue', s=8, alpha=0.5, label='E_full (원본)')
        c = analysis_result['centroid_full']
        ax.scatter([c[0]], [c[1]], [c[2]],
                   c='blue', s=150, marker='*', label='중심 (E_full)')

    # E_rot_only 점군 (주황색)
    if 'world_pts_rot' in analysis_result:
        pts = analysis_result['world_pts_rot']
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c='tomato', s=8, alpha=0.5, label='E_rot_only (t=0)')
        c = analysis_result['centroid_rot']
        ax.scatter([c[0]], [c[1]], [c[2]],
                   c='red', s=150, marker='*', label='중심 (E_rot_only)')

    # 두 중심을 잇는 선 (Translation 성분의 영향)
    if 'centroid_full' in analysis_result and 'centroid_rot' in analysis_result:
        c_f = analysis_result['centroid_full']
        c_r = analysis_result['centroid_rot']
        shift = analysis_result.get('centroid_shift', 0.0)
        ax.plot([c_f[0], c_r[0]], [c_f[1], c_r[1]], [c_f[2], c_r[2]],
                'k--', linewidth=1.5, label=f'Translation 영향: {shift:.3f}m')

    ax.set_xlabel('X (m)', fontsize=8)
    ax.set_ylabel('Y (m)', fontsize=8)
    ax.set_zlabel('Z (m)', fontsize=8)
    ax.set_title(
        f'{cam_label} 신호등 Back-projection\nE_full vs E_rot_only', fontsize=9
    )
    ax.legend(fontsize=7, loc='upper left')


# ============================================================
# [메인] 전체 파이프라인 실행 (FRONT 단일 카메라)
# ============================================================
def main():
    """
    메인 실행 함수 (FRONT 카메라 단일 뷰):

    1. Waymo 데이터 로드
    2. FRONT 카메라에서 Mask / Depth / Extrinsic 추출
    3. 신호등 Back-projection 및 종속성 분석
       - E_full vs E_rot_only(Translation 제거) 비교
    4. 시각화 플롯 생성 (2×4 레이아웃)
       행 1: 이미지 / Depth / Mask / Extrinsic 행렬 표시
       행 2: 종속성 다이어그램 / 3D 산점도 / 수치 요약 / Depth 히스토그램
    5. 결과 저장
    """
    # ---- 설정값 ----
    TFRECORD_PATH = (
        './data/individual_files_validation_segment-'
        '12496433400137459534_120_000_140_000_with_camera_labels.tfrecord'
    )
    FRAME_IDX   = 1
    OUTPUT_PATH = './dependency_check_front_only.png'   # ★ 저장 경로 변경

    print("=" * 60)
    print("[1단계] Waymo TFRecord 파일에서 프레임 로드 중...")
    print("=" * 60)

    wfe = WaymoFrameExtractor(TFRECORD_PATH)
    frame_list = wfe.get_frame_list()
    frame = frame_list[FRAME_IDX]
    print(f"총 {len(frame_list)}개 프레임 로드 완료. 프레임 {FRAME_IDX} 사용.")

    print("\n" + "=" * 60)
    print("[2단계] 딥러닝 모델 초기화 (Segmentation + Depth)")
    print("=" * 60)

    ps = PanopticSegmenter()
    de = DepthProEstimator()

    print("\n" + "=" * 60)
    print("[3단계] FRONT 카메라 데이터 추출 (신호등 검출)")
    print("=" * 60)

    data_front = extract_single_camera_data(
        frame,
        dataset_pb2.CameraName.FRONT,
        ps, de
    )

    print("\n" + "=" * 60)
    print("[4단계] 신호등 Back-projection 종속성 분석 (E_full vs E_rot_only)")
    print("=" * 60)

    analysis = analyze_single_camera_backprojection(data_front, cam_label='FRONT')

    print("\n" + "=" * 60)
    print("[5단계] 시각화 플롯 생성 및 저장")
    print("=" * 60)

    # ---- 플롯 레이아웃: 2×4 ─────────────────────────────────
    # 행 1: 이미지 / Depth / Mask / Extrinsic 행렬
    # 행 2: 종속성 다이어그램 / 3D 산점도 / 수치 요약 / Depth 히스토그램
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        "신호등(Traffic Light) - Mask · Depth · Camera Extrinsic 데이터 종속성 검증\n"
        "(FRONT 카메라 단일 뷰 — E_full vs E_rot_only 비교)",
        fontsize=13, fontweight='bold', y=0.99
    )

    # ── 행 1: FRONT 카메라 시각화 ────────────────────────────
    ax_img   = fig.add_subplot(2, 4, 1)   # RGB 이미지
    ax_dep   = fig.add_subplot(2, 4, 2)   # Depth Map
    ax_mask  = fig.add_subplot(2, 4, 3)   # Panoptic Mask
    visualize_single_camera(data_front, [ax_img, ax_dep, ax_mask], "FRONT")

    # ── 행 1 컬럼 4: Extrinsic 행렬 표시 ────────────────────
    ax_ext = fig.add_subplot(2, 4, 4)
    ax_ext.axis('off')
    ax_ext.set_title(
        "FRONT Extrinsic 행렬\n(R 블록 | t 열)", fontsize=9, fontweight='bold'
    )
    E = data_front['E']

    # R 블록과 t 열을 색으로 구분하여 표시
    ext_lines = ["E (4×4) =\n"]
    for r_idx, row in enumerate(E):
        row_str = "  ["
        for c_idx, v in enumerate(row):
            # t 열(c_idx==3)은 별도 강조 표기
            marker = "★" if (c_idx == 3 and r_idx < 3) else " "
            row_str += f"{marker}{v:8.4f}"
        row_str += " ]"
        ext_lines.append(row_str)

    # Translation 수치 별도 요약
    tx, ty, tz = E[0, 3], E[1, 3], E[2, 3]
    ext_lines.append(f"\n★ Translation (t):\n"
                     f"  tx={tx:.4f}, ty={ty:.4f}, tz={tz:.4f}")
    ext_lines.append("\nE_rot_only: 위 t 열을 모두 0으로 설정")

    ax_ext.text(
        0.05, 0.95, "\n".join(ext_lines),
        transform=ax_ext.transAxes,
        fontsize=7, va='top', ha='left', fontfamily='monospace',
        bbox=dict(facecolor='#eaf2ff', edgecolor='#3498db', alpha=0.9, pad=6)
    )

    # ── 행 2 컬럼 1~2: 데이터 종속성 다이어그램 ─────────────
    ax_diag = fig.add_subplot(2, 4, (5, 6))
    visualize_dependency_diagram(ax_diag)

    # ── 행 2 컬럼 3: 3D Back-projection 산점도 ───────────────
    ax_3d = fig.add_subplot(2, 4, 7, projection='3d')
    visualize_backprojection_3d(analysis, "FRONT", ax_3d)

    # ── 행 2 컬럼 4: 수치 요약 + Depth 히스토그램 ───────────
    ax_summary = fig.add_subplot(2, 4, 8)
    ax_summary.axis('off')

    summary_lines = ["[FRONT 신호등 종속성 수치 요약]\n"]
    n_tl = len(data_front['traffic_light_segs'])
    summary_lines.append(f"신호등 검출: {n_tl}개")

    if analysis:
        ds = analysis['depth_stats']
        summary_lines.append(f"\n── Depth 통계 (Mask 내부) ──")
        summary_lines.append(f"  평균:   {ds['mean']:.2f} m")
        summary_lines.append(f"  표준편차: {ds['std']:.2f} m")
        summary_lines.append(f"  최소:   {ds['min']:.2f} m")
        summary_lines.append(f"  최대:   {ds['max']:.2f} m")
        summary_lines.append(f"  중앙값:  {ds['median']:.2f} m")
        summary_lines.append(f"  샘플 픽셀: {analysis['pixel_count']}개")

        if 'centroid_full' in analysis:
            cf = analysis['centroid_full']
            summary_lines.append(f"\n── 3D 중심 (E_full) ──")
            summary_lines.append(f"  X={cf[0]:.3f}, Y={cf[1]:.3f}, Z={cf[2]:.3f} m")

        if 'centroid_rot' in analysis:
            cr = analysis['centroid_rot']
            summary_lines.append(f"\n── 3D 중심 (E_rot_only) ──")
            summary_lines.append(f"  X={cr[0]:.3f}, Y={cr[1]:.3f}, Z={cr[2]:.3f} m")

        if 'centroid_shift' in analysis:
            shift = analysis['centroid_shift']
            summary_lines.append(f"\nTranslation 영향 거리: {shift:.4f} m")
            summary_lines.append("→ 이 오프셋이 t 성분의 기여량")
    else:
        summary_lines.append("\n⚠ 신호등 미검출 — 분석 불가")

    ax_summary.text(
        0.05, 0.97, "\n".join(summary_lines),
        transform=ax_summary.transAxes,
        fontsize=8, va='top', ha='left',
        bbox=dict(facecolor='#e8f8e8', edgecolor='#27ae60', alpha=0.9, pad=6)
    )
    ax_summary.set_title("FRONT 수치 검증 요약", fontsize=9, fontweight='bold')

    # ── 레이아웃 조정 및 저장 ────────────────────────────────
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    print(f"\n결과 이미지 저장 완료: {OUTPUT_PATH}")
    plt.show()


# ============================================================
# 스크립트 직접 실행 시 메인 함수 호출
# ============================================================
if __name__ == "__main__":
    main()