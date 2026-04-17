# ============================================================
# (★ LOOK-AT 카메라 회전 버전 ★)
# ------------------------------------------------------------
#  [목적]
#   촬영된 프레임에서 오직 object mask 영역만 남긴 이미지를
#   생성하고, 해당 객체가 항상 이미지 '정면·중앙'에 오도록
#   카메라를 재설정한 object-centric 데이터셋을 구축한다.
#
#  [★ 이전 버전과의 핵심 차이점 ★]
#   이전: intrinsic(principal point + focal) 만 조정
#          → 이미지 상의 2D crop/resize 에 불과.
#          → 객체가 프레임 가장자리에 있거나 perspective 가
#            기울어져 있으면 "정면 카메라" 로 정렬되지 않음.
#   현재: 카메라 CENTER 는 그대로 두되, 방향(R_c2w) 을
#         object center 를 향하도록 돌리고, focal length 는
#         객체의 실제 크기·거리를 기반으로 계산.
#         카메라 중심이 같으므로 pure rotation homography
#         H = K_new · R_rel · K_orig^(-1)  로
#         왜곡 없이 이미지 재투영이 가능하다 (파노라마 원리).
# ============================================================

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'NanumGothic'

from PIL import Image
import tensorflow as tf
from waymo_open_dataset import dataset_pb2

from utils import (WaymoFrameExtractor, PanopticSegmenter,
                   DepthProEstimator, save_image)

# ──────────────────────────────────────────────────────────
# [상수] 세그멘테이션 라벨 ID 및 그룹화 파라미터
# ──────────────────────────────────────────────────────────
TRAFFIC_LIGHT_LABEL_ID       = 6
POLE_LABEL_ID                = 5
TRAFFIC_SIGN_FRONT_LABEL_ID  = 7
ADJACENCY_DILATE_KERNEL_SIZE = 15

# ──────────────────────────────────────────────────────────
# [상수] Object-centric 이미지 정합 파라미터
# ──────────────────────────────────────────────────────────
ALIGN_OUT_SIZE    = 512          # 출력 정사각 해상도
TARGET_FILL_RATIO = 0.7          # 객체 크기가 출력에서 차지할 비율
BBOX_PADDING      = 4            # (이 버전에서는 3D extent 계산에만 참고)
BACKGROUND_COLOR  = (0, 0, 0)

# world up — Waymo/COLMAP 모두 Z up
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)


# ============================================================
# [0] Waymo ↔ CV 카메라 좌표계 변환 헬퍼
# ============================================================
R_CV_TO_WAYMO_CAM = np.array([
    [ 0,  0,  1],
    [-1,  0,  0],
    [ 0, -1,  0],
], dtype=np.float64)

T_CV_TO_WAYMO_CAM = np.eye(4, dtype=np.float64)
T_CV_TO_WAYMO_CAM[:3, :3] = R_CV_TO_WAYMO_CAM


def waymo_extrinsic_to_cv_extrinsic(E_waymo):
    """Waymo extrinsic(cam→vehicle) 을 OpenCV/COLMAP extrinsic 으로."""
    return E_waymo @ T_CV_TO_WAYMO_CAM


# ============================================================
# [1] 세그멘테이션 결과에서 관심 객체 그룹핑
# ============================================================

def _label_name_to_group(label_name: str):
    name = label_name.lower()
    if 'traffic light' in name: return 'traffic_light'
    if 'pole'          in name: return 'pole'
    if 'traffic sign'  in name: return 'traffic_sign'
    return None


def find_groupable_segments(segments_info, model):
    groupable = []
    for seg in segments_info:
        lid = seg['label_id']
        gt  = None
        if hasattr(model.model.config, 'id2label'):
            gt = _label_name_to_group(
                model.model.config.id2label.get(lid, ''))
        if gt is None:
            if   lid == TRAFFIC_LIGHT_LABEL_ID:      gt = 'traffic_light'
            elif lid == POLE_LABEL_ID:               gt = 'pole'
            elif lid == TRAFFIC_SIGN_FRONT_LABEL_ID: gt = 'traffic_sign'
        if gt:
            s = dict(seg)
            s['group_type'] = gt
            groupable.append(s)
    return groupable


def _masks_are_adjacent(ma, mb, ks=ADJACENCY_DILATE_KERNEL_SIZE):
    k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    da = cv2.dilate(ma.astype(np.uint8), k, iterations=1)
    db = cv2.dilate(mb.astype(np.uint8), k, iterations=1)
    return np.logical_and(da > 0, db > 0).sum() > 0


def group_adjacent_segments(groupable_segs, seg_map):
    n = len(groupable_segs)
    if n == 0: return []
    masks  = [seg_map == s['id'] for s in groupable_segs]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if _masks_are_adjacent(masks[i], masks[j]):
                union(i, j)

    from collections import defaultdict
    cl = defaultdict(list)
    for idx in range(n):
        cl[find(idx)].append(idx)

    H, W = seg_map.shape
    groups = []
    for mi in cl.values():
        segs = [groupable_segs[i] for i in mi]
        gt   = {s['group_type'] for s in segs}
        cm   = np.zeros((H, W), dtype=bool)
        for i in mi: cm |= masks[i]
        groups.append({
            'segments': segs,
            'seg_ids':  [s['id'] for s in segs],
            'group_types':       gt,
            'combined_mask':     cm,
            'has_traffic_light': 'traffic_light' in gt,
        })

    valid = [g for g in groups if g['has_traffic_light']]
    valid.sort(key=lambda g: g['combined_mask'].sum(), reverse=True)
    return valid


# ============================================================
# [2] 카메라 파라미터 / 포즈 조회
# ============================================================

def get_camera_intrinsics(frame, camera_name):
    for cal in frame.context.camera_calibrations:
        if cal.name == camera_name:
            i = cal.intrinsic
            return np.array([[i[0], 0,    i[2]],
                             [0,    i[1], i[3]],
                             [0,    0,    1  ]], dtype=np.float64)
    raise ValueError(f"카메라 {camera_name} intrinsic 없음")


def get_image_size(frame, camera_name):
    for cal in frame.context.camera_calibrations:
        if cal.name == camera_name:
            return cal.width, cal.height
    raise ValueError(f"카메라 {camera_name} size 없음")


def get_camera_extrinsic(frame, camera_name):
    for cal in frame.context.camera_calibrations:
        if cal.name == camera_name:
            return np.array(cal.extrinsic.transform).reshape(4, 4)
    raise ValueError(f"카메라 {camera_name} extrinsic 없음")


def get_vehicle_pose(frame):
    return np.array(frame.pose.transform).reshape(4, 4)


# ============================================================
# [3] 마스크 내부 픽셀 역투영
# ============================================================

def unproject_masked_pixels(depth_map, object_mask, K,
                             depth_min=0.1, depth_max=80.0,
                             sample_stride=1):
    H, W = depth_map.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    vs_g = np.arange(0, H, sample_stride)
    us_g = np.arange(0, W, sample_stride)
    uu, vv = np.meshgrid(us_g, vs_g)
    uu = uu.ravel(); vv = vv.ravel()

    mask_vals  = object_mask[vv, uu]
    depth_vals = depth_map[vv, uu]
    valid = mask_vals & (depth_vals >= depth_min) & (depth_vals <= depth_max)

    uu = uu[valid].astype(np.float64)
    vv = vv[valid].astype(np.float64)
    dd = depth_vals[valid].astype(np.float64)

    if len(dd) == 0:
        return np.zeros((0, 3)), np.zeros((0, 2))

    X = (uu - cx) / fx * dd
    Y = (vv - cy) / fy * dd
    Z = dd

    pts_cam = np.stack([X, Y, Z], axis=1)
    uvs     = np.stack([uu, vv], axis=1)
    return pts_cam, uvs


def camera_points_to_global(pts_cam, E_cv_cam2vehicle, ego_pose):
    if len(pts_cam) == 0:
        return np.zeros((0, 3))
    N = pts_cam.shape[0]
    pts_cam_h = np.concatenate([pts_cam, np.ones((N, 1))], axis=1)
    T_cam2global = ego_pose @ E_cv_cam2vehicle
    pts_g = (T_cam2global @ pts_cam_h.T).T
    return pts_g[:, :3]


# ============================================================
# [4] Object-centric 좌표계 정규화 헬퍼
# ============================================================

def compute_object_center(all_points_global):
    """Median 기반 객체 중심 (outlier 에 강건)."""
    if len(all_points_global) == 0:
        return np.zeros(3)
    return np.median(all_points_global, axis=0)


def compute_object_extent_3d(all_points_global, pct_lo=5, pct_hi=95):
    """
    ★ NEW ★
    3D 포인트 클라우드의 물리적 크기(월드 단위) 를 robust 하게 추정.
    (각 축에서 pct_lo~pct_hi 범위의 길이 중 최대값)

    이 값이 focal length 결정의 기준이 되어,
    모든 프레임에서 '같은 zoom 비율' 로 객체를 보이게 한다.
    """
    if len(all_points_global) == 0:
        return 1.0
    lo = np.percentile(all_points_global, pct_lo, axis=0)
    hi = np.percentile(all_points_global, pct_hi, axis=0)
    extent = hi - lo
    L = float(np.max(extent))
    return max(L, 1e-3)


def make_object_centric_transform(object_center):
    """global → object frame (평행이동만)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = -object_center
    return T


# ============================================================
# [5] ★ 핵심: Look-at 카메라 구성 (extrinsic 회전 + intrinsic) ★
# ============================================================

def look_at_rotation(cam_pos, target_pos, world_up=WORLD_UP):
    """
    카메라 위치 → target 을 바라보는 R_c2w (OpenCV convention).

    [규약]
      OpenCV 카메라 좌표계: X=right, Y=down, Z=forward
      world 의 up 은 Z+ (Waymo/COLMAP).
    [유도]
      z_c = normalize(target - cam_pos)          (forward)
      x_c = normalize(z_c × world_up)            (image right)
      y_c = z_c × x_c                            (image down)

    R_c2w = [x_c | y_c | z_c]  (column-stacked, 3×3 orthonormal)
    """
    fwd = target_pos - cam_pos
    n = np.linalg.norm(fwd)
    if n < 1e-9:
        return np.eye(3)
    z_c = fwd / n

    x_c = np.cross(z_c, world_up)
    if np.linalg.norm(x_c) < 1e-6:
        # forward 와 world_up 이 평행한 극단 케이스
        x_c = np.cross(z_c, np.array([0.0, 1.0, 0.0]))
    x_c = x_c / np.linalg.norm(x_c)

    y_c = np.cross(z_c, x_c)      # 이미 단위벡터 (z_c ⟂ x_c)

    R_c2w = np.column_stack([x_c, y_c, z_c])
    return R_c2w


def compute_look_at_camera(K_orig, T_c2g_orig, object_mask,
                            object_center, out_size,
                            fill_ratio=TARGET_FILL_RATIO,
                            world_up=WORLD_UP,
                            max_samples=4000,
                            bbox_pct=(0.0, 100.0)):
    """
    ★ 핵심 함수 (mask 기반 정확한 꽉 채우기) ★
    원본 카메라의 CENTER 는 유지하고, 방향만 object_center 를
    향하도록 회전. focal length 와 principal point 는 **mask 픽셀을
    새 카메라의 normalized projection 으로 변환한 실제 2D bbox** 가
    이미지에 fill_ratio 비율로 꽉 차도록 계산.

    [왜 3D extent 가 아니라 2D mask 를 쓰는가?]
      · 3D extent 는 depth 예측 노이즈가 섞여 정확도가 떨어진다.
      · mask 는 panoptic segmentation 결과라 픽셀 단위로 정확하다.
      · Pure rotation 만 적용하므로 픽셀의 "방향" 만 알면 되고
        depth 가 전혀 필요 없다 (단일 homography 로 정의됨).

    [zoom / crop 대응]
      zoom → f_new  (focal length)
      crop → cx, cy (principal point 이동 = 2D translation)
      Mask bbox 중심이 이미지 정중앙에 오고, bbox 긴 변이
      out_size · fill_ratio 픽셀로 맞춰진다.

    Returns:
      K_new:      (3,3) 새 intrinsic
      T_c2g_new:  (4,4) 새 camera→global (R 만 변경, t 유지)
      H_warp:     (3,3) 원본 픽셀 → 새 픽셀 호모그래피
                  H = K_new · R_rel · K_orig^(-1),
                  where R_rel = R_w2c_new · R_c2w_orig
      diag:       딕셔너리 (dist, f_new, bbox_norm, extent)
    """
    cam_pos = T_c2g_orig[:3, 3].copy()

    # 1) Look-at 회전 — object_center 를 바라보게
    R_c2w_new = look_at_rotation(cam_pos, object_center, world_up)

    T_c2g_new = np.eye(4, dtype=np.float64)
    T_c2g_new[:3, :3] = R_c2w_new
    T_c2g_new[:3, 3]  = cam_pos

    # 상대 회전 (원본 cam → 새 cam)
    R_c2w_orig = T_c2g_orig[:3, :3]
    R_w2c_new  = R_c2w_new.T
    R_rel      = R_w2c_new @ R_c2w_orig

    # 2) Mask 픽셀을 normalized projection (새 카메라 좌표계) 으로 변환
    #    · 원본 픽셀 → ray 방향: r_o = K_orig^-1 · [u, v, 1]ᵀ
    #    · 회전:             r_n = R_rel · r_o
    #    · normalized:       (xₙ, yₙ) = (r_n[0]/r_n[2], r_n[1]/r_n[2])
    ys, xs = np.where(object_mask)
    N = len(xs)

    dist = float(np.linalg.norm(object_center - cam_pos))

    if N == 0:
        # Fallback: 객체 중심 방향만 사용
        f_new = out_size * fill_ratio * max(dist, 1e-6) / 2.0
        cx_n, cy_n = out_size / 2.0, out_size / 2.0
        bbox_norm = (0.0, 0.0, 0.0, 0.0)
        extent = 0.0
    else:
        # 대량 픽셀은 서브샘플링 (속도)
        if N > max_samples:
            idx = np.linspace(0, N - 1, max_samples, dtype=int)
            xs = xs[idx]; ys = ys[idx]

        M = len(xs)
        ones = np.ones(M, dtype=np.float64)
        uv1 = np.stack([xs.astype(np.float64),
                        ys.astype(np.float64), ones], axis=0)   # (3, M)

        K_inv = np.linalg.inv(K_orig)
        rays_orig = K_inv @ uv1                                  # (3, M)
        rays_new  = R_rel @ rays_orig                            # (3, M)

        Zn = rays_new[2]
        valid = Zn > 1e-4
        Xn = rays_new[0][valid] / Zn[valid]
        Yn = rays_new[1][valid] / Zn[valid]

        if len(Xn) < 3:
            f_new = out_size * fill_ratio * max(dist, 1e-6) / 2.0
            cx_n, cy_n = out_size / 2.0, out_size / 2.0
            bbox_norm = (0.0, 0.0, 0.0, 0.0)
            extent = 0.0
        else:
            # Robust bbox (percentile 선택 가능 — 기본 0/100 이면 정확 bbox)
            lo, hi = bbox_pct
            x_min = float(np.percentile(Xn, lo))
            x_max = float(np.percentile(Xn, hi))
            y_min = float(np.percentile(Yn, lo))
            y_max = float(np.percentile(Yn, hi))

            x_c = 0.5 * (x_min + x_max)
            y_c = 0.5 * (y_min + y_max)
            dx  = 0.5 * (x_max - x_min)
            dy  = 0.5 * (y_max - y_min)
            half_ext = max(dx, dy, 1e-6)
            extent = float(half_ext * 2.0)

            # ── zoom 결정 (focal length) ──
            #   f_new · 2 · half_ext = out_size · fill_ratio
            f_new = (out_size * fill_ratio) / (2.0 * half_ext)

            # ── crop 결정 (principal point) ──
            #   bbox 중심 (x_c, y_c) 가 이미지 중심 (out_size/2, out_size/2) 로
            #   u_new = f_new · xₙ + cx_n  ⇒  cx_n = out_size/2 − f_new · x_c
            cx_n = out_size / 2.0 - f_new * x_c
            cy_n = out_size / 2.0 - f_new * y_c

            bbox_norm = (x_min, y_min, x_max, y_max)

    # 3) 새 intrinsic
    K_new = np.array([[f_new, 0.0,   cx_n],
                      [0.0,   f_new, cy_n],
                      [0.0,   0.0,   1.0 ]], dtype=np.float64)

    # 4) Pure rotation homography
    H_warp = K_new @ R_rel @ np.linalg.inv(K_orig)

    diag = {
        'dist':      dist,
        'f_new':     float(f_new),
        'bbox_norm': bbox_norm,      # (x_min, y_min, x_max, y_max) normalized
        'extent':    extent,         # normalized bbox 긴 변
        'cx':        float(cx_n),
        'cy':        float(cy_n),
    }
    return K_new, T_c2g_new, H_warp, diag


# ============================================================
# [6] 이미지/마스크/Depth 와핑
# ============================================================

def warp_image_and_mask(image, mask, H_warp, out_size,
                        bg_color=BACKGROUND_COLOR):
    """원본 RGB·마스크를 homography 로 와핑 + mask-only 합성."""
    dsize = (out_size, out_size)

    img_warped = cv2.warpPerspective(
        image, H_warp, dsize, flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)

    mask_u8 = (mask.astype(np.uint8) * 255)
    mask_warped_u8 = cv2.warpPerspective(
        mask_u8, H_warp, dsize, flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask_warped = mask_warped_u8 > 127

    img_masked_out = img_warped.copy()
    img_masked_out[~mask_warped] = np.array(bg_color, dtype=np.uint8)

    return img_masked_out, img_warped, mask_warped


def warp_depth_pure_rotation(depth_orig, K_orig, K_new, R_rel, out_size):
    """
    ★ NEW ★
    Pure rotation around camera center 에서 depth map 을 정확히 warp.
    - 순수 회전이어도 depth 값(Z 성분) 은 좌표축이 돌아가므로 달라진다.
    - 각 새 픽셀 → inverse homography 로 원본 픽셀 조회 → 원본 depth
      샘플링 → pure rotation 관계로 새 Z 계산.

    이 방식은 inverse warp 이므로 결과에 hole 이 생기지 않는다.
    """
    H_o, W_o = depth_orig.shape

    # 새 이미지의 픽셀 그리드
    vs_n, us_n = np.meshgrid(np.arange(out_size),
                             np.arange(out_size),
                             indexing='ij')
    ones = np.ones_like(us_n)
    uv1_n = np.stack([us_n.ravel(), vs_n.ravel(), ones.ravel()],
                     axis=0).astype(np.float64)

    # 새 → 원본 픽셀 (inverse homography)
    #   H_inv = K_orig · R_rel^T · K_new^{-1}
    H_inv = K_orig @ R_rel.T @ np.linalg.inv(K_new)
    uv1_o = H_inv @ uv1_n
    uv1_o = uv1_o / uv1_o[2:3]
    u_o = uv1_o[0].reshape(out_size, out_size)
    v_o = uv1_o[1].reshape(out_size, out_size)

    # 원본 depth 를 nearest 로 샘플링
    ui = np.clip(np.round(u_o).astype(int), 0, W_o - 1)
    vi = np.clip(np.round(v_o).astype(int), 0, H_o - 1)
    in_bounds = (u_o >= 0) & (u_o < W_o) & (v_o >= 0) & (v_o < H_o)

    depth_o = depth_orig[vi, ui].astype(np.float32)
    depth_o = np.where(in_bounds, depth_o, 0.0)

    # 원본 카메라 좌표 (X_o, Y_o, Z_o)
    fx_o, fy_o = K_orig[0, 0], K_orig[1, 1]
    cx_o, cy_o = K_orig[0, 2], K_orig[1, 2]
    X_o = (u_o - cx_o) / fx_o * depth_o
    Y_o = (v_o - cy_o) / fy_o * depth_o
    Z_o = depth_o

    # 새 카메라 좌표의 Z 성분 = R_rel의 3번째 행 · (X_o, Y_o, Z_o)
    Z_n = (R_rel[2, 0] * X_o +
           R_rel[2, 1] * Y_o +
           R_rel[2, 2] * Z_o)
    Z_n = np.where((Z_n > 0.01) & (depth_o > 0), Z_n, 0.0).astype(np.float32)
    return Z_n


def transform_uvs_with_homography(uvs, H_warp):
    """원본 픽셀 (N,2) → 새 픽셀 (N,2)."""
    if len(uvs) == 0:
        return uvs.copy()
    N = len(uvs)
    pts_h = np.concatenate([uvs, np.ones((N, 1))], axis=1)
    pts_n = (H_warp @ pts_h.T).T
    pts_n = pts_n[:, :2] / pts_n[:, 2:3]
    return pts_n


# ============================================================
# [7] COLMAP I/O
# ============================================================

def rotation_matrix_to_quaternion(R):
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        qw = 0.25 / s
        qx = (R[2,1] - R[1,2]) * s
        qy = (R[0,2] - R[2,0]) * s
        qz = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        qw = (R[2,1] - R[1,2]) / s
        qx = 0.25 * s
        qy = (R[0,1] + R[1,0]) / s
        qz = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        qw = (R[0,2] - R[2,0]) / s
        qx = (R[0,1] + R[1,0]) / s
        qy = 0.25 * s
        qz = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        qw = (R[1,0] - R[0,1]) / s
        qx = (R[0,2] + R[2,0]) / s
        qy = (R[1,2] + R[2,1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz])
    q /= np.linalg.norm(q)
    if q[0] < 0: q = -q
    return q


def camera_to_obj_to_colmap_pose(T_c2obj):
    T_w2c = np.linalg.inv(T_c2obj)
    R_w2c = T_w2c[:3, :3]
    t_w2c = T_w2c[:3, 3]
    qvec  = rotation_matrix_to_quaternion(R_w2c)
    return qvec, t_w2c


def write_colmap_images_txt(path, image_entries, points2d_per_image):
    with open(path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, "
                "CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        n = len(image_entries)
        total = sum(len(points2d_per_image.get(e['image_id'], []))
                    for e in image_entries)
        mean_obs = total / n if n > 0 else 0
        f.write(f"# Number of images: {n}, "
                f"mean observations per image: {mean_obs:.1f}\n")
        for e in image_entries:
            qw, qx, qy, qz = e['qvec']
            tx, ty, tz     = e['tvec']
            f.write(f"{e['image_id']} "
                    f"{qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f} "
                    f"{tx:.10f} {ty:.10f} {tz:.10f} "
                    f"{e['camera_id']} {e['name']}\n")
            pts2d = points2d_per_image.get(e['image_id'], [])
            if len(pts2d) > 0:
                parts = [f"{x:.2f} {y:.2f} {p3d_id}"
                         for (x, y, p3d_id) in pts2d]
                f.write(' '.join(parts) + '\n')
            else:
                f.write('\n')
    print(f"  [COLMAP] images.txt 저장: {path} ({n}개 이미지)")


def write_colmap_points3d_txt(path, points3d_list):
    with open(path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, "
                "TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        total_track = sum(len(p['track']) for p in points3d_list)
        mean_track  = (total_track / len(points3d_list)
                       if points3d_list else 0)
        f.write(f"# Number of points: {len(points3d_list)}, "
                f"mean track length: {mean_track:.4f}\n")
        for p in points3d_list:
            track_str = ' '.join(
                f"{img_id} {pt2d_idx}" for img_id, pt2d_idx in p['track'])
            f.write(f"{p['id']} "
                    f"{p['x']:.6f} {p['y']:.6f} {p['z']:.6f} "
                    f"{p['r']} {p['g']} {p['b']} "
                    f"{p['error']:.4f} {track_str}\n")
    print(f"  [COLMAP] points3D.txt 저장: {path} ({len(points3d_list)}개)")


def write_depth_map_bin(path, depth_map):
    H, W = depth_map.shape
    header = f"{W}&{H}&1&"
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        depth_map.astype(np.float32).tofile(f)


# ============================================================
# [8] Cross-view Track 생성
# ============================================================

def build_tracks_object_centric(frame_data_list, depth_consistency_thresh=1.5):
    n_frames = len(frame_data_list)
    if n_frames == 0:
        return [], {}

    T_w2c_list = [np.linalg.inv(fd['T_c2obj']) for fd in frame_data_list]
    K_list     = [fd['K_new']        for fd in frame_data_list]
    mask_list  = [fd['mask_warped']  for fd in frame_data_list]
    dep_list   = [fd['depth_warped'] for fd in frame_data_list]
    H, W = mask_list[0].shape

    points3d_list      = []
    points2d_per_image = {fd['image_id']: [] for fd in frame_data_list}
    pt2d_counter       = {fd['image_id']: 0  for fd in frame_data_list}
    point3d_id = 1

    total_candidates = sum(len(fd['src_pts_obj']) for fd in frame_data_list)
    print(f"  [Tracks] 총 후보 3D 포인트: {total_candidates}개")

    for src_fi, fd in enumerate(frame_data_list):
        src_pts    = fd['src_pts_obj']
        src_uvs    = fd['src_uvs_new']
        src_rgb    = fd['src_rgb']
        src_img_id = fd['image_id']

        for pi in range(len(src_pts)):
            xyz = src_pts[pi]
            u_src, v_src = src_uvs[pi]

            if not (0 <= u_src < W and 0 <= v_src < H):
                continue

            r, g, b = src_rgb[pi]

            track = []
            src_pt2d_idx = pt2d_counter[src_img_id]
            pt2d_counter[src_img_id] += 1
            points2d_per_image[src_img_id].append(
                (float(u_src), float(v_src), point3d_id))
            track.append((src_img_id, src_pt2d_idx))

            xyz_h = np.array([xyz[0], xyz[1], xyz[2], 1.0])
            for fi2, fd2 in enumerate(frame_data_list):
                if fi2 == src_fi: continue
                pt_cam = T_w2c_list[fi2] @ xyz_h
                X2, Y2, Z2 = pt_cam[0], pt_cam[1], pt_cam[2]
                if Z2 < 0.1: continue

                K2 = K_list[fi2]
                u2 = K2[0, 0] * X2 / Z2 + K2[0, 2]
                v2 = K2[1, 1] * Y2 / Z2 + K2[1, 2]
                u2i, v2i = int(round(u2)), int(round(v2))
                if u2i < 0 or u2i >= W or v2i < 0 or v2i >= H: continue
                if not mask_list[fi2][v2i, u2i]: continue

                d2 = dep_list[fi2][v2i, u2i]
                if d2 < 0.1: continue
                if abs(Z2 - d2) > depth_consistency_thresh: continue

                img_id2    = fd2['image_id']
                pt2d_idx2  = pt2d_counter[img_id2]
                pt2d_counter[img_id2] += 1
                points2d_per_image[img_id2].append(
                    (float(u2), float(v2), point3d_id))
                track.append((img_id2, pt2d_idx2))

            points3d_list.append({
                'id':    point3d_id,
                'x':     float(xyz[0]),
                'y':     float(xyz[1]),
                'z':     float(xyz[2]),
                'r':     int(r),
                'g':     int(g),
                'b':     int(b),
                'error': 0.0,
                'track': track,
            })
            point3d_id += 1

    track_lens = [len(p['track']) for p in points3d_list]
    multi_view = sum(1 for t in track_lens if t > 1)
    print(f"  [Tracks] 생성된 3D 포인트: {len(points3d_list)}개")
    print(f"           다중 뷰 관측 (track≥2): {multi_view}개 "
          f"({multi_view/max(len(points3d_list),1)*100:.1f}%)")
    if track_lens:
        print(f"           평균 track 길이: {np.mean(track_lens):.2f}")
        print(f"           최대 track 길이: {max(track_lens)}")
    return points3d_list, points2d_per_image


# ============================================================
# [9] 메인 파이프라인
# ============================================================

def build_dataset(frame_list, camera_name,
                  ps_model, de_model,
                  output_dir='./aligned_dataset',
                  start_frame=0,
                  max_frames=10, frame_stride=6,
                  sample_stride=2,
                  depth_min=0.5, depth_max=80.0,
                  out_size=ALIGN_OUT_SIZE,
                  fill_ratio=TARGET_FILL_RATIO):

    dir_images     = os.path.join(output_dir, 'images')
    dir_images_raw = os.path.join(output_dir, 'images_warped')
    dir_masks      = os.path.join(output_dir, 'masks')
    dir_sparse     = os.path.join(output_dir, 'sparse', '0')
    dir_depth_maps = os.path.join(output_dir, 'stereo', 'depth_maps')
    for d in [dir_images, dir_images_raw, dir_masks,
              dir_sparse, dir_depth_maps]:
        os.makedirs(d, exist_ok=True)
    print(f"출력 디렉터리: {output_dir}")

    sampled = frame_list[start_frame::frame_stride][:max_frames]
    print(f"  start_frame={start_frame}, frame_stride={frame_stride} "
          f"→ {len(sampled)}개 프레임")

    # ──────────────────────────────────────────────────────
    # Pass 1: 원본 뷰에서 3D 포인트 수집
    # ──────────────────────────────────────────────────────
    per_frame      = []
    all_pts_global = []

    print(f"\n[Pass 1] 원본 뷰에서 객체 3D 포인트 수집")
    for i, frame in enumerate(sampled):
        orig_idx = i * frame_stride + start_frame
        print(f"--- 프레임 {i+1}/{len(sampled)} (원본 #{orig_idx}) ---")

        image = None
        for img_obj in frame.images:
            if img_obj.name == camera_name:
                image = tf.io.decode_jpeg(img_obj.image).numpy()
                break
        if image is None:
            print(f"  이미지 없음, 스킵"); continue

        K        = get_camera_intrinsics(frame, camera_name)
        E_waymo  = get_camera_extrinsic(frame, camera_name)
        E_cv     = waymo_extrinsic_to_cv_extrinsic(E_waymo)
        ego_pose = get_vehicle_pose(frame)

        seg_result = ps_model.segment(image)
        seg_map    = seg_result['segmentation'].cpu().numpy()
        groups = group_adjacent_segments(
            find_groupable_segments(seg_result['segments_info'], ps_model),
            seg_map)
        if not groups:
            print(f"  ⚠ 신호등 그룹 미검출, 스킵"); continue
        best_group  = groups[0]
        object_mask = best_group['combined_mask']
        print(f"  ✓ 그룹: [{', '.join(sorted(best_group['group_types']))}]"
              f" | 마스크 픽셀 {object_mask.sum()}")

        depth_map = de_model.get_depth_map(image)

        pts_cam, uvs = unproject_masked_pixels(
            depth_map, object_mask, K,
            depth_min=depth_min, depth_max=depth_max,
            sample_stride=sample_stride)
        if len(pts_cam) == 0:
            print(f"  ⚠ 유효 depth 없음, 스킵"); continue

        uu_i = uvs[:, 0].astype(int)
        vv_i = uvs[:, 1].astype(int)
        rgb  = image[vv_i, uu_i, :]

        pts_global = camera_points_to_global(pts_cam, E_cv, ego_pose)
        all_pts_global.append(pts_global)

        T_c2g = ego_pose @ E_cv

        per_frame.append({
            'image_id':    i + 1,
            'orig_idx':    orig_idx,
            'image':       image,
            'mask':        object_mask,
            'depth':       depth_map,
            'K':           K,
            'T_c2g':       T_c2g,
            'pts_cam':     pts_cam,
            'pts_global':  pts_global,
            'uvs':         uvs,
            'rgb':         rgb,
            'group_types': sorted(best_group['group_types']),
        })

    if len(per_frame) == 0:
        print("\n[ERROR] 유효 프레임 없음.")
        return []

    all_pts_concat = np.concatenate(all_pts_global, axis=0)
    object_center  = compute_object_center(all_pts_concat)
    L_world        = compute_object_extent_3d(all_pts_concat)   # 참고값
    T_g2obj        = make_object_centric_transform(object_center)

    print(f"\n{'='*60}")
    print(f"[Object center 확정]")
    print(f"  전체 3D 포인트: {len(all_pts_concat)}개")
    print(f"  ★ 객체 중심 (median, global): "
          f"[{object_center[0]:.2f}, "
          f"{object_center[1]:.2f}, "
          f"{object_center[2]:.2f}]")
    print(f"  ℹ  객체 3D extent (참고): {L_world:.2f} m "
          f"(focal 계산에는 mask 2D bbox 사용)")
    print(f"{'='*60}")

    # ──────────────────────────────────────────────────────
    # Pass 2: ★ Look-at 카메라 재설정 + 이미지 와핑 ★
    # ──────────────────────────────────────────────────────
    print(f"\n[Pass 2] Look-at 카메라 정합 (mask-based 꽉 채우기)")
    print(f"         출력 크기: {out_size}x{out_size} | "
          f"fill_ratio: {fill_ratio}")
    print(f"         · zoom(f) · crop(cx,cy) 모두 mask bbox 로 자동 조정")

    colmap_image_entries  = []
    frame_data_for_tracks = []

    for fd in per_frame:
        K_orig    = fd['K']
        mask_orig = fd['mask']
        image     = fd['image']
        depth     = fd['depth']
        T_c2g_o   = fd['T_c2g']

        # (a) ★ Look-at 카메라 구성 (mask 기반 꽉 채우기) ★
        #     - zoom  = f_new
        #     - crop  = cx_n, cy_n (bbox center 를 이미지 중앙으로 이동)
        K_new, T_c2g_new, H_warp, diag = compute_look_at_camera(
            K_orig, T_c2g_o, mask_orig, object_center,
            out_size, fill_ratio=fill_ratio, world_up=WORLD_UP)
        dist = diag['dist']

        # 이 프레임에 해당하는 R_rel (depth warp 에 사용)
        R_rel = T_c2g_new[:3, :3].T @ T_c2g_o[:3, :3]

        # (b) 이미지 + 마스크 와핑 → mask-only 출력 생성
        img_masked, img_warped, mask_warped = warp_image_and_mask(
            image, mask_orig, H_warp, out_size)

        # (c) depth warp (pure rotation 정확식)
        depth_warped = warp_depth_pure_rotation(
            depth, K_orig, K_new, R_rel, out_size)
        masked_depth_warped = np.where(mask_warped, depth_warped, 0.0)

        # (d) 원본 UV → 새 이미지 UV (호모그래피)
        uvs_new = transform_uvs_with_homography(fd['uvs'], H_warp)

        # (e) 3D 포인트 → object frame
        if len(fd['pts_global']) > 0:
            pts_g_h = np.concatenate(
                [fd['pts_global'],
                 np.ones((len(fd['pts_global']), 1))], axis=1)
            pts_obj = (T_g2obj @ pts_g_h.T).T[:, :3]
        else:
            pts_obj = np.zeros((0, 3))

        # (f) 새 카메라 포즈: cam → global → object frame
        T_c2obj = T_g2obj @ T_c2g_new

        qvec, tvec     = camera_to_obj_to_colmap_pose(T_c2obj)
        image_filename = f"frame_{fd['orig_idx']:04d}.jpg"
        mask_filename  = image_filename.replace('.jpg', '.png')
        this_cam_id    = fd['image_id']

        colmap_image_entries.append({
            'image_id':  fd['image_id'],
            'qvec':      qvec,
            'tvec':      tvec,
            'camera_id': this_cam_id,
            'name':      image_filename,
            'K_new':     K_new,
            'w':         out_size,
            'h':         out_size,
        })

        frame_data_for_tracks.append({
            'image_id':     fd['image_id'],
            'T_c2obj':      T_c2obj,
            'K_new':        K_new,
            'mask_warped':  mask_warped,
            'depth_warped': masked_depth_warped,
            'src_pts_obj':  pts_obj,
            'src_uvs_new':  uvs_new,
            'src_rgb':      fd['rgb'],
        })

        # 디버그 출력: 카메라가 object 를 얼마나 잘 바라보는지
        cam_pos = T_c2g_o[:3, 3]
        fwd_new = T_c2g_new[:3, 2]  # z_c in world = forward
        to_obj  = (object_center - cam_pos)
        to_obj /= (np.linalg.norm(to_obj) + 1e-9)
        cos_err = float(np.dot(fwd_new, to_obj))

        bbox_norm = diag['bbox_norm']
        print(f"  F{fd['orig_idx']:04d}: "
              f"dist={dist:.2f}m | f={K_new[0,0]:.1f}px | "
              f"cx,cy=({K_new[0,2]:.1f},{K_new[1,2]:.1f}) | "
              f"look-at cos={cos_err:.4f} | "
              f"mask-only 픽셀 {mask_warped.sum()}")

        # ── 파일 저장 ──
        save_image(img_masked,
                   os.path.join(dir_images, image_filename))
        save_image(img_warped,
                   os.path.join(dir_images_raw, image_filename))
        mask_u8 = (mask_warped.astype(np.uint8) * 255)
        save_image(mask_u8,
                   os.path.join(dir_masks, mask_filename))
        write_depth_map_bin(
            os.path.join(dir_depth_maps,
                         f"{image_filename}.geometric.bin"),
            masked_depth_warped)

    # ── cameras.txt (프레임별 K_new 기록; PINHOLE 모델) ──
    cam_txt_path = os.path.join(dir_sparse, 'cameras.txt')
    with open(cam_txt_path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(colmap_image_entries)}\n")
        for e in colmap_image_entries:
            K = e['K_new']
            params = [K[0,0], K[1,1], K[0,2], K[1,2]]
            ps = ' '.join(f"{p:.10f}" for p in params)
            f.write(f"{e['camera_id']} PINHOLE "
                    f"{e['w']} {e['h']} {ps}\n")
    print(f"  [COLMAP] cameras.txt 저장: {cam_txt_path} "
          f"({len(colmap_image_entries)}개 카메라)")

    # ──────────────────────────────────────────────────────
    # Pass 3: Cross-view track + COLMAP images/points3D
    # ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Pass 3] Cross-view track 생성 (object frame)")
    print(f"{'='*60}")
    points3d_list, points2d_per_image = build_tracks_object_centric(
        frame_data_for_tracks, depth_consistency_thresh=1.5)

    if colmap_image_entries:
        write_colmap_images_txt(
            os.path.join(dir_sparse, 'images.txt'),
            colmap_image_entries, points2d_per_image)
        write_colmap_points3d_txt(
            os.path.join(dir_sparse, 'points3D.txt'),
            points3d_list)

    print(f"\n{'='*60}")
    print(f"[완료] Look-at 기반 Object-aligned Mask-only 데이터셋")
    print(f"{'='*60}")
    print(f"  프레임       : {len(per_frame)}개")
    print(f"  3D 포인트    : {len(points3d_list)}개")
    print(f"  이미지 크기  : {out_size}x{out_size}")
    print(f"  fill_ratio   : {fill_ratio} (mask bbox → image bbox)")
    print(f"  원점(월드)   : Object center (median, global)")
    print(f"  카메라 정렬  : 모든 프레임이 object_center 를 정면 응시")
    print(f"  zoom/crop    : mask 2D bbox 기반 per-frame intrinsic 자동조정")

    results = []
    for fd, entry in zip(per_frame, colmap_image_entries):
        results.append({
            'frame_idx':   fd['orig_idx'],
            'image_id':    fd['image_id'],
            'image_path':  os.path.join(dir_images, entry['name']),
            'warped_path': os.path.join(dir_images_raw, entry['name']),
            'mask_path':   os.path.join(dir_masks,
                                         entry['name'].replace('.jpg', '.png')),
            'qvec':        entry['qvec'].tolist(),
            'tvec':        entry['tvec'].tolist(),
            'group_types': fd['group_types'],
        })
    return results


# ============================================================
# [10] 시각화
# ============================================================

def visualize_results(results, save_path='./aligned_comparison.png'):
    n = len(results)
    if n == 0:
        print("시각화할 프레임 없음"); return

    fig, axes = plt.subplots(n, 3, figsize=(13, 4.2*n), squeeze=False)
    fig.suptitle(
        "Look-at Object-aligned Mask-only Dataset\n"
        "(좌: 전체 와핑 RGB | 가운데: 정합 마스크 | 우: mask-only 출력)",
        fontsize=12, fontweight='bold', y=0.995)

    for row, res in enumerate(results):
        img_warp = np.array(Image.open(res['warped_path']))
        mask     = np.array(Image.open(res['mask_path']))
        img_only = np.array(Image.open(res['image_path']))

        tx, ty, tz = res['tvec']

        axes[row, 0].imshow(img_warp)
        axes[row, 0].set_title(
            f"F{res['frame_idx']} | 전체 와핑", fontsize=9)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(mask, cmap='gray')
        axes[row, 1].set_title(
            f"F{res['frame_idx']} | 정합 마스크", fontsize=9)
        axes[row, 1].axis('off')

        axes[row, 2].imshow(img_only)
        axes[row, 2].set_title(
            f"F{res['frame_idx']} | mask-only\n"
            f"{'+'.join(res['group_types'])} | "
            f"t=({tx:.1f},{ty:.1f},{tz:.1f})",
            fontsize=9)
        axes[row, 2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f"시각화 저장: {save_path}")


# ============================================================
# [메인]
# ============================================================

def main():
    TFRECORD_PATH = (
        './data/validation/segment-6001094526418694294_4609_470_4629_470_with_camera_labels.tfrecord'
    )
    CAMERA        = dataset_pb2.CameraName.FRONT
    OUTPUT_DIR    = './multiview_dataset'
    COMPARE_PATH  = './multiview_comparison.png'
    START_FRAME   = 135
    MAX_FRAMES    = 21
    FRAME_STRIDE  = 3
    SAMPLE_STRIDE = 2
    DEPTH_MIN     = 0.5
    DEPTH_MAX     = 80.0
    OUT_SIZE      = 512
    FILL_RATIO    = 0.9   # 0.9 = 객체가 이미지에 거의 꽉 차게
                          # 1.0 에 가까울수록 꽉 차고 여백이 줄어듦

    print("=" * 60)
    print("[1] Waymo TFRecord 로드")
    print("=" * 60)
    wfe        = WaymoFrameExtractor(TFRECORD_PATH)
    frame_list = wfe.get_frame_list()
    print(f"  로드: {len(frame_list)}개 프레임")

    print("\n" + "=" * 60)
    print("[2] 모델 초기화 (Panoptic + Depth)")
    print("=" * 60)
    ps = PanopticSegmenter()
    de = DepthProEstimator()

    print("\n" + "=" * 60)
    print("[3] Look-at Object-aligned Mask-only 데이터셋 생성")
    print("    - 카메라 CENTER 유지 + 방향만 객체를 향해 회전")
    print("    - focal length 는 객체 물리 크기/거리로 결정")
    print("    - 모든 프레임이 객체를 '정면·중앙' 으로 응시")
    print("=" * 60)
    results = build_dataset(
        frame_list, CAMERA, ps, de,
        output_dir=OUTPUT_DIR,
        start_frame=START_FRAME,
        max_frames=MAX_FRAMES,
        frame_stride=FRAME_STRIDE,
        sample_stride=SAMPLE_STRIDE,
        depth_min=DEPTH_MIN,
        depth_max=DEPTH_MAX,
        out_size=OUT_SIZE,
        fill_ratio=FILL_RATIO)

    print("\n" + "=" * 60)
    print("[4] 시각화")
    print("=" * 60)
    visualize_results(results, save_path=COMPARE_PATH)

    print("\n" + "=" * 60)
    print("[완료]")
    print("=" * 60)
    print(f"  출력: {OUTPUT_DIR}/")
    print(f"    · images/            (★ mask-only RGB — 최종 출력)")
    print(f"    · images_warped/     (전체 와핑 RGB — 참고용)")
    print(f"    · masks/             (정합된 이진 마스크)")
    print(f"    · sparse/0/          (COLMAP, object-centric 좌표계)")
    print(f"    · stereo/depth_maps/ (정합된 masked depth)")


if __name__ == "__main__":
    main()