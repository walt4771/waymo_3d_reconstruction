# ============================================================
# 3_render_object_aligned.py
# ------------------------------------------------------------
#  [목적]
#   촬영된 여러 프레임에서 "같은 객체(신호등 등)"가 항상 이미지의
#   정면·중앙에 오도록 카메라 intrinsic 을 재계산(정합)하여
#   object-centric 데이터셋을 구축한다.
#
#  [이번 버전의 변경점]
#   (1) ★ 프레임당 신호등 후보를 최대 N개(기본 3)까지 보관하고,
#       consensus 에 가장 가까운 후보를 프레임마다 선택한다.
#       → 가장 큰 그룹이 다른 신호등일 때도 2/3순위가 맞으면 살릴 수 있음.
#   (2) ★ CUDA OOM 완화: 모델 호출 직후 torch.cuda.empty_cache()+gc.collect()
#       호출, 추론 출력은 즉시 NumPy 로 옮기고 GPU 텐서 del.
#   (3) 저장 이미지는 mask-only 가 아닌 **온전한 와핑 이미지**.
#   (4) 모든 함수/핵심 라인에 한글 주석 상세.
# ============================================================

import os                                         # 디렉터리 생성/파일 경로 처리
import gc                                         # ★ (new) 수동 GC
import numpy as np                                # 수치 연산
import cv2                                        # 이미지 와핑/모폴로지 연산
import torch                                      # ★ (new) cuda cache flush 용
import matplotlib.pyplot as plt                   # 결과 시각화
plt.rcParams["font.family"] = 'NanumGothic'       # 한글 폰트 지정 (Linux 환경)

from PIL import Image                             # 이미지 읽기(시각화 용도)
import tensorflow as tf                           # Waymo TFRecord 디코딩
from waymo_open_dataset import dataset_pb2        # Waymo 데이터 스키마

# 내부 유틸 모듈 (프로젝트에 포함되어 있다고 가정)
from utils import (WaymoFrameExtractor, PanopticSegmenter,
                   DepthProEstimator, save_image)

# ──────────────────────────────────────────────────────────
# [상수] 세그멘테이션 라벨 ID 및 그룹화 파라미터
# ──────────────────────────────────────────────────────────
TRAFFIC_LIGHT_LABEL_ID       = 6    # 신호등 라벨 ID (fallback 용)
POLE_LABEL_ID                = 5    # 기둥 라벨 ID
TRAFFIC_SIGN_FRONT_LABEL_ID  = 7    # 표지판(정면) 라벨 ID
ADJACENCY_DILATE_KERNEL_SIZE = 15   # 인접 판정을 위해 마스크를 부풀릴 커널 크기

# ──────────────────────────────────────────────────────────
# [상수] Object-centric 정합(align) 파라미터
# ──────────────────────────────────────────────────────────
ALIGN_OUT_SIZE    = 512              # 정합 후 출력 이미지 한 변 픽셀 수 (정사각형)
TARGET_FILL_RATIO = 0.7              # 긴 변이 전체의 몇 %가 되도록 확대할지
BBOX_PADDING      = 4                # bbox 주변 여백(픽셀)
BACKGROUND_COLOR  = (0, 0, 0)        # warpPerspective 의 borderValue (밖 영역)

# ──────────────────────────────────────────────────────────
# [상수] 프레임 간 객체 일관성 검사 파라미터
# ──────────────────────────────────────────────────────────
# global 좌표계에서 "한 객체"로 인정할 최대 편차(m).
OBJECT_CENTER_CONSISTENCY_M = 3.0
# 일관성 검사에서 consensus 를 재추정하는 반복 횟수
CONSISTENCY_ITERATIONS = 5
# ★ (new) 프레임당 고려할 신호등 후보 최대 수
MAX_CANDIDATES_PER_FRAME = 3


# ============================================================
# [0] Waymo ↔ OpenCV 카메라 좌표계 변환
# ============================================================
R_CV_TO_WAYMO_CAM = np.array([
    [ 0,  0,  1],
    [-1,  0,  0],
    [ 0, -1,  0],
], dtype=np.float64)

T_CV_TO_WAYMO_CAM = np.eye(4, dtype=np.float64)
T_CV_TO_WAYMO_CAM[:3, :3] = R_CV_TO_WAYMO_CAM


def waymo_extrinsic_to_cv_extrinsic(E_waymo):
    """Waymo extrinsic → OpenCV/COLMAP extrinsic."""
    return E_waymo @ T_CV_TO_WAYMO_CAM


# ============================================================
# [★ new] CUDA 메모리 정리 헬퍼
# ============================================================
def _cuda_flush():
    """
    모델 호출 사이에 GPU 캐시/파이썬 객체를 정리해 OOM 을 완화.
    - gc.collect(): 아직 참조가 남아있는 Python 객체 정리
    - torch.cuda.empty_cache(): PyTorch allocator 의 free block 을 드라이버에 반환
    (inference 경로에서는 safe — 학습 state 가 없으므로 속도 저하도 미미)
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # IPC 단편화도 함께 해소 (torch>=1.11)
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


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
            s = dict(seg); s['group_type'] = gt
            groupable.append(s)
    return groupable


def _masks_are_adjacent(ma, mb, ks=ADJACENCY_DILATE_KERNEL_SIZE):
    k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    da = cv2.dilate(ma.astype(np.uint8), k, iterations=1)
    db = cv2.dilate(mb.astype(np.uint8), k, iterations=1)
    return np.logical_and(da > 0, db > 0).sum() > 0


def group_adjacent_segments(groupable_segs, seg_map):
    """
    인접 segment Union-Find 로 그룹핑.
    신호등 포함 그룹만 반환. 면적 내림차순.
    """
    n = len(groupable_segs)
    if n == 0:
        return []
    masks  = [seg_map == s['id'] for s in groupable_segs]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

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
        for i in mi:
            cm |= masks[i]
        groups.append({
            'segments':          segs,
            'seg_ids':           [s['id'] for s in segs],
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
# [3] 마스크 내부 픽셀 역투영 (depth → 3D)
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
    if len(all_points_global) == 0:
        return np.zeros(3)
    return np.median(all_points_global, axis=0)


def make_object_centric_transform(object_center):
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = -object_center
    return T


def compute_per_frame_center(pts_global):
    if len(pts_global) == 0:
        return None
    return np.median(pts_global, axis=0)


# ============================================================
# [4.5] ★ (수정) 다중 후보 기반 프레임 일관성 검사 ★
# ============================================================

def filter_frames_by_object_consistency_multi(
        per_frame,
        max_distance_m=OBJECT_CENTER_CONSISTENCY_M,
        iterations=CONSISTENCY_ITERATIONS):
    """
    각 프레임이 여러 후보(candidates) 를 가질 때, consensus 에 가장
    가까운 후보를 프레임마다 선택한다. 어떤 후보도 threshold 내에
    들지 않는 프레임만 폐기한다.

    [알고리즘]
      1) 모든 프레임의 모든 후보 center 의 median → 초기 consensus
      2) 반복:
         a) 각 프레임에서 consensus 와의 거리 최소인 후보 선택
            (단, 거리 <= max_distance_m 만 유효; 아니면 프레임 거부)
         b) 선택된 후보들의 center median → consensus 갱신
         c) 이전 iteration 과 동일하면 조기 종료

    Returns:
      filtered_per_frame  : 선택된 후보 정보가 프레임 최상위 레벨에
                             주입(flatten)된 dict 리스트
      consensus_center    : 최종 consensus (global)
      rejected_info_list  : 폐기된 프레임 디버그 정보
    """
    if len(per_frame) == 0:
        return [], np.zeros(3), []

    # ── (1) 초기 consensus: 모든 후보 center 의 median ──
    all_c = []
    for fd in per_frame:
        for cand in fd['candidates']:
            if cand.get('center_global') is not None:
                all_c.append(cand['center_global'])
    if len(all_c) == 0:
        # 어떤 후보도 유효한 3D center 를 못 뽑았으면 전부 폐기
        rejected = [{
            'orig_idx': fd['orig_idx'], 'image_id': fd['image_id'],
            'n_candidates': len(fd['candidates']),
            'min_dist': float('inf'),
            'reason': '유효 후보 없음 (모든 후보의 center 계산 불가)',
        } for fd in per_frame]
        return [], np.zeros(3), rejected

    consensus = np.median(np.array(all_c), axis=0)

    # 현재 각 프레임의 선택된 후보 인덱스 (-1 = 거부)
    selected = [-1] * len(per_frame)

    # ── (2) 반복 (선택 ↔ consensus 재추정) ──
    for it in range(iterations):
        new_selected = []
        chosen_centers = []
        for fi, fd in enumerate(per_frame):
            best_ci   = -1
            best_dist = np.inf
            for ci, cand in enumerate(fd['candidates']):
                c = cand.get('center_global')
                if c is None:
                    continue
                d = float(np.linalg.norm(c - consensus))
                # threshold 내에서만 후보로 인정, 그 중 최소 거리
                if d <= max_distance_m and d < best_dist:
                    best_dist = d
                    best_ci   = ci
            new_selected.append(best_ci)
            if best_ci >= 0:
                chosen_centers.append(
                    fd['candidates'][best_ci]['center_global'])

        # consensus 갱신
        if len(chosen_centers) == 0:
            break
        new_consensus = np.median(np.array(chosen_centers), axis=0)

        # 수렴 판정: 선택이 바뀌지 않으면 종료
        if new_selected == selected \
           and np.linalg.norm(new_consensus - consensus) < 1e-6:
            selected = new_selected
            consensus = new_consensus
            break
        selected = new_selected
        consensus = new_consensus

    # ── (3) 최종 결과 구성 ──
    filtered      = []
    rejected_info = []
    for fi, fd in enumerate(per_frame):
        if selected[fi] >= 0:
            # 선택된 후보 정보를 프레임 최상위에 주입(flatten)
            chosen = fd['candidates'][selected[fi]]
            new_fd = {
                'image_id':   fd['image_id'],
                'orig_idx':   fd['orig_idx'],
                'image':      fd['image'],
                'depth':      fd['depth'],
                'K':          fd['K'],
                'T_c2g':      fd['T_c2g'],
                # ↓ 선택된 후보 내용 (mask/pts/uvs/rgb/group_types)
                'mask':        chosen['mask'],
                'pts_cam':     chosen['pts_cam'],
                'pts_global':  chosen['pts_global'],
                'uvs':         chosen['uvs'],
                'rgb':         chosen['rgb'],
                'group_types': chosen['group_types'],
                'selected_candidate_idx': selected[fi],
                'n_candidates':           len(fd['candidates']),
            }
            filtered.append(new_fd)
        else:
            # 모든 후보가 threshold 를 초과 → 폐기
            dists = []
            for cand in fd['candidates']:
                c = cand.get('center_global')
                if c is not None:
                    dists.append(float(np.linalg.norm(c - consensus)))
            min_dist = min(dists) if dists else float('inf')
            rejected_info.append({
                'orig_idx':     fd['orig_idx'],
                'image_id':     fd['image_id'],
                'n_candidates': len(fd['candidates']),
                'min_dist':     min_dist,
                'reason': (f"{len(fd['candidates'])}개 후보 모두 "
                           f"consensus 로부터 최소 {min_dist:.2f}m "
                           f"(> {max_distance_m}m)"),
            })

    return filtered, consensus, rejected_info


# ============================================================
# [5] 핵심: 객체 정합용 intrinsic 재계산 + 이미지 와핑
# ============================================================
def compute_aligned_intrinsics(K_orig, mask, out_size,
                               fill_ratio=TARGET_FILL_RATIO,
                               padding=BBOX_PADDING):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return K_orig.copy(), np.eye(3), None, 1.0

    x0 = max(int(xs.min()) - padding, 0)
    y0 = max(int(ys.min()) - padding, 0)
    x1 = int(xs.max()) + padding
    y1 = int(ys.max()) + padding
    bw = x1 - x0
    bh = y1 - y0
    u_c = (x0 + x1) / 2.0
    v_c = (y0 + y1) / 2.0
    L   = max(bw, bh)

    scale = (out_size * fill_ratio) / max(L, 1.0)

    fx_o, fy_o = K_orig[0, 0], K_orig[1, 1]
    cx_o, cy_o = K_orig[0, 2], K_orig[1, 2]

    fx_n = scale * fx_o
    fy_n = scale * fy_o
    cx_n = out_size / 2.0 - scale * (u_c - cx_o)
    cy_n = out_size / 2.0 - scale * (v_c - cy_o)

    K_new = np.array([[fx_n, 0,    cx_n],
                      [0,    fy_n, cy_n],
                      [0,    0,    1  ]], dtype=np.float64)

    H_warp = K_new @ np.linalg.inv(K_orig)
    return K_new, H_warp, (x0, y0, x1, y1), scale


def warp_image_and_mask(image, mask, depth, H_warp, out_size,
                        bg_color=BACKGROUND_COLOR):
    dsize = (out_size, out_size)
    img_warped = cv2.warpPerspective(
        image, H_warp, dsize, flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)

    mask_u8 = (mask.astype(np.uint8) * 255)
    mask_warped_u8 = cv2.warpPerspective(
        mask_u8, H_warp, dsize, flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask_warped = mask_warped_u8 > 127

    depth_warped = cv2.warpPerspective(
        depth.astype(np.float32), H_warp, dsize,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    return img_warped, mask_warped, depth_warped


def transform_uvs_with_homography(uvs, H_warp):
    if len(uvs) == 0:
        return uvs.copy()
    N = len(uvs)
    pts_h = np.concatenate([uvs, np.ones((N, 1))], axis=1)
    pts_n = (H_warp @ pts_h.T).T
    pts_n = pts_n[:, :2] / pts_n[:, 2:3]
    return pts_n


# ============================================================
# [6] COLMAP I/O
# ============================================================
def rotation_matrix_to_quaternion(R):
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        s  = 0.5 / np.sqrt(tr + 1.0)
        qw = 0.25 / s
        qx = (R[2,1] - R[1,2]) * s
        qy = (R[0,2] - R[2,0]) * s
        qz = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s  = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        qw = (R[2,1] - R[1,2]) / s
        qx = 0.25 * s
        qy = (R[0,1] + R[1,0]) / s
        qz = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s  = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        qw = (R[0,2] - R[2,0]) / s
        qx = (R[0,1] + R[1,0]) / s
        qy = 0.25 * s
        qz = (R[1,2] + R[2,1]) / s
    else:
        s  = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        qw = (R[1,0] - R[0,1]) / s
        qx = (R[0,2] + R[2,0]) / s
        qy = (R[1,2] + R[2,1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz])
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q


def camera_to_obj_to_colmap_pose(T_c2obj):
    T_w2c = np.linalg.inv(T_c2obj)
    R_w2c = T_w2c[:3, :3]
    t_w2c = T_w2c[:3, 3]
    qvec  = rotation_matrix_to_quaternion(R_w2c)
    return qvec, t_w2c


def write_colmap_cameras_txt(path, camera_id, model,
                              width, height, params):
    with open(path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        ps = ' '.join(f"{p:.10f}" for p in params)
        f.write(f"{camera_id} {model} {width} {height} {ps}\n")
    print(f"  [COLMAP] cameras.txt 저장: {path}")


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
# [7] Cross-view Track 생성
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
                if fi2 == src_fi:
                    continue
                pt_cam = T_w2c_list[fi2] @ xyz_h
                X2, Y2, Z2 = pt_cam[0], pt_cam[1], pt_cam[2]
                if Z2 < 0.1:
                    continue

                K2 = K_list[fi2]
                u2 = K2[0, 0] * X2 / Z2 + K2[0, 2]
                v2 = K2[1, 1] * Y2 / Z2 + K2[1, 2]
                u2i, v2i = int(round(u2)), int(round(v2))

                if u2i < 0 or u2i >= W or v2i < 0 or v2i >= H:
                    continue
                if not mask_list[fi2][v2i, u2i]:
                    continue
                d2 = dep_list[fi2][v2i, u2i]
                if d2 < 0.1:
                    continue
                if abs(Z2 - d2) > depth_consistency_thresh:
                    continue

                img_id2   = fd2['image_id']
                pt2d_idx2 = pt2d_counter[img_id2]
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
# [★ new] 단일 후보 그룹 → 3D 포인트/center 계산 헬퍼
# ============================================================
def _build_candidate_from_group(group, depth_map, image, K,
                                E_cv, ego_pose,
                                depth_min, depth_max, sample_stride):
    """
    한 '그룹(마스크)' 에 대해 역투영 → 카메라 3D → global 3D → center
    를 계산해 후보 dict 로 반환. 유효 포인트가 0개면 None.
    """
    object_mask = group['combined_mask']

    pts_cam, uvs = unproject_masked_pixels(
        depth_map, object_mask, K,
        depth_min=depth_min, depth_max=depth_max,
        sample_stride=sample_stride)
    if len(pts_cam) == 0:
        return None

    uu_i = uvs[:, 0].astype(int)
    vv_i = uvs[:, 1].astype(int)
    rgb  = image[vv_i, uu_i, :]

    pts_global = camera_points_to_global(pts_cam, E_cv, ego_pose)
    center     = compute_per_frame_center(pts_global)
    if center is None:
        return None

    return {
        'mask':          object_mask,
        'pts_cam':       pts_cam,
        'pts_global':    pts_global,
        'uvs':           uvs,
        'rgb':           rgb,
        'center_global': center,
        'group_types':   sorted(group['group_types']),
        'mask_pixels':   int(object_mask.sum()),
    }


# ============================================================
# [8] 메인 파이프라인
# ============================================================
def build_dataset(frame_list, camera_name,
                  ps_model, de_model,
                  output_dir='./aligned_dataset',
                  start_frame=0,
                  max_frames=10, frame_stride=6,
                  sample_stride=2,
                  depth_min=0.5, depth_max=80.0,
                  out_size=ALIGN_OUT_SIZE,
                  fill_ratio=TARGET_FILL_RATIO,
                  consistency_m=OBJECT_CENTER_CONSISTENCY_M,
                  max_candidates=MAX_CANDIDATES_PER_FRAME):
    """
    ★ 이번 버전:
      - 프레임당 신호등 후보 최대 max_candidates 개 수집
      - consensus 에 가장 가까운 후보를 프레임마다 선택
      - 모든 후보가 threshold 초과인 프레임만 폐기
      - Pass 1 루프 내 CUDA 메모리 정리
    """
    dir_images = os.path.join(output_dir, 'images')
    dir_masks  = os.path.join(output_dir, 'masks')
    dir_sparse = os.path.join(output_dir, 'sparse', '0')
    dir_depth_maps = os.path.join(output_dir, 'stereo', 'depth_maps')
    for d in [dir_images, dir_masks, dir_sparse, dir_depth_maps]:
        os.makedirs(d, exist_ok=True)
    print(f"출력 디렉터리: {output_dir}")

    sampled = frame_list[start_frame::frame_stride][:max_frames]
    print(f"  start_frame={start_frame}, frame_stride={frame_stride} "
          f"→ 샘플 {len(sampled)}개")

    # ──────────────────────────────────────────────────────
    # Pass 1: 각 프레임에서 후보 최대 N개 수집 (★ 변경)
    # ──────────────────────────────────────────────────────
    per_frame = []
    print(f"\n[Pass 1] 프레임당 최대 {max_candidates}개 후보 수집")
    for i, frame in enumerate(sampled):
        orig_idx = start_frame + i * frame_stride
        print(f"--- 프레임 {i+1}/{len(sampled)} (원본 #{orig_idx}) ---")

        # (a) 이미지 디코딩
        image = None
        for img_obj in frame.images:
            if img_obj.name == camera_name:
                image = tf.io.decode_jpeg(img_obj.image).numpy()
                break
        if image is None:
            print(f"  이미지 없음, 스킵"); continue

        # (b) 카메라 파라미터
        K        = get_camera_intrinsics(frame, camera_name)
        E_waymo  = get_camera_extrinsic(frame, camera_name)
        E_cv     = waymo_extrinsic_to_cv_extrinsic(E_waymo)
        ego_pose = get_vehicle_pose(frame)

        # (c) Panoptic segmentation — ★ OOM 방지: inference_mode + flush
        try:
            with torch.inference_mode():
                seg_result = ps_model.segment(image)
        except torch.cuda.OutOfMemoryError:
            print(f"  ⚠ seg OOM → cache flush 후 재시도")
            _cuda_flush()
            with torch.inference_mode():
                seg_result = ps_model.segment(image)

        seg_tensor = seg_result['segmentation']
        # GPU 텐서 → numpy 즉시 복사 (참조 끊기)
        seg_map = (seg_tensor.detach().cpu().numpy()
                   if hasattr(seg_tensor, 'detach') else np.asarray(seg_tensor))
        del seg_tensor
        segs_info = seg_result['segments_info']
        del seg_result
        _cuda_flush()  # ★ (new)

        groups = group_adjacent_segments(
            find_groupable_segments(segs_info, ps_model), seg_map)
        if not groups:
            print(f"  ⚠ 신호등 그룹 미검출, 스킵")
            del image, seg_map; _cuda_flush(); continue

        # (d) Depth 추정 — ★ OOM 방지
        try:
            with torch.inference_mode():
                depth_map = de_model.get_depth_map(image)
        except torch.cuda.OutOfMemoryError:
            print(f"  ⚠ depth OOM → cache flush 후 재시도")
            _cuda_flush()
            with torch.inference_mode():
                depth_map = de_model.get_depth_map(image)

        # depth 도 즉시 numpy 로
        if hasattr(depth_map, 'detach'):
            depth_map = depth_map.detach().cpu().numpy()
        depth_map = np.asarray(depth_map, dtype=np.float32)
        _cuda_flush()  # ★ (new)

        # (e) 후보 최대 N개 처리
        candidates = []
        for g in groups[:max_candidates]:
            cand = _build_candidate_from_group(
                g, depth_map, image, K, E_cv, ego_pose,
                depth_min, depth_max, sample_stride)
            if cand is not None:
                candidates.append(cand)

        if len(candidates) == 0:
            print(f"  ⚠ 유효 후보 없음 (모두 depth 부족), 스킵")
            del image, depth_map, seg_map; _cuda_flush(); continue

        # 각 후보 요약 로그
        c_log = ', '.join(
            f"[{ci}] {'+'.join(c['group_types'])} "
            f"(px={c['mask_pixels']}, "
            f"center=[{c['center_global'][0]:.1f},"
            f"{c['center_global'][1]:.1f},"
            f"{c['center_global'][2]:.1f}])"
            for ci, c in enumerate(candidates))
        print(f"  ✓ 후보 {len(candidates)}/{min(len(groups), max_candidates)}개: {c_log}")

        T_c2g = ego_pose @ E_cv
        per_frame.append({
            'image_id':   i + 1,
            'orig_idx':   orig_idx,
            'image':      image,
            'depth':      depth_map,
            'K':          K,
            'T_c2g':      T_c2g,
            'candidates': candidates,
        })

        # 세그 맵은 이 이후 불필요
        del seg_map, segs_info, groups
        _cuda_flush()

    if len(per_frame) == 0:
        print("\n[ERROR] 유효 프레임 없음."); return []

    # ──────────────────────────────────────────────────────
    # Pass 1.5: ★ 다중 후보 기반 일관성 검사 (★ 변경)
    # ──────────────────────────────────────────────────────
    print(f"\n[Pass 1.5] 프레임 간 객체 일관성 검사 "
          f"(threshold={consistency_m}m, 후보당 최대 {max_candidates}개)")
    filtered_frames, consensus_center, rejected = \
        filter_frames_by_object_consistency_multi(
            per_frame,
            max_distance_m=consistency_m,
            iterations=CONSISTENCY_ITERATIONS)

    # 후보 선택 요약 로그
    picked_counts = {0: 0, 1: 0, 2: 0}
    for fd in filtered_frames:
        k = fd.get('selected_candidate_idx', 0)
        picked_counts[k] = picked_counts.get(k, 0) + 1
    if filtered_frames:
        print(f"  ✓ 후보 선택 분포: "
              + ', '.join(f"[{k}]={v}개" for k, v in sorted(picked_counts.items())))

    if rejected:
        print(f"  ⚠ 폐기된 프레임 {len(rejected)}개:")
        for r in rejected:
            print(f"    · F{r['orig_idx']:04d} "
                  f"(image_id={r['image_id']}, "
                  f"후보 {r['n_candidates']}개): {r['reason']}")
    else:
        print(f"  ✓ 모든 프레임이 동일 객체로 판정됨")

    if len(filtered_frames) == 0:
        print("\n[ERROR] 일관성 검사 후 남은 프레임 없음. "
              "threshold/후보수 를 늘리거나 입력을 점검하세요.")
        return []

    object_center = consensus_center
    T_g2obj       = make_object_centric_transform(object_center)

    print(f"\n{'='*60}")
    print(f"[Object center 확정]")
    print(f"  유지 프레임 : {len(filtered_frames)}개 / 전체 {len(per_frame)}개")
    print(f"  ★ 객체 중심 (global consensus, median): "
          f"[{object_center[0]:.2f}, "
          f"{object_center[1]:.2f}, "
          f"{object_center[2]:.2f}]")
    print(f"{'='*60}")

    # ──────────────────────────────────────────────────────
    # Pass 2: 정합 (intrinsic 재계산 + 온전한 이미지 와핑)
    # ──────────────────────────────────────────────────────
    print(f"\n[Pass 2] 객체 중앙 정합 (intrinsic 재계산 + 이미지 와핑)")
    print(f"         출력 크기: {out_size}x{out_size} | "
          f"fill_ratio: {fill_ratio}")

    colmap_image_entries  = []
    frame_data_for_tracks = []

    for fd in filtered_frames:
        K_orig = fd['K']
        mask   = fd['mask']
        image  = fd['image']
        depth  = fd['depth']
        T_c2g  = fd['T_c2g']

        K_new, H_warp, bbox, scale = compute_aligned_intrinsics(
            K_orig, mask, out_size, fill_ratio=fill_ratio)

        img_warped, mask_warped, depth_warped = \
            warp_image_and_mask(image, mask, depth, H_warp, out_size)

        masked_depth_warped = np.where(mask_warped, depth_warped, 0.0)
        uvs_new = transform_uvs_with_homography(fd['uvs'], H_warp)

        if len(fd['pts_global']) > 0:
            pts_g_h = np.concatenate(
                [fd['pts_global'],
                 np.ones((len(fd['pts_global']), 1))], axis=1)
            pts_obj = (T_g2obj @ pts_g_h.T).T[:, :3]
        else:
            pts_obj = np.zeros((0, 3))

        T_c2obj = T_g2obj @ T_c2g
        qvec, tvec     = camera_to_obj_to_colmap_pose(T_c2obj)
        image_filename = f"frame_{fd['orig_idx']:04d}.jpg"
        mask_filename  = image_filename.replace('.jpg', '.png')

        this_cam_id = fd['image_id']

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

        sel = fd.get('selected_candidate_idx', 0)
        print(f"  F{fd['orig_idx']:04d} [cand {sel}/{fd['n_candidates']-1}]: "
              f"bbox={bbox} | scale={scale:.2f}x | "
              f"K_new=(fx={K_new[0,0]:.1f}, cx={K_new[0,2]:.1f}, "
              f"cy={K_new[1,2]:.1f}) | 마스크 픽셀(참고)={mask_warped.sum()}")

        # 파일 저장
        save_image(img_warped,
                   os.path.join(dir_images, image_filename))
        mask_u8 = (mask_warped.astype(np.uint8) * 255)
        save_image(mask_u8,
                   os.path.join(dir_masks, mask_filename))
        write_depth_map_bin(
            os.path.join(dir_depth_maps,
                         f"{image_filename}.geometric.bin"),
            masked_depth_warped)

    # cameras.txt 출력
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
    print(f"[완료] Object-aligned 데이터셋 (온전한 이미지)")
    print(f"{'='*60}")
    print(f"  입력 프레임  : {len(per_frame)}개")
    print(f"  폐기 프레임  : {len(rejected)}개 (객체 불일치)")
    print(f"  최종 프레임  : {len(filtered_frames)}개")
    print(f"  3D 포인트    : {len(points3d_list)}개")
    print(f"  이미지 크기  : {out_size}x{out_size}")
    print(f"  fill_ratio   : {fill_ratio}")
    print(f"  원점 (world) : Object consensus center (global median)")

    results = []
    for fd, entry in zip(filtered_frames, colmap_image_entries):
        results.append({
            'frame_idx':   fd['orig_idx'],
            'image_id':    fd['image_id'],
            'image_path':  os.path.join(dir_images, entry['name']),
            'mask_path':   os.path.join(dir_masks,
                                         entry['name'].replace('.jpg', '.png')),
            'qvec':        entry['qvec'].tolist(),
            'tvec':        entry['tvec'].tolist(),
            'group_types': fd['group_types'],
            'selected_candidate_idx': fd.get('selected_candidate_idx', 0),
        })
    return results


# ============================================================
# [9] 시각화
# ============================================================
def visualize_results(results, save_path='./aligned_comparison.png'):
    n = len(results)
    if n == 0:
        print("시각화할 프레임 없음"); return

    fig, axes = plt.subplots(n, 2, figsize=(9, 4.2*n), squeeze=False)
    fig.suptitle(
        "Object-aligned Dataset\n"
        "(좌: 온전한 와핑 RGB | 우: 정합 마스크 — 참고용)",
        fontsize=12, fontweight='bold', y=0.995)

    for row, res in enumerate(results):
        img  = np.array(Image.open(res['image_path']))
        mask = np.array(Image.open(res['mask_path']))
        tx, ty, tz = res['tvec']
        sel = res.get('selected_candidate_idx', 0)

        axes[row, 0].imshow(img)
        axes[row, 0].set_title(
            f"F{res['frame_idx']} | cand {sel} | 온전한 와핑 RGB\n"
            f"{'+'.join(res['group_types'])} | "
            f"t=({tx:.1f},{ty:.1f},{tz:.1f})",
            fontsize=9)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(mask, cmap='gray')
        axes[row, 1].set_title(
            f"F{res['frame_idx']} | 정합 마스크", fontsize=9)
        axes[row, 1].axis('off')

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
    FILL_RATIO    = 0.7
    CONSISTENCY_M = OBJECT_CENTER_CONSISTENCY_M
    MAX_CAND      = MAX_CANDIDATES_PER_FRAME       # ★ (new) 후보 3개

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
    print("[3] Object-aligned 데이터셋 생성")
    print("    - 프레임당 신호등 후보 최대 3개 수집")
    print("    - consensus 에 가장 가까운 후보를 프레임마다 선택")
    print("    - Pass 1 루프 내 CUDA 캐시 flush 로 OOM 완화")
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
        fill_ratio=FILL_RATIO,
        consistency_m=CONSISTENCY_M,
        max_candidates=MAX_CAND)

    print("\n" + "=" * 60)
    print("[4] 시각화")
    print("=" * 60)
    visualize_results(results, save_path=COMPARE_PATH)

    print("\n" + "=" * 60)
    print("[완료]")
    print("=" * 60)
    print(f"  출력: {OUTPUT_DIR}/")
    print(f"    · images/            (★ 온전한 와핑 RGB — 최종 출력)")
    print(f"    · masks/             (정합된 이진 마스크 — 보조용)")
    print(f"    · sparse/0/          (COLMAP, object-centric 좌표계)")
    print(f"    · stereo/depth_maps/ (정합된 masked depth)")


if __name__ == "__main__":
    main()