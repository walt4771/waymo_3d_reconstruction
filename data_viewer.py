# ================================================================================
# Waymo 추출 데이터 통합 시각화 스크립트
# ================================================================================
# [입력] ./output/ (extract_waymo_dataset.py 결과)
#   ├── image/      *.jpg
#   ├── depth/      *.npy, *.jpg
#   ├── panoptic/   *.npy, *.jpg, *_info.txt
#   ├── extrinsic/  *.txt
#   └── intrinsic/  *.txt
#
# [출력] ./visualization/
#   ├── per_frame/         프레임별 통합 시각화 (이미지/뎁스/판옵틱)
#   ├── intrinsic/         프레임별 Intrinsic 기하학적 시각화 (주점, FoV)
#   ├── summary_grid.png   전체 프레임 요약 격자
#   └── extrinsic_detailed/  (Camera Extrinsic 상세 분석)
#       ├── 1_multiview_3d.png
#       ├── 2_topdown_2d.png
#       ├── 3_6dof_timeline.png
#       ├── 4_frustums_with_thumbnails.png
#       └── 5_vehicle_vs_camera.png
# ================================================================================

# 운영체제와 상호작용하기 위한 모듈 (파일 경로, 폴더 생성 등)
import os
# 정규 표현식 모듈 (텍스트에서 특정 패턴의 문자열을 추출할 때 사용)
import re
# 수치 해석 및 다차원 배열 연산을 위한 핵심 라이브러리
import numpy as np
# 파이썬 이미지 처리 라이브러리 (이미지 로드 및 크기 조절)
from PIL import Image
# 그래프 및 시각화를 위한 라이브러리
import matplotlib
# 화면 출력이 없는 환경(서버 등)에서도 파일로 이미지를 저장할 수 있도록 백엔드를 'Agg'로 설정
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# 2D 도형(직사각형, 화살표 등)을 그리기 위한 모듈
from matplotlib.patches import Rectangle, FancyArrowPatch
# 범례나 선을 직접 그리기 위한 모듈
from matplotlib.lines import Line2D
# 3D 그래프를 그리기 위한 툴킷
from mpl_toolkits.mplot3d import Axes3D
# 3D 공간에 다각형(Frustum 등)을 그리기 위한 모듈
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ================================================================================
# [Loaders] 텍스트 파일 파싱
# ================================================================================

def load_extrinsic_txt(path: str) -> dict:
    """
    텍스트 파일에서 카메라의 외부 파라미터(Extrinsic)를 읽어옵니다.
    출력 형식: {'cam2vehicle': 4x4 배열, 'vehicle2world': 4x4 배열, 'cam2world': 4x4 배열}
    """
    # 텍스트 파일을 읽기 모드(utf-8 인코딩)로 엽니다.
    with open(path, 'r', encoding='utf-8') as f:
        # 파일의 모든 줄을 읽어 리스트로 저장합니다.
        lines = f.readlines()

    # 파싱된 변환 행렬을 저장할 딕셔너리 초기화
    matrices = {'cam2vehicle': None, 'vehicle2world': None, 'cam2world': None}
    # 현재 읽고 있는 행렬의 종류를 추적하는 변수
    current_key = None
    # 숫자 데이터를 임시로 모아둘 리스트
    buffer = []

    # 텍스트 파일 내의 헤더 문자열과 딕셔너리 키를 매핑
    section_map = {
        '[1] Extrinsic: Camera → Vehicle': 'cam2vehicle',
        '[2] Vehicle Pose: Vehicle → World': 'vehicle2world',
        '[3] Extrinsic: Camera → World': 'cam2world',
    }

    # 텍스트 파일의 각 줄을 순회합니다.
    for line in lines:
        # 줄 앞뒤의 공백 및 개행문자 제거
        stripped = line.strip()
        found = False
        
        # 현재 줄이 새로운 섹션의 시작(헤더)인지 확인
        for header, key in section_map.items():
            if stripped.startswith(header):
                # 새로운 섹션을 만났고, 이전에 버퍼에 4줄 이상의 데이터가 있다면 행렬로 변환하여 저장
                if current_key and len(buffer) >= 4:
                    matrices[current_key] = np.array(buffer[:4])
                # 현재 처리 중인 키를 업데이트
                current_key = key
                # 버퍼 초기화
                buffer = []
                found = True
                break
        
        # 헤더를 찾은 줄이면 다음 줄로 넘어감
        if found: continue
        # 빈 줄이거나 주석('#'), 또는 다른 알 수 없는 괄호('[')로 시작하는 줄은 무시
        if not stripped or stripped.startswith('#') or stripped.startswith('['):
            continue
            
        try:
            # 공백을 기준으로 문자열을 분리하고, 각각을 실수(float)로 변환
            row = [float(x) for x in stripped.split()]
            # 4개의 숫자로 이루어진 유효한 행렬의 행이라면 버퍼에 추가
            if len(row) == 4:
                buffer.append(row)
        except ValueError:
            # 숫자로 변환할 수 없는 예외 상황 발생 시 무시
            continue
            
    # 마지막 섹션의 데이터가 버퍼에 남아있다면 마저 행렬로 변환하여 저장
    if current_key and len(buffer) >= 4:
        matrices[current_key] = np.array(buffer[:4])

    # 완성된 행렬 딕셔너리 반환
    return matrices

def load_intrinsic_txt(path: str) -> dict:
    """
    텍스트 파일에서 카메라의 내부 파라미터(Intrinsic)를 읽어옵니다.
    출력 형식: {'K': 3x3 카메라 행렬, 'distortion': 왜곡 계수, 'width': 너비, 'height': 높이, 'raw': 원본 데이터}
    """
    # 텍스트 파일을 통째로 읽어 하나의 문자열로 저장합니다.
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 정규 표현식을 사용해 너비(width)와 높이(height) 정수값을 추출
    width = int(re.search(r'width\s*:\s*(\d+)', text).group(1))
    height = int(re.search(r'height\s*:\s*(\d+)', text).group(1))
    
    # 추출할 파라미터 키 목록 (초점거리, 주점, 왜곡 계수들)
    keys = ['f_u', 'f_v', 'c_u', 'c_v', 'k1', 'k2', 'p1', 'p2', 'k3']
    # 각 키에 해당하는 실수값을 정규식으로 추출하여 numpy 배열로 변환
    raw = np.array([float(re.search(rf'{k}\s*:\s*([-\d.eE+]+)', text).group(1)) for k in keys])

    # 추출한 파라미터(초점거리, 주점)를 이용하여 3x3 Camera Intrinsic 행렬(K) 구성
    K = np.array([
        [raw[0], 0.0,    raw[2]],  # [fx,  0, cx]
        [0.0,    raw[1], raw[3]],  # [ 0, fy, cy]
        [0.0,    0.0,    1.0   ],  # [ 0,  0,  1]
    ])
    
    # K 행렬, 왜곡 파라미터(5개), 해상도 등을 딕셔너리로 묶어서 반환
    return {'K': K, 'distortion': raw[4:9], 'width': width, 'height': height, 'raw': raw}

def load_panoptic_info_txt(path: str) -> dict:
    """
    Panoptic Segmentation 정보를 담은 텍스트 파일을 파싱합니다.
    출력 형식: {segment_id: {'label_id': int, 'class_name': str, 'score': float}}
    """
    # 세그먼트 ID를 키로 하는 정보 딕셔너리 초기화
    info_dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 데이터 영역이 시작되었는지 확인하는 플래그
    in_data = False
    for line in lines:
        stripped = line.strip()
        # '---' 로 시작하는 구분선 아래부터 실제 데이터로 간주
        if stripped.startswith('---'):
            in_data = True
            continue
        # 데이터 영역 밖이거나, 빈 줄이거나, 주석 줄이면 건너뜀
        if not in_data or not stripped or stripped.startswith('#'):
            continue
        
        # 공백 기준으로 데이터 분할
        parts = stripped.split()
        # 데이터가 최소 4개(seg_id, label_id, class_name, score)가 안되면 무시
        if len(parts) < 4: continue
        try:
            # 첫 번째 값은 세그먼트 고유 ID
            seg_id = int(parts[0])
            info_dict[seg_id] = {
                'label_id': int(parts[1]),                       # 두 번째 값은 클래스 라벨 ID
                'class_name': ' '.join(parts[2:-2]),             # 중간 값들은 띄어쓰기가 포함될 수 있는 클래스 이름
                'score': float(parts[-2]),                       # 뒤에서 두 번째 값은 신뢰도(score)
            }
        except (ValueError, IndexError):
            # 파싱 오류 발생 시 해당 줄 건너뜀
            continue
    return info_dict


# ================================================================================
# [Geometry Utils] 좌표계 변환 및 기하 유틸
# ================================================================================

def rotation_matrix_to_waymo_euler(R: np.ndarray) -> tuple:
    """
    3x3 회전 행렬을 Roll, Pitch, Yaw 각도(Degree 단위)로 변환합니다. (Waymo 좌표계 기준)
    """
    # Yaw 각도 계산 (Z축 회전)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    # Pitch 각도 계산 (Y축 회전)
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    # Roll 각도 계산 (X축 회전)
    roll = np.arctan2(R[2, 1], R[2, 2])
    # 라디안을 디그리(도) 단위로 변환하여 반환
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

def compute_waymo_frustum_corners(K: np.ndarray, width: int, height: int,
                                   cam2world: np.ndarray, depth: float = 3.0) -> tuple:
    """
    카메라의 시야각을 나타내는 절두체(Frustum)의 3D 모서리 좌표를 계산합니다.
    """
    # 이미지 평면의 4개 모서리 픽셀 좌표 지정 [좌상, 우상, 우하, 좌하]
    img_corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    # 카메라 행렬에서 초점거리와 주점 추출
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    
    # 픽셀 좌표를 정규화된 카메라 좌표계(z=1 평면)의 광선(Ray) 방향 벡터로 변환
    std_rays = np.column_stack([
        (img_corners[:, 0] - cx) / fx,  # x 좌표 정규화
        (img_corners[:, 1] - cy) / fy,  # y 좌표 정규화
        np.ones(4, dtype=np.float32),   # z 좌표는 1로 설정
    ])
    
    # 일반적인 카메라 좌표계(Z-forward, Y-down)를 Waymo 좌표계(X-forward, Y-left, Z-up)로 변환
    waymo_rays = np.column_stack([std_rays[:, 2], -std_rays[:, 0], -std_rays[:, 1]])
    
    # 지정한 깊이(depth)만큼 광선을 늘려 Frustum의 끝 평면 좌표 계산 (카메라 좌표계 기준)
    far_cam = waymo_rays * depth
    
    # Extrinsic 행렬에서 회전 행렬(R)과 이동 벡터(t) 추출
    R, t = cam2world[:3, :3], cam2world[:3, 3]
    
    # 카메라 좌표계의 Frustum 모서리를 World 좌표계로 변환
    far_world = (R @ far_cam.T).T + t
    
    # 원점(카메라 위치)과 World 좌표계로 변환된 Frustum 모서리 4개 반환
    return t, far_world

def draw_camera_3d(ax, cam2world: np.ndarray, K: np.ndarray, width: int, height: int,
                    color, label: str = None, frustum_depth: float = 3.0,
                    axis_length: float = 0.7, draw_axes: bool = True,
                    draw_frustum: bool = True, frustum_alpha: float = 0.15):
    """
    3D 그래프(ax) 상에 카메라의 위치, 방향축, 그리고 시야각(Frustum)을 그립니다.
    """
    # 카메라의 3D 위치(pos)와 회전(R) 추출
    pos, R = cam2world[:3, 3], cam2world[:3, :3]
    
    # 카메라 위치를 3D 산점도로 점으로 표시
    ax.scatter(*pos, color=color, s=60, edgecolors='black', linewidths=0.6, zorder=5, label=label)
    
    # 카메라의 로컬 축(Forward, Left, Up)을 그리는 옵션
    if draw_axes:
        # 회전 행렬의 각 열 벡터에 길이를 곱하여 축 벡터 생성
        fwd, lft, up_ = R[:, 0] * axis_length, R[:, 1] * axis_length, R[:, 2] * axis_length
        # Forward 방향 (X축, 빨간색) 선 그리기
        ax.plot([pos[0], pos[0]+fwd[0]], [pos[1], pos[1]+fwd[1]], [pos[2], pos[2]+fwd[2]], color='red', lw=1.3, alpha=0.7)
        # Left 방향 (Y축, 초록색) 선 그리기
        ax.plot([pos[0], pos[0]+lft[0]], [pos[1], pos[1]+lft[1]], [pos[2], pos[2]+lft[2]], color='green', lw=1.3, alpha=0.7)
        # Up 방향 (Z축, 파란색) 선 그리기
        ax.plot([pos[0], pos[0]+up_[0]], [pos[1], pos[1]+up_[1]], [pos[2], pos[2]+up_[2]], color='blue', lw=1.3, alpha=0.7)
        
    # 카메라의 시야각(Frustum) 피라미드를 그리는 옵션
    if draw_frustum:
        # 원점과 끝 평면 모서리 4개 계산
        origin, corners = compute_waymo_frustum_corners(K, width, height, cam2world, depth=frustum_depth)
        # 원점에서 각 모서리로 뻗어나가는 4개의 선(광선) 그리기
        for c in corners:
            ax.plot([origin[0], c[0]], [origin[1], c[1]], [origin[2], c[2]], color=color, lw=0.8, alpha=0.7)
        
        # 끝 평면의 모서리를 이어 사각형(프레임) 그리기 (첫 모서리를 다시 붙여 닫힌 도형으로 만듦)
        rect = np.vstack([corners, corners[0:1]])
        ax.plot(rect[:, 0], rect[:, 1], rect[:, 2], color=color, lw=1.3, alpha=0.9)
        
        # 끝 평면 내부를 반투명한 색상으로 칠하기 위해 다각형 컬렉션 추가
        poly = Poly3DCollection([corners.tolist()], facecolors=color, edgecolors=color, alpha=frustum_alpha)
        ax.add_collection3d(poly)

def set_axes_equal_3d(ax):
    """
    3D 그래프의 X, Y, Z 축의 스케일을 동일하게 맞추어 비율 왜곡을 방지합니다.
    """
    # 현재 각 축의 한계값(limit)을 가져옴
    xl, yl, zl = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    # 세 축 중 가장 큰 범위(길이)를 계산
    max_range = max(abs(xl[1]-xl[0]), abs(yl[1]-yl[0]), abs(zl[1]-zl[0]))
    # 각 축의 중심점을 계산
    xm, ym, zm = np.mean(xl), np.mean(yl), np.mean(zl)
    
    # 중심점을 기준으로 가장 큰 범위에 맞춰 모든 축의 한계값을 재설정
    ax.set_xlim3d([xm - max_range/2, xm + max_range/2])
    ax.set_ylim3d([ym - max_range/2, ym + max_range/2])
    ax.set_zlim3d([zm - max_range/2, zm + max_range/2])


# ================================================================================
# [Visualizers - 2D & Intrinsic]
# ================================================================================

def visualize_per_frame_combined(image: np.ndarray, depth_map: np.ndarray, seg_map: np.ndarray,
                                  info_dict: dict, frame_idx: int, save_path: str, max_depth: float = 50.0):
    """
    한 프레임에 대한 원본 이미지, 깊이 맵(Depth), 판옵틱 분할(Panoptic)을 가로로 배치하여 저장합니다.
    """
    # 1행 3열의 서브플롯 생성 (가로로 긴 비율)
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    
    # [1] 첫 번째 패널: 원본 이미지 렌더링
    axes[0].imshow(image)
    axes[0].set_title(f"Frame {frame_idx} - Original Image", fontsize=13)
    axes[0].axis('off') # 축 숨기기

    # [2] 두 번째 패널: Depth Map 렌더링 (turbo 컬러맵 적용, 거리 상한선 설정)
    im = axes[1].imshow(depth_map, cmap='turbo', vmin=0, vmax=max_depth)
    axes[1].set_title(f"Frame {frame_idx} - Depth Map (DepthPro)", fontsize=13)
    axes[1].axis('off')
    # 컬러바 추가하여 깊이 값(m) 기준 표시
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Depth (m)', fontsize=11)

    # [3] 세 번째 패널: Panoptic Segmentation Map 준비
    unique_ids = np.unique(seg_map) # 존재하는 고유한 세그먼트 ID들 추출
    # ID 개수에 맞춰 tab20 컬러맵에서 색상 추출
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(unique_ids), 20)))
    # 컬러 분할 맵을 저장할 빈 배열(RGB) 생성
    color_seg = np.zeros((*seg_map.shape, 3), dtype=np.float32)
    
    # 각 세그먼트 ID마다 고유한 색상을 배열에 채색
    for i, sid in enumerate(unique_ids):
        if sid == 0: continue # ID가 0인 배경(또는 무효 영역)은 스킵
        color_seg[seg_map == sid] = colors[i % len(colors)][:3]

    # 원본 이미지와 컬러 분할 맵을 반반 섞어서(알파 블렌딩) 렌더링
    blended = (0.5 * image / 255.0 + 0.5 * color_seg).clip(0, 1)
    axes[2].imshow(blended)
    axes[2].set_title(f"Frame {frame_idx} - Panoptic (Mask2Former)", fontsize=13)
    axes[2].axis('off')

    # 세그먼트의 중앙 위치를 찾아 클래스 이름(텍스트) 오버레이
    for sid in unique_ids:
        if sid == 0 or sid not in info_dict: continue
        ys, xs = np.where(seg_map == sid)
        if len(xs) == 0: continue
        # 세그먼트 좌표의 평균값을 구해서 중앙점 텍스트 삽입
        axes[2].text(int(xs.mean()), int(ys.mean()), info_dict[sid]['class_name'],
                     fontsize=9, color='white', ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))
                     
    # 플롯 간격 자동 조절 후 이미지 저장
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig) # 메모리 누수 방지를 위해 figure 닫기

def visualize_intrinsic_on_image(image: np.ndarray, intr: dict, frame_idx: int, save_path: str):
    """
    Intrinsic 파라미터를 3개의 패널로 직관적으로 시각화합니다.
    1. 왼쪽: 원본 이미지 위의 주점(Optical Center) vs 화면 중심 오프셋
    2. 중앙: 위에서 내려다본 Pinhole Camera 물리 도식 (초점 거리, FoV, 렌즈 어긋남)
    3. 오른쪽: 파라미터 수치 정보
    """
    # Intrinsic 정보 할당
    K, width, height = intr['K'], intr['width'], intr['height']
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    
    # 초점거리와 해상도를 이용해 가로/세로 시야각(FoV) 계산
    fov_x = 2 * np.degrees(np.arctan(width / (2 * fx)))
    fov_y = 2 * np.degrees(np.arctan(height / (2 * fy)))
    # 픽셀 좌표 기준 이미지의 정중앙 계산
    img_cx, img_cy = width / 2.0, height / 2.0

    # 1행 3열 서브플롯 생성 (각 패널의 가로 비율을 지정)
    fig, (ax_img, ax_pin, ax_info) = plt.subplots(
        1, 3, figsize=(24, 7), gridspec_kw={'width_ratios': [1.5, 1.5, 1]}
    )
    
    # ==========================================================
    # [Panel 1] 원본 이미지 위 오버레이 (어디가 진짜 중심인가?)
    # ==========================================================
    # 원본 이미지 출력
    ax_img.imshow(image)
    # 이미지의 외곽선을 나타내는 청록색 사각형 추가
    ax_img.add_patch(Rectangle((0, 0), width, height, fill=False, edgecolor='cyan', lw=2))
    
    # 이미지의 기하학적 중심에 회색 'x' 표시
    ax_img.plot(img_cx, img_cy, 'x', color='lightgray', markersize=15, mew=3, 
                label=f'Image Center ({img_cx:.0f}, {img_cy:.0f})')
    # 실제 렌즈의 광학적 중심(주점)에 빨간색 '+' 표시
    ax_img.plot(cx, cy, '+', color='red', markersize=20, mew=4, 
                label=f'Principal Point (Optical Center)\ncx={cx:.1f}, cy={cy:.1f}')
    
    # 이미지 중심과 광학적 중심이 어긋나 있을 경우(1px 이상), 그 차이를 화살표로 연결
    if abs(cx - img_cx) > 1 or abs(cy - img_cy) > 1:
        ax_img.annotate('', xy=(img_cx, img_cy), xytext=(cx, cy), 
                        arrowprops=dict(arrowstyle='->', color='yellow', lw=2))

    # 패널의 여백을 약간 두어 시각적으로 편하게 설정 (y축은 아래로 갈수록 값이 커지므로 역순)
    ax_img.set_xlim(-50, width + 50)
    ax_img.set_ylim(height + 50, -50)
    ax_img.set_title(f"Frame {frame_idx} - Image Plane View", fontsize=14, fontweight='bold')
    ax_img.legend(loc='lower left', fontsize=10, framealpha=0.9)


    # ==========================================================
    # [Panel 2] Top-Down Pinhole Model (초점거리와 화각의 물리적 의미)
    # ==========================================================
    # 카메라 렌즈 위치를 원점으로 검은 점 표시
    ax_pin.plot(0, 0, 'ko', markersize=10, label='Camera Lens (Origin)')
    
    # 렌즈를 수직으로 통과하는 빛의 경로인 광축(Optical Axis)을 점선으로 표시
    ax_pin.plot([0, 0], [0, fx * 1.2], 'k-.', lw=1.5, label='Optical Axis')
    
    # 카메라 원점(x=0)에서 뻗어나간 광축은 이미지의 cx 픽셀에 닿습니다.
    # 따라서 1D 센서 라인의 왼쪽 끝(u=0)은 -cx, 오른쪽 끝(u=width)은 width-cx에 위치하게 됩니다.
    left_x = -cx
    right_x = width - cx
    
    # 원점부터 양 끝단으로 퍼져나가는 영역(시야각 FoV)을 청록색으로 칠함
    ax_pin.fill_between([left_x, 0, right_x], [fx, 0, fx], color='cyan', alpha=0.1)
    
    # 초점거리(f_x)만큼 떨어져 있는 이미지 센서 평면을 굵은 파란색 선으로 그림
    ax_pin.plot([left_x, right_x], [fx, fx], color='blue', lw=5, solid_capstyle='round', label='Image Sensor Plane')
    
    # 광축이 센서 평면과 만나는 곳(주점)에 빨간 점 표시
    ax_pin.plot(0, fx, 'ro', markersize=8, label='Principal Point (Hits Optical Axis)')
    # 물리적 센서의 정중앙에 회색 'x' 표시
    ax_pin.plot((left_x + right_x) / 2, fx, 'x', color='gray', markersize=10, mew=3, label='Physical Sensor Center')
    
    # 시야각(화각)의 좌/우 경계선을 파란 점선으로 표시
    ax_pin.plot([0, left_x], [0, fx], 'b--', lw=1.5)
    ax_pin.plot([0, right_x], [0, fx], 'b--', lw=1.5)
    
    # 초점거리(Focal Length)를 나타내는 양방향 화살표와 텍스트 라벨 추가
    ax_pin.annotate('', xy=(0, 0), xytext=(0, fx), arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    ax_pin.text(fx * 0.05, fx / 2, f"Focal Length ($f_x$) = {fx:.1f} px", 
                color='purple', fontsize=12, fontweight='bold', va='center')
    
    # 시야각(FoV) 수치를 상단에 텍스트 상자로 추가
    ax_pin.text(0, fx * 0.8, f"Horizontal FoV: {fov_x:.1f}°", 
                ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 도식의 축 범위 설정 및 비율을 1:1로 맞춰 실제 각도 왜곡 방지
    ax_pin.set_xlim(-width * 0.8, width * 0.8)
    ax_pin.set_ylim(-fx * 0.1, fx * 1.3)
    ax_pin.set_aspect('equal') 
    ax_pin.grid(True, alpha=0.3)
    ax_pin.set_title("Top-Down Pinhole Schematic", fontsize=14, fontweight='bold')
    ax_pin.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=10)


    # ==========================================================
    # [Panel 3] 수치 데이터 텍스트 정보
    # ==========================================================
    # 패널 3에 출력할 수치 정보들을 미리 포맷팅하여 긴 문자열로 준비
    info_text = (
        f"Camera Intrinsic Parameters\n{'=' * 40}\n\n"
        f"1. Resolution (해상도)\n  Width: {width} px\n  Height: {height} px\n\n"
        f"2. Focal Length (초점 거리 - 줌의 정도)\n  f_x = {fx:.2f} px\n  f_y = {fy:.2f} px\n\n"
        f"3. Principal Point (주점 - 렌즈 중심의 어긋남)\n  c_x = {cx:.2f} px\n  c_y = {cy:.2f} px\n"
        f"  (Offset from center: Δx = {cx - img_cx:+.1f}, Δy = {cy - img_cy:+.1f})\n\n"
        f"4. Field of View (화각 - 보이는 범위)\n  Horizontal: {fov_x:.2f}°\n  Vertical: {fov_y:.2f}°\n\n"
        f"5. Lens Distortion (왜곡 계수)\n"
        f"  k1 = {intr['distortion'][0]:+.5f}  (방사 왜곡)\n"
        f"  k2 = {intr['distortion'][1]:+.5f}\n"
        f"  p1 = {intr['distortion'][2]:+.5f}  (접선 왜곡)\n"
        f"  p2 = {intr['distortion'][3]:+.5f}\n"
        f"  k3 = {intr['distortion'][4]:+.5f}\n\n"
        f"6. Camera Matrix (K)\n"
        f"  [[{K[0,0]:8.2f}, {K[0,1]:8.2f}, {K[0,2]:8.2f}],\n"
        f"   [{K[1,0]:8.2f}, {K[1,1]:8.2f}, {K[1,2]:8.2f}],\n"
        f"   [{K[2,0]:8.2f}, {K[2,1]:8.2f}, {K[2,2]:8.2f}]]\n"
    )
    # 패널 3은 축을 숨기고 텍스트만 표시
    ax_info.axis('off')
    # 준비된 문자열을 고정폭 글꼴(monospace) 박스 형태로 출력
    ax_info.text(0.05, 0.95, info_text, fontsize=11, fontfamily='monospace', 
                 va='top', ha='left', transform=ax_info.transAxes,
                 bbox=dict(boxstyle='round', facecolor='whitesmoke', edgecolor='gray', alpha=1.0))
    
    # 레이아웃 정돈 및 이미지 파일 저장
    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    
def visualize_summary_grid(image_list, depth_list, seg_list, frame_indices, save_path, max_depth=50.0):
    """
    수집된 모든 프레임의 이미지, 뎁스, 판옵틱 맵을 3행 N열의 격자(Grid) 형태로 한 장에 요약하여 저장합니다.
    """
    n_frames = len(image_list)
    # 3행 x (프레임 수)열의 서브플롯 생성. 프레임이 많을수록 가로가 길어지도록 동적 사이즈 할당
    fig, axes = plt.subplots(3, n_frames, figsize=(2.5 * n_frames, 7))
    # 프레임이 1개뿐인 경우 numpy 배열 차원을 3x1로 맞춰줌
    if n_frames == 1: axes = axes.reshape(3, 1)

    # 각 열(프레임) 단위로 순회하며 이미지 출력
    for col, (img, dm, seg, fidx) in enumerate(zip(image_list, depth_list, seg_list, frame_indices)):
        # 1행: 원본 이미지
        axes[0, col].imshow(img); axes[0, col].set_title(f'f{fidx}', fontsize=10); axes[0, col].axis('off')
        # 2행: Depth Map
        axes[1, col].imshow(dm, cmap='turbo', vmin=0, vmax=max_depth); axes[1, col].axis('off')
        # 3행: Panoptic Segmentation Map
        axes[2, col].imshow(seg, cmap='tab20'); axes[2, col].axis('off')

    # 맨 왼쪽 첫 번째 열에 각 행의 이름(Image, Depth, Panoptic)을 세로 텍스트로 라벨링
    for row, label in enumerate(['Image', 'Depth', 'Panoptic']):
        axes[row, 0].set_ylabel(label, fontsize=12, rotation=90, labelpad=20)
        axes[row, 0].axis('on'); axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])

    # 전체 제목 추가 후 저장
    plt.suptitle(f'Summary Grid ({n_frames} frames extracted)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# ================================================================================
# [Visualizers - Extrinsic 3D]
# ================================================================================

def visualize_multiview_3d(extrinsics, intrinsics, save_path, frustum_depth=4.0):
    """
    카메라의 3D 이동 궤적과 Frustum을 4가지 각도(Perspective, Top, Side, Front)에서 관찰합니다.
    """
    # 모든 프레임의 World 좌표계 기준 카메라 위치 추출 (N x 3)
    positions = np.array([e['cam2world'][:3, 3] for e in extrinsics])
    # 궤적의 중심점(평균 위치) 계산 (시점을 중앙으로 맞추기 위함)
    center = positions.mean(axis=0)
    # 프레임 순서에 따라 색상을 파란색->빨간색 그라데이션으로 생성
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(extrinsics)))

    fig = plt.figure(figsize=(20, 16))
    # 4개의 뷰포트에 대한 설정 (제목, 고도각(elev), 방위각(azim))
    view_specs = [('Perspective View', 20, -60), ('Top View', 90, -90), 
                  ('Side View', 0, 0), ('Front View', 0, 90)]

    # 각 뷰포트 안에 3D 장면을 그리는 내부 헬퍼 함수
    def _draw_scene(ax):
        # 모든 카메라 프레임을 순회하며 렌더링
        for i, (ext, intr) in enumerate(zip(extrinsics, intrinsics)):
            # 보기 편하도록 중심점을 원점으로 평행이동
            cam2w = ext['cam2world'].copy(); cam2w[:3, 3] -= center
            # 앞서 만든 함수를 호출하여 3D 카메라와 시야각 피라미드를 그림
            draw_camera_3d(ax, cam2w, intr['K'], intr['width'], intr['height'], color=colors[i], frustum_depth=frustum_depth)
            # 카메라 바로 위에 프레임 번호 텍스트 삽입
            ax.text(*cam2w[:3, 3]+[0,0,0.4], f"f{ext['frame_idx']}", fontsize=8, ha='center')
            
        # 카메라들의 궤적을 점선으로 연결하여 이동 경로 표시
        rel_pos = positions - center
        ax.plot(rel_pos[:, 0], rel_pos[:, 1], rel_pos[:, 2], 'k--', lw=1.2, alpha=0.4)
        
        # 월드 좌표계의 원점 기준축 (X:빨강, Y:초록, Z:파랑) 표시
        ax.plot([0, 1.5], [0, 0], [0, 0], color='red', lw=2)
        ax.plot([0, 0], [0, 1.5], [0, 0], color='green', lw=2)
        ax.plot([0, 0], [0, 0], [0, 1.5], color='blue', lw=2)
        
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        # 비율 왜곡 방지
        set_axes_equal_3d(ax)

    # 4가지 뷰포트에 대해 서브플롯을 생성하고 시점을 바꿔가며 렌더링
    for idx, (title, elev, azim) in enumerate(view_specs):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        _draw_scene(ax)           # 장면 그리기
        ax.view_init(elev=elev, azim=azim) # 시점(각도) 설정
        ax.set_title(title, fontsize=13, pad=10)

    fig.suptitle(f'Camera Extrinsic - 4-Way 3D View', fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(save_path, dpi=110, bbox_inches='tight')
    plt.close(fig)

def visualize_topdown_2d(extrinsics, save_path, heading_arrow_length=1.5):
    """
    자율주행 차량/카메라의 이동 경로를 위에서 내려다본(Bird's Eye View) 2D 평면도로 시각화합니다.
    """
    # X, Y, Z 위치 추출
    positions = np.array([e['cam2world'][:3, 3] for e in extrinsics])
    # Z축 무시하고 X, Y 평면으로 투영 후 시작점(또는 평균점) 기준으로 영점 조절
    pos_xy = positions[:, :2] - positions[:, :2].mean(axis=0)
    
    # 카메라가 바라보는 전방(X축)의 2D 방향 벡터 추출
    headings = np.array([e['cam2world'][:3, 0][:2] for e in extrinsics])
    # 카메라의 Yaw(회전각) 추출
    yaws = np.array([rotation_matrix_to_waymo_euler(e['cam2world'][:3, :3])[2] for e in extrinsics])
    # 프레임별 색상
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(extrinsics)))

    fig, ax = plt.subplots(figsize=(14, 12))
    # 전체 이동 경로 점선 표시
    ax.plot(pos_xy[:, 0], pos_xy[:, 1], 'k--', lw=1.2, alpha=0.4, zorder=1)

    for i, (ext, h, y) in enumerate(zip(extrinsics, headings, yaws)):
        x, yp = pos_xy[i]
        # 해당 프레임 위치에 원 그리기
        ax.scatter(x, yp, color=colors[i], s=180, edgecolors='black', lw=1.2, zorder=5)
        # 바라보는 방향을 화살표로 렌더링 (벡터 정규화 후 길이 곱함)
        h_norm = h / (np.linalg.norm(h) + 1e-9) * heading_arrow_length
        ax.annotate('', xy=(x+h_norm[0], yp+h_norm[1]), xytext=(x, yp), arrowprops=dict(arrowstyle='->', color=colors[i], lw=2), zorder=4)
        # 프레임 번호와 Yaw 각도를 말풍선 텍스트로 표시
        ax.annotate(f"f{ext['frame_idx']}\n(yaw={y:+.1f}°)", xy=(x, yp), xytext=(8, 8), textcoords='offset points',
                    fontsize=9, bbox=dict(boxstyle='round', facecolor='white', edgecolor=colors[i], alpha=0.85))

    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.set_title("Top-Down (Bird's Eye) View", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)

def visualize_6dof_timeline(extrinsics, save_path):
    """
    시간(프레임)의 흐름에 따른 카메라의 6자유도(Translation X/Y/Z, Rotation Roll/Pitch/Yaw) 변화량을 라인 그래프로 그립니다.
    """
    # 프레임 번호 리스트 추출
    f_idx = np.array([e['frame_idx'] for e in extrinsics])
    # 3D 위치(Translation) 추출
    pos = np.array([e['cam2world'][:3, 3] for e in extrinsics])
    # 첫 프레임 위치를 0으로 맞춘 상대적인 이동량 계산
    rel_pos = pos - pos[0]
    # 회전 행렬을 오일러 각도(Roll, Pitch, Yaw)로 변환
    eulers = np.array([rotation_matrix_to_waymo_euler(e['cam2world'][:3, :3]) for e in extrinsics])

    # 2행 3열 서브플롯 생성 (위: Translation, 아래: Rotation)
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    trans_labels = [('X (forward)', 'tab:blue'), ('Y (left)', 'tab:green'), ('Z (up)', 'tab:orange')]
    rot_labels = [('Roll', 'tab:red'), ('Pitch', 'tab:purple'), ('Yaw', 'tab:brown')]

    # Translation X, Y, Z 그래프 그리기
    for i, (label, col) in enumerate(trans_labels):
        axes[0, i].plot(f_idx, rel_pos[:, i], 'o-', color=col, lw=2); axes[0, i].set_title(f"Translation: {label}")
        axes[0, i].grid(True, alpha=0.4); axes[0, i].axhline(y=0, color='gray', ls=':')
    # Rotation Roll, Pitch, Yaw 그래프 그리기
    for i, (label, col) in enumerate(rot_labels):
        axes[1, i].plot(f_idx, eulers[:, i], 'o-', color=col, lw=2); axes[1, i].set_title(f"Rotation: {label}")
        axes[1, i].grid(True, alpha=0.4); axes[1, i].axhline(y=0, color='gray', ls=':')

    fig.suptitle("6-DOF Timeline", fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)

def visualize_frustums_with_thumbnails(extrinsics, intrinsics, thumb_list, save_path, frustum_depth=4.0):
    """
    왼쪽 거대한 패널에는 모든 카메라의 3D Frustum을 겹쳐 그리고, 오른쪽에는 각 프레임에 해당하는 원본 이미지 썸네일을 배치합니다.
    """
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(extrinsics)))
    fig = plt.figure(figsize=(22, 13))
    # 그리드 시스템을 사용하여 비율이 다른 서브플롯을 배치 (3:3:3:1 비율)
    gs = fig.add_gridspec(nrows=len(extrinsics), ncols=4, width_ratios=[3, 3, 3, 1])

    # 왼쪽 패널: 3D Frustum 통합 뷰
    ax3d = fig.add_subplot(gs[:, :3], projection='3d')
    center = np.array([e['cam2world'][:3, 3] for e in extrinsics]).mean(axis=0)

    # 궤적을 그리며 3D 카메라 렌더링
    for i, (ext, intr) in enumerate(zip(extrinsics, intrinsics)):
        cam2w = ext['cam2world'].copy(); cam2w[:3, 3] -= center
        draw_camera_3d(ax3d, cam2w, intr['K'], intr['width'], intr['height'], color=colors[i], frustum_depth=frustum_depth, frustum_alpha=0.12)
    ax3d.set_title('3D Frustums', fontsize=13); set_axes_equal_3d(ax3d)

    # 오른쪽 패널: 프레임 썸네일 세로 리스트
    for i, (ext, img) in enumerate(zip(extrinsics, thumb_list)):
        # 각 행의 오른쪽 끝 셀에 이미지 할당
        ax_t = fig.add_subplot(gs[i, 3]); ax_t.imshow(img); ax_t.set_xticks([]); ax_t.set_yticks([])
        # 썸네일 테두리를 해당 프레임의 3D 색상과 동일하게 칠하여 매칭
        for spine in ax_t.spines.values(): spine.set_edgecolor(colors[i]); spine.set_linewidth(3)
        ax_t.set_ylabel(f"f{ext['frame_idx']}", rotation=0, labelpad=20, ha='right', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches='tight')
    plt.close(fig)

def visualize_vehicle_vs_camera(extrinsics, save_path):
    """
    동일 프레임에서의 차량(Vehicle) 위치와 해당 카메라(Camera) 위치의 차이점(장착 오프셋)을 3D 및 2D 궤적으로 시각화합니다.
    """
    n = len(extrinsics)
    colors = plt.cm.coolwarm(np.linspace(0, 1, n))
    # Vehicle(차량 중심)의 3D 위치 궤적
    vp = np.array([e['vehicle2world'][:3, 3] for e in extrinsics])
    # Camera(렌즈 중심)의 3D 위치 궤적
    cp = np.array([e['cam2world'][:3, 3] for e in extrinsics])
    
    # 기준점을 차량 궤적의 중심에 맞춰 평행이동
    center = vp.mean(axis=0); vp -= center; cp -= center

    fig = plt.figure(figsize=(22, 10))
    ax3d = fig.add_subplot(1, 2, 1, projection='3d') # 3D 뷰 서브플롯
    ax2d = fig.add_subplot(1, 2, 2)                  # 2D 평면 서브플롯

    # 3D 뷰 렌더링
    ax3d.plot(vp[:, 0], vp[:, 1], vp[:, 2], 'b-', lw=2, label='Vehicle') # 파란선: 차량
    ax3d.plot(cp[:, 0], cp[:, 1], cp[:, 2], 'r-', lw=2, label='Camera')  # 빨간선: 카메라
    # 동일 시간대(동일 인덱스)의 차량과 카메라를 선으로 연결하여 장착 오프셋 확인
    for i in range(n): ax3d.plot([vp[i,0], cp[i,0]], [vp[i,1], cp[i,1]], [vp[i,2], cp[i,2]], color=colors[i], lw=1.0)
    ax3d.set_title('3D: Vehicle vs Camera'); set_axes_equal_3d(ax3d)

    # 2D 평면 렌더링 (Z축 생략)
    ax2d.plot(vp[:, 0], vp[:, 1], 'b-s', label='Vehicle'); ax2d.plot(cp[:, 0], cp[:, 1], 'r-o', label='Camera')
    for i in range(n): ax2d.plot([vp[i,0], cp[i,0]], [vp[i,1], cp[i,1]], color=colors[i], lw=1.0)
    ax2d.set_aspect('equal'); ax2d.grid(True)

    fig.suptitle('Vehicle Pose vs Camera Pose', fontsize=15, y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ================================================================================
# [Main]
# ================================================================================
# 이 스크립트가 직접 실행될 때만 작동하는 메인 블록
if __name__ == "__main__":
    # ── 1. 기본 설정 및 시각화 결과를 저장할 폴더 구조 생성 ──
    OUTPUT_ROOT = './output'      # 데이터를 읽어올 입력 폴더
    VIS_ROOT = './visualization'  # 시각화 결과를 저장할 최상위 출력 폴더
    MAX_DEPTH = 50.0              # Depth Map 시각화 거리 한계 (m)
    FRUSTUM_DEPTH = 4.0           # 3D Frustum 시각화 깊이

    # 세부 출력 폴더 경로를 딕셔너리로 정의
    DIRS = {
        'per_frame': os.path.join(VIS_ROOT, 'per_frame'),
        'intrinsic': os.path.join(VIS_ROOT, 'intrinsic'),
        'extrinsic_detailed': os.path.join(VIS_ROOT, 'extrinsic_detailed')
    }
    # 지정한 폴더가 없다면 모두 생성 (exist_ok=True 옵션으로 이미 있어도 에러 안 남)
    for d in DIRS.values(): os.makedirs(d, exist_ok=True)

    # ── 2. 입력 폴더에서 처리 가능한 프레임 탐지 ──
    # extrinsic 폴더 안의 '.txt' 파일 목록을 찾아서 알파벳 순(프레임 순)으로 정렬
    extr_files = sorted([f for f in os.listdir(os.path.join(OUTPUT_ROOT, 'extrinsic')) if f.endswith('.txt')])
    if not extr_files:
        raise FileNotFoundError(f"{OUTPUT_ROOT}/extrinsic/ 에서 파일을 찾을 수 없습니다.")
    
    # 정규표현식을 사용해 파일명(예: frame_0001.txt)에서 프레임 번호만 정수로 추출
    frame_indices = [int(re.search(r'frame_(\d+)', f).group(1)) for f in extr_files]
    print(f"[INFO] 발견된 프레임: {frame_indices} (총 {len(frame_indices)}개)\n")

    # ── 3. 데이터 로드 및 개별 프레임 시각화 실행루프 ──
    # 전체 요약 시각화 등에서 한꺼번에 쓸 수 있도록 모든 데이터를 담아둘 리스트 생성
    all_images_full, all_images_thumb = [], []
    all_depths, all_segs, all_extrinsics, all_intrinsics = [], [], [], []

    # 각 프레임별로 루프 실행
    for fidx in frame_indices:
        prefix = f"frame_{fidx:04d}" # 4자리 숫자로 포맷팅 (예: 1 -> "frame_0001")
        print(f"[PROC] {prefix} 처리 중...")

        # 이미지 로드 및 썸네일(축소판) 생성 (Lanczos 리샘플링 사용으로 해상도 저하 최소화)
        img_pil = Image.open(os.path.join(OUTPUT_ROOT, 'image', f"{prefix}.jpg"))
        img_full = np.array(img_pil)
        w_new = 400 # 썸네일의 고정 너비 설정
        img_thumb = np.array(img_pil.resize((w_new, int(img_pil.height * w_new / img_pil.width)), Image.Resampling.LANCZOS))
        
        # NumPy 배열로 저장된 깊이 및 판옵틱 분할 맵 로드
        depth_map = np.load(os.path.join(OUTPUT_ROOT, 'depth', f"{prefix}.npy"))
        seg_map = np.load(os.path.join(OUTPUT_ROOT, 'panoptic', f"{prefix}.npy"))
        
        # Panoptic ID와 라벨 정보를 담고 있는 텍스트 로드
        info_dict = load_panoptic_info_txt(os.path.join(OUTPUT_ROOT, 'panoptic', f"{prefix}_info.txt"))
        
        # Extrinsic(외부 파라미터) 텍스트 로드 후 해당 딕셔너리에 프레임 번호 주입
        ext = load_extrinsic_txt(os.path.join(OUTPUT_ROOT, 'extrinsic', f"{prefix}.txt"))
        ext['frame_idx'] = fidx
        
        # Intrinsic(내부 파라미터) 텍스트 로드
        intr = load_intrinsic_txt(os.path.join(OUTPUT_ROOT, 'intrinsic', f"{prefix}.txt"))

        # [시각화 함수 호출] 해당 프레임의 데이터를 가지고 2D 및 Intrinsic 시각화 이미지 생성/저장
        visualize_per_frame_combined(img_full, depth_map, seg_map, info_dict, fidx, os.path.join(DIRS['per_frame'], f"{prefix}.png"), MAX_DEPTH)
        visualize_intrinsic_on_image(img_full, intr, fidx, os.path.join(DIRS['intrinsic'], f"{prefix}.png"))

        # 추후 통합 시각화를 위해 로드한 데이터들을 전체 리스트에 누적(Append)
        all_images_full.append(img_full); all_images_thumb.append(img_thumb)
        all_depths.append(depth_map); all_segs.append(seg_map)
        all_extrinsics.append(ext); all_intrinsics.append(intr)

    # ── 4. 전체 프레임 요약 이미지 (General) 생성 ──
    print("\n[PROC] Summary Grid 생성 중...")
    visualize_summary_grid(all_images_full, all_depths, all_segs, frame_indices, os.path.join(VIS_ROOT, 'summary_grid.png'), MAX_DEPTH)

    # ── 5. Extrinsic 관련 통합/상세 3D 시각화 (Detailed) 생성 ──
    # (앞서 모아둔 모든 프레임의 Extrinsic, Intrinsic 데이터를 한꺼번에 집어넣어 그립니다)
    print("[PROC] Extrinsic 상세 시각화 5종 생성 중...")
    ext_dir = DIRS['extrinsic_detailed']
    visualize_multiview_3d(all_extrinsics, all_intrinsics, os.path.join(ext_dir, '1_multiview_3d.png'), FRUSTUM_DEPTH)
    visualize_topdown_2d(all_extrinsics, os.path.join(ext_dir, '2_topdown_2d.png'))
    visualize_6dof_timeline(all_extrinsics, os.path.join(ext_dir, '3_6dof_timeline.png'))
    visualize_frustums_with_thumbnails(all_extrinsics, all_intrinsics, all_images_thumb, os.path.join(ext_dir, '4_frustums_with_thumbnails.png'), FRUSTUM_DEPTH)
    visualize_vehicle_vs_camera(all_extrinsics, os.path.join(ext_dir, '5_vehicle_vs_camera.png'))

    # 모든 작업 완료 메시지 출력
    print("\n✓ 모든 시각화 완료!")
    print(f"✓ 출력 위치: {VIS_ROOT}/")
    print(f"  ├── per_frame/         : [원본|뎁스|판옵틱] 통합 시각화")
    print(f"  ├── intrinsic/         : Intrinsic 기하학적 시각화")
    print(f"  ├── summary_grid.png   : 전체 프레임 요약 격자")
    print(f"  └── extrinsic_detailed/: 5종 Camera Extrinsic 상세 분석")