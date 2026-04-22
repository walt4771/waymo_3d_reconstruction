import tensorflow as tf
from waymo_open_dataset import dataset_pb2
import torch
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
import numpy as np
import depth_pro
import sys
sys.path.append('ml-depth-pro')

class WaymoFrameExtractor:
    def __init__(self, tfrecord_path: str):
        """
        Args:
            tfrecord_path (str): Waymo dataset tfrecord 파일의 경로
        """
        self.tfrecord_path = tfrecord_path

    def get_frame_list(self) -> list:
        """
        TFRecord 파일에서 frame 객체를 추출하여 리스트로 반환합니다.
        
        Returns:
            list: dataset_pb2.Frame 객체들의 리스트
        """
        dataset = tf.data.TFRecordDataset(self.tfrecord_path, compression_type='')
        frame_list = []
        
        for data in dataset:
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            frame_list.append(frame)
            
        return frame_list
    
    def get_camera_extrinsics(self, frame: dataset_pb2.Frame) -> dict:
            """
            주어진 프레임에서 모든 카메라의 외부 파라미터(Extrinsic parameters)를 추출합니다.
            
            Args:
                frame (dataset_pb2.Frame): 외부 파라미터를 추출할 단일 프레임 객체
                
            Returns:
                dict: 카메라 이름(dataset_pb2.CameraName)을 Key로 하고, 
                    4x4 변환 행렬(numpy.ndarray)을 Value로 가지는 딕셔너리.
            """
            camera_extrinsics = {}
            
            # frame.context.camera_calibrations에는 각 카메라의 캘리브레이션 정보가 들어있습니다.
            for camera_calibration in frame.context.camera_calibrations:
                camera_name = camera_calibration.name
                
                # extrinsic.transform은 길이가 16인 1차원 리스트이므로 4x4 행렬로 변환합니다.
                extrinsic_matrix = np.array(camera_calibration.extrinsic.transform).reshape(4, 4)
                
                camera_extrinsics[camera_name] = extrinsic_matrix
                
            return camera_extrinsics
        

class PanopticSegmenter:
    def __init__(self, device_id: str = None):
        """
        Mask2Former 모델 및 프로세서 초기화
        """
        model_id = "facebook/mask2former-swin-large-cityscapes-panoptic"
        self.device = torch.device(device_id if device_id else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # 프로세서와 모델 로드
        self.processor = Mask2FormerImageProcessor.from_pretrained(model_id)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

    def segment(self, image: Image.Image) -> dict:
        """
        이미지를 입력받아 panoptic segmentation을 수행합니다.
        
        Args:
            image (PIL.Image.Image): 분석할 원본 이미지
            
        Returns:
            dict: 'segmentation'(텐서) 및 'segments_info'(리스트)를 포함하는 딕셔너리
        """
        # 이미지 전처리
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # 추론
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # 후처리 (원본 이미지 크기로 복원)
        target_sizes = [image.shape[:2]]   # NumPy 배열인 경우: shape는 (height, width, channels)
        # target_sizes = [image.size[::-1]] # PIL Image인 경우: size는 (width, height)

        results = self.processor.post_process_panoptic_segmentation(
            outputs, target_sizes=target_sizes
        )[0]
        
        return results
    

class DepthProEstimator:
    def __init__(self, device_id: str = None):
        """
        DepthPro 모델 및 전처리 트랜스폼 초기화
        """
        self.device = torch.device(device_id if device_id else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.max_depth = 50.0 # 최대 거리 50m 설정

        # DepthPro 모델과 트랜스폼 로드
        self.model, self.transform = depth_pro.create_model_and_transforms()
        self.model.to(self.device)
        self.model.eval()

    def get_depth_map(self, image_input) -> np.ndarray:
        """
        이미지 경로 또는 Numpy 배열을 입력받아 Depth Map을 numpy array로 반환합니다.
        
        Args:
            image_input (str | np.ndarray): 입력 이미지 파일 경로 또는 Numpy 배열
            
        Returns:
            np.ndarray: Depth Map (Numpy 배열)
        """
        f_px = None  # 초점 거리 초기화

        # 1. 입력이 파일 경로(문자열)인 경우
        #image, _, f_px = depth_pro.load_rgb(image_input)
            
        # 2. 입력이 Numpy 배열인 경우
        # Numpy 배열을 PIL Image로 변환 (depth_pro의 transform이 PIL Image를 기대함)
        # 만약 배열이 0~1 사이의 float 형태라면 0~255 uint8로 변환
        if image_input.dtype in [np.float32, np.float64]:
            img_array_uint8 = (image_input * 255).astype(np.uint8)
        else:
            img_array_uint8 = image_input.astype(np.uint8)
            
        image = Image.fromarray(img_array_uint8)


        # 전처리
        input_tensor = self.transform(image).to(self.device)
        
        # 추론 수행
        with torch.no_grad():
            prediction = self.model.infer(input_tensor, f_px=f_px)
            
        # 결과 추출 및 CPU numpy 변환
        depth_map_npy = prediction["depth"].cpu().numpy()

        # 50m를 초과하는 값은 50으로 제한 (0 이하의 에러 값도 0으로 하한선 설정)
        dm_clipped = np.clip(depth_map_npy, a_min=0.0, a_max=self.max_depth)

        # 0 ~ 50m 범위를 0 ~ 255로 선형 변환
        # 가까운 곳은 0(검은색), 50m 이상 먼 곳은 255(흰색)로 맵핑됩니다.
        dm_vis = (dm_clipped / self.max_depth * 255).astype(np.uint8)
        
        return dm_clipped


def save_image(img_array, img_name):

    # 1. Numpy 배열을 PIL Image 객체로 변환
    # 주의: 배열의 데이터 타입이 반드시 np.uint8 이어야 합니다.
    img = Image.fromarray(img_array)

    # 2. JPG 파일로 저장
    img.save(img_name, format="JPEG")


if __name__=="__main__":
    # Waymo Dataset Images, Extrinsic
    wfe = WaymoFrameExtractor('./data/individual_files_validation_segment-10203656353524179475_7625_000_7645_000_with_camera_labels.tfrecord')
    frame = wfe.get_frame_list()[1]

    # frame 변수에 추출한 첫 번째 프레임이 담겨있다고 가정합니다. (예: frame = frames[0])
    target_camera = dataset_pb2.CameraName.FRONT
    front_img_array = None
    front_extrinsic = None

    # frame.images를 순회하는 단일 for문
    for img in frame.images:
        if img.name == target_camera:
            # 1. 이미지 추출 및 디코딩
            front_img_array = tf.io.decode_jpeg(img.image).numpy()
            
            # 2. frame.context에서 현재 카메라 이름과 일치하는 캘리브레이션 정보 매칭
            calibration = next(
                (cal for cal in frame.context.camera_calibrations if cal.name == target_camera), 
                None
            )
            
            # 3. 캘리브레이션 정보가 존재하면 4x4 Extrinsic 행렬로 변환
            if calibration:
                front_extrinsic = np.array(calibration.extrinsic.transform).reshape(4, 4)
                
            # 결과 출력
            print("Front Camera Image Shape:", front_img_array.shape)
            print("Front Camera Extrinsic Matrix:\n", front_extrinsic)
            
            # 원하는 전면 카메라 데이터를 모두 찾았으므로 불필요한 추가 순회를 막기 위해 루프 종료
            break

                
    ps = PanopticSegmenter()
    ps_res = ps.segment(front_img_array)
    # print(ps_res) # 딕셔너리 구조 확인용

    de = DepthProEstimator()
    dm = de.get_depth_map(front_img_array)
    # print(dm)

    # ---------------------------------------------------------
    # 이미지 저장 처리 부분
    # ---------------------------------------------------------

    # 1. Panoptic Segmentation 결과 처리
    # 딕셔너리에서 'segmentation' 텐서를 추출하고 CPU로 이동 후 Numpy 배열로 변환
    ps_map = ps_res['segmentation'].cpu().numpy()
    
    # 클래스/인스턴스 ID(정수) 형태이므로 눈으로 볼 수 있게 0~255로 스케일링
    if ps_map.max() > 0:
        ps_vis = (ps_map / ps_map.max() * 255).astype(np.uint8)
    else:
        ps_vis = ps_map.astype(np.uint8)

    # 2. Depth Map 결과 처리
    # 실수(float) 형태의 깊이 값을 0~255 사이의 uint8 이미지로 정규화
    dm_norm = (dm - dm.min()) / (dm.max() - dm.min() + 1e-8)
    dm_vis = (dm_norm * 255).astype(np.uint8)

    # 변환된 Numpy 배열 저장
    save_image(ps_vis, 'ps.jpg')
    save_image(dm_vis, 'dm.jpg')
