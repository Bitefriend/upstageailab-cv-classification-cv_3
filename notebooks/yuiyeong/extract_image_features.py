from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import entropy, kurtosis, skew
from tqdm import tqdm


class ImagePropertyExtractor:
    def __init__(self, image_meta_df: pd.DataFrame, directory: Path):
        self.image_meta_df = image_meta_df
        self.directory = directory

    def extract(self, n_jobs: int = -1) -> pd.DataFrame:
        """병렬 처리를 통한 특성 추출"""
        if n_jobs == -1:
            n_jobs = cpu_count()

        # 이미지 경로와 타겟 정보를 튜플로 묶기
        image_info_list = [(self.directory / row["ID"], row["target"]) for _, row in self.image_meta_df.iterrows()]
        total_images = len(image_info_list)

        # 병렬 처리
        with Pool(n_jobs) as pool:
            properties = pool.map(self._process_single_image, image_info_list)

        # 병렬 처리 (tqdm 사용)
        with Pool(n_jobs) as pool:
            properties = list(
                tqdm(pool.imap(self._process_single_image, image_info_list), total=total_images, desc="이미지 처리 중")
            )

        return pd.DataFrame(properties)

    @staticmethod
    def _process_single_image(image_info: tuple) -> dict:
        """단일 이미지 처리"""
        image_path, target = image_info

        if not image_path.exists():
            raise ValueError(f"파일이 존재하지 않음: {image_path}")

        # 이미지 로드
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"이미지 로드 실패: {image_path}")

        # 그레이스케일 변환
        if len(image.shape) == 2:
            image_gray = image
        else:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 모든 특성 추출
        properties = {}

        # 물리적 특성
        physical_props = ImagePropertyExtractor._extract_physical_properties(image_path, image)
        properties.update(physical_props)

        # 시각적 특성
        visual_props = ImagePropertyExtractor._extract_visual_properties(
            image, image_gray, physical_props["channels"], physical_props["total_pixels"]
        )
        properties.update(visual_props)

        # 나머지 특성들
        properties.update(ImagePropertyExtractor._extract_text_density(image_gray))
        properties.update(ImagePropertyExtractor._extract_noise_quality(image_gray))
        properties.update(ImagePropertyExtractor._extract_histogram_features(image_gray))
        properties.update(ImagePropertyExtractor._extract_geometric_features(image_gray))
        properties.update(ImagePropertyExtractor._extract_texture_features(image_gray))

        properties["target"] = target

        return properties

    @staticmethod
    def _extract_physical_properties(image_path: Path, image: np.ndarray) -> dict:
        """물리적 특성 추출"""
        file_size_kb = image_path.stat().st_size / 1024

        # 이미지 기본 정보
        if len(image.shape) == 2:  # 그레이스케일
            height, width = image.shape
            channels = 1
            mode = "GRAY"
        else:  # 컬러
            height, width, channels = image.shape
            if channels == 3:
                mode = "BGR"
            elif channels == 4:
                mode = "BGRA"
            else:
                mode = f"UNKNOWN ({channels} channels)"

        return {
            "width": int(width),
            "height": int(height),
            "channels": int(channels),
            "aspect_ratio": float(width) / float(height),
            "file_size_kb": int(file_size_kb),
            "color_mode": mode,
            "total_pixels": int(width * height),
        }

    @staticmethod
    def _extract_visual_properties(image: np.ndarray, image_gray: np.ndarray, channels: int, total_pixels: int) -> dict:
        """시각적 특성 추출"""
        brightness = image_gray.mean()
        contrast = image_gray.std()
        sharpness = cv2.Laplacian(image_gray, cv2.CV_64F).var()

        edges = cv2.Canny(image_gray, 100, 200)
        edge_ratio = np.count_nonzero(edges) / total_pixels

        if channels >= 3:
            b_mean, g_mean, r_mean = image.mean(axis=(0, 1))[:3]
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1].mean()
            hue = hsv[:, :, 0].mean()
        else:
            r_mean = g_mean = b_mean = brightness
            saturation = hue = 0

        return {
            "brightness": brightness,
            "contrast": contrast,
            "sharpness": sharpness,
            "edge_ratio": edge_ratio,
            "b_mean": b_mean,
            "g_mean": g_mean,
            "r_mean": r_mean,
            "saturation": saturation,
            "hue": hue,
        }

    @staticmethod
    def _extract_text_density(image_gray: np.ndarray) -> dict:
        """텍스트 밀도 관련 특성"""
        _, binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_density = np.sum(binary == 0) / binary.size
        white_space_ratio = np.sum(binary == 255) / binary.size

        return {"text_density": text_density, "white_space_ratio": white_space_ratio}

    @staticmethod
    def _extract_noise_quality(image_gray: np.ndarray) -> dict:
        """노이즈 및 품질 관련 특성"""
        laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)
        noise_level = np.std(laplacian)
        blur_metric = cv2.Laplacian(image_gray, cv2.CV_64F).var()
        jpeg_quality_estimate = ImagePropertyExtractor._estimate_jpeg_quality(image_gray)

        return {"noise_level": noise_level, "blur_metric": blur_metric, "jpeg_quality_estimate": jpeg_quality_estimate}

    @staticmethod
    def _extract_histogram_features(image_gray: np.ndarray) -> dict:
        """히스토그램 기반 특성"""
        hist = cv2.calcHist([image_gray], [0], None, [256], [0, 256]).flatten()
        hist_norm = hist / hist.sum()

        hist_entropy = entropy(hist_norm)
        hist_skewness = skew(image_gray.flatten())
        hist_kurtosis = kurtosis(image_gray.flatten())

        return {
            "histogram_entropy": hist_entropy,
            "histogram_skewness": hist_skewness,
            "histogram_kurtosis": hist_kurtosis,
        }

    @staticmethod
    def _extract_geometric_features(image_gray: np.ndarray) -> dict:
        """기하학적 변환 관련 특성"""
        edges = cv2.Canny(image_gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

        skew_angle = 0
        if lines is not None:
            angles = [line[0][1] for line in lines]
            skew_angle = np.median(angles) * 180 / np.pi - 90

        h, w = image_gray.shape
        is_rotated_90 = (h > w * 1.2) or (w > h * 1.2)

        margin_size = 50
        top_margin = image_gray[:margin_size, :].mean()
        bottom_margin = image_gray[-margin_size:, :].mean()
        left_margin = image_gray[:, :margin_size].mean()
        right_margin = image_gray[:, -margin_size:].mean()

        margin_uniformity = np.std([top_margin, bottom_margin, left_margin, right_margin])
        border_mean = np.mean([top_margin, bottom_margin, left_margin, right_margin])
        has_black_borders = border_mean < 50

        return {
            "skew_angle": skew_angle,
            "is_rotated_90": is_rotated_90,
            "margin_uniformity": margin_uniformity,
            "has_black_borders": has_black_borders,
        }

    @staticmethod
    def _extract_texture_features(image_gray: np.ndarray) -> dict:
        """텍스처 및 주파수 도메인 특성"""
        f_transform = np.fft.fft2(image_gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)

        h, w = image_gray.shape
        center_h, center_w = h // 2, w // 2
        mask = np.zeros((h, w))
        cv2.circle(mask, (center_w, center_h), min(h, w) // 4, 1, -1)
        high_freq_energy = np.sum(magnitude_spectrum * (1 - mask))

        grad_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.mean(np.sqrt(grad_x**2 + grad_y**2))

        local_std = ndimage.generic_filter(image_gray, np.std, size=5)
        local_contrast_std = np.std(local_std)

        return {
            "frequency_energy": high_freq_energy,
            "gradient_magnitude": gradient_magnitude,
            "local_contrast_std": local_contrast_std,
            "texture_complexity": gradient_magnitude * local_contrast_std,
        }

    @staticmethod
    def _estimate_jpeg_quality(gray_img: np.ndarray) -> float:
        """JPEG 품질 추정 (블록 아티팩트 기반)"""
        h, w = gray_img.shape
        block_diffs = []

        for i in range(8, h - 8, 8):
            diff = np.abs(gray_img[i - 1, :] - gray_img[i, :]).mean()
            block_diffs.append(diff)

        for j in range(8, w - 8, 8):
            diff = np.abs(gray_img[:, j - 1] - gray_img[:, j]).mean()
            block_diffs.append(diff)

        quality_estimate = 100 - np.mean(block_diffs) * 2
        return np.clip(quality_estimate, 0, 100)


def find_project_root() -> Path:
    """
    pyproject.toml 파일을 기준으로 루트 디렉토리를 찾는다.
    :return: Path: 프로젝트 루트 디렉토리 경로
    """

    current_path = Path().resolve()

    while current_path != current_path.parent:
        if (current_path / "pyproject.toml").exists():
            return current_path

        current_path = current_path.parent

    raise FileNotFoundError("프로젝트 루트 디렉토리를 찾을 수 없습니다.")


# 사용 예시
if __name__ == "__main__":
    ROOT_DIR = find_project_root()
    DATA_DIR = ROOT_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    TRAIN_DATA_DIR = RAW_DATA_DIR / "train"
    TEST_DATA_DIR = RAW_DATA_DIR / "test"

    TRAIN_META_FILE_NAME = "train.csv"
    TEST_META_FILE_NAME = "sample_submission.csv"

    # 데이터 로드
    # train_img_meta_df = pd.read_csv(RAW_DATA_DIR / TRAIN_META_FILE_NAME)
    print("Load metadata..")
    test_img_meta_df = pd.read_csv(RAW_DATA_DIR / TEST_META_FILE_NAME)

    # 특성 추출
    print("Init an extractor..")
    extractor = ImagePropertyExtractor(test_img_meta_df, TEST_DATA_DIR)
    print("Start extraction..")
    test_props_df = extractor.extract(n_jobs=10)

    # 결과 저장
    test_props_df.to_csv(RAW_DATA_DIR / "test_props.csv", index=False)
    print(f"추출 완료: {len(test_props_df)} 이미지")
