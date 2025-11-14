# core/sift_features.py
import cv2
import numpy as np
from typing import Tuple, List


def _to_gray(image: np.ndarray) -> np.ndarray:
    """
    Chuyển ảnh về grayscale nếu đang là BGR / BGRA.
    """
    if image is None:
        raise ValueError("Input image is None")

    if len(image.shape) == 2:
        # đã là gray
        return image

    if len(image.shape) == 3:
        h, w, c = image.shape
        if c == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif c == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    raise ValueError(f"Unsupported image shape for grayscale conversion: {image.shape}")


def create_sift(
        n_features: int = 0,
        contrast_threshold: float = 0.04,
        edge_threshold: int = 10,
        sigma: float = 1.6,
) -> cv2.SIFT:
    """
    Tạo đối tượng SIFT với tham số có thể chỉnh.

    n_features = 0  -> không giới hạn số keypoints (mặc định OpenCV).
    contrast_threshold, edge_threshold, sigma -> điều chỉnh độ nhạy.
    """
    sift = cv2.SIFT_create(
        nfeatures=n_features,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
        sigma=sigma,
    )
    return sift


def detect_and_describe(
        image: np.ndarray,
        n_features: int = 0,
        contrast_threshold: float = 0.04,
        edge_threshold: int = 10,
        sigma: float = 1.6,
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Phát hiện keypoints và mô tả bằng SIFT.

    Parameters
    ----------
    image : np.ndarray
        Ảnh BGR hoặc Gray (OpenCV).
    n_features : int
        Số keypoints tối đa (0 = không giới hạn).
    contrast_threshold : float
        Ngưỡng độ tương phản (thấp hơn -> phát hiện được nhiều keypoints hơn).
    edge_threshold : int
        Ngưỡng loại bỏ keypoint nằm trên biên có đáp ứng quá mạnh.
    sigma : float
        Độ lệch chuẩn Gaussian ban đầu.

    Returns
    -------
    keypoints : List[cv2.KeyPoint]
    descriptors : np.ndarray
        Mảng (N, 128) float32, mỗi hàng là 1 descriptor SIFT.
    """
    if image is None:
        raise ValueError("Input image is None")

    gray = _to_gray(image)

    sift = create_sift(
        n_features=n_features,
        contrast_threshold=contrast_threshold,
        edge_threshold=edge_threshold,
        sigma=sigma,
    )

    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # descriptors có thể là None nếu không phát hiện được keypoint nào
    if descriptors is None:
        descriptors = np.empty((0, 128), dtype=np.float32)

    return keypoints, descriptors
