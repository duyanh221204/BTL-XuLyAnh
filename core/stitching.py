# core/stitching.py
"""
Các hàm ghép ảnh panorama dựa trên SIFT + Homography.

Dùng chung với:
- core.sift_features.detect_and_describe
- core.matching.match_descriptors, keypoints_to_numpy
"""

import cv2
import numpy as np
from typing import List, Sequence

from core.sift_features import detect_and_describe
from core.matching import match_descriptors, keypoints_to_numpy


def estimate_homography(
    img1: np.ndarray,
    img2: np.ndarray,
    ratio: float = 0.6,
    ransac_thresh: float = 4.0,
    matcher_method: str = "bf",
):
    """
    Tính ma trận Homography H để biến đổi img2 sang hệ tọa độ img1.

    Trả về
    -------
    H : np.ndarray shape (3, 3)
    matches : list[cv2.DMatch]
    mask : np.ndarray (N, 1) 0/1 (inlier mask từ RANSAC)
    """
    # 1. SIFT
    kp1, desc1 = detect_and_describe(img1)
    kp2, desc2 = detect_and_describe(img2)

    # 2. Matching
    matches = match_descriptors(desc1, desc2, ratio=ratio, method=matcher_method)

    # 3. Lấy tọa độ
    pts1, pts2 = keypoints_to_numpy(kp1, kp2, matches)

    if len(pts1) < 4:
        raise RuntimeError(
            f"Không đủ điểm match để tính Homography (cần >= 4, hiện có {len(pts1)})."
        )

    # 4. RANSAC để tìm H: điểm ảnh 2 (pts2) -> ảnh 1 (pts1)
    H, mask = cv2.findHomography(
        pts2, pts1, cv2.RANSAC, ransacReprojThreshold=ransac_thresh
    )
    if H is None:
        raise RuntimeError("Không tính được Homography (cv2.findHomography trả về None).")

    return H, matches, mask


def stitch_pair(
    img_left: np.ndarray,
    img_right: np.ndarray,
    ratio: float = 0.6,
    ransac_thresh: float = 4.0,
    matcher_method: str = "bf",
) -> np.ndarray:
    """
    Ghép 2 ảnh (img_left ở bên trái, img_right ở bên phải) thành 1 panorama đơn.
    """
    H, matches, mask = estimate_homography(
        img_left, img_right,
        ratio=ratio,
        ransac_thresh=ransac_thresh,
        matcher_method=matcher_method,
    )

    h1, w1 = img_left.shape[:2]
    h2, w2 = img_right.shape[:2]

    # 4 góc ảnh right sau khi warp sang left
    corners_right = np.float32([
        [0, 0],
        [w2, 0],
        [w2, h2],
        [0, h2],
    ]).reshape(-1, 1, 2)
    warped_corners_right = cv2.perspectiveTransform(corners_right, H)

    # 4 góc ảnh left
    corners_left = np.float32([
        [0, 0],
        [w1, 0],
        [w1, h1],
        [0, h1],
    ]).reshape(-1, 1, 2)

    all_corners = np.concatenate([corners_left, warped_corners_right], axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Dịch để không bị toạ độ âm
    translation = [-xmin, -ymin]
    T = np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1],
    ])

    # Warp ảnh right
    panorama = cv2.warpPerspective(
        img_right,
        T @ H,
        (xmax - xmin, ymax - ymin),
    )

    # Dán ảnh left vào
    panorama[
        translation[1] : translation[1] + h1,
        translation[0] : translation[0] + w1,
    ] = img_left

    return panorama


def stitch_sequence(
    images: Sequence[np.ndarray],
    ratio: float = 0.6,
    ransac_thresh: float = 4.0,
    matcher_method: str = "bf",
) -> np.ndarray:
    """
    Ghép N ảnh [img0, img1, img2, ...] thành 1 panorama.
    Giả định các ảnh đã được sắp xếp theo thứ tự trái -> phải.
    """
    if not images:
        raise ValueError("Danh sách images trống.")
    if len(images) == 1:
        return images[0]

    pano = images[0]
    for img in images[1:]:
        pano = stitch_pair(
            pano,
            img,
            ratio=ratio,
            ransac_thresh=ransac_thresh,
            matcher_method=matcher_method,
        )
    return pano


def stitch_from_paths(
    image_paths: Sequence[str],
    ratio: float = 0.6,
    ransac_thresh: float = 4.0,
    matcher_method: str = "bf",
) -> np.ndarray:
    """
    Ghép 1 panorama từ danh sách đường dẫn ảnh.

    Ví dụ:
        pano = stitch_from_paths([
            "data/pano1_1.jpg",
            "data/pano1_2.jpg",
            "data/pano1_3.jpg",
        ])
    """
    if not image_paths:
        raise ValueError("Danh sách đường dẫn ảnh trống.")

    images: List[np.ndarray] = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(f"Không đọc được ảnh: {p}")
        images.append(img)

    return stitch_sequence(
        images,
        ratio=ratio,
        ransac_thresh=ransac_thresh,
        matcher_method=matcher_method,
    )
