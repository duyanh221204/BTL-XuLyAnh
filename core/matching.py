import cv2
import numpy as np
from typing import List, Tuple


def create_matcher(method: str = "bf") -> cv2.DescriptorMatcher:
    """
    Tạo matcher cho SIFT (dùng norm L2).
    method = "bf"   -> Brute-Force
    method = "flann"-> FLANN-based matcher
    """
    method = method.lower()
    if method == "bf":
        # crossCheck = False để dùng được knnMatch + ratio test
        return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    elif method == "flann":
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
        search_params = dict(checks=50)
        return cv2.FlannBasedMatcher(index_params, search_params)
    else:
        raise ValueError(f"Unknown matcher method: {method}")


def match_descriptors(
        desc1: np.ndarray,
        desc2: np.ndarray,
        ratio: float,
        method: str,
) -> List[cv2.DMatch]:
    """
    Ghép descriptor SIFT giữa 2 ảnh bằng Lowe's ratio test.

    Parameters
    ----------
    desc1, desc2 : np.ndarray
        Descriptors của ảnh 1 và ảnh 2, shape (N, 128), (M, 128).
    ratio : float
        Ngưỡng Lowe's ratio test. Thường dùng 0.6 ~ 0.8.
    method : str
        "bf" hoặc "flann".

    Returns
    -------
    good_matches : List[cv2.DMatch]
        Danh sách match đã lọc.
    """
    if desc1 is None or desc2 is None:
        return []

    if len(desc1) == 0 or len(desc2) == 0:
        return []

    matcher = create_matcher(method)

    # knnMatch: với mỗi descriptor bên ảnh 1 -> tìm 2 "hàng xóm" gần nhất ở ảnh 2
    raw_matches = matcher.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for m, n in raw_matches:
        # Lowe's ratio test:
        # chỉ chấp nhận m nếu khoảng cách của m nhỏ hơn ratio * khoảng cách của n
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    # Sắp xếp theo khoảng cách tăng dần (match tốt trước)
    good_matches = sorted(good_matches, key=lambda x: x.distance)

    return good_matches


def draw_matches(
        img1: np.ndarray,
        kp1: List[cv2.KeyPoint],
        img2: np.ndarray,
        kp2: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
        max_draw: int = 50,
) -> np.ndarray:
    """
    Vẽ các match giữa 2 ảnh để debug / demo.

    Parameters
    ----------
    img1, img2 : np.ndarray
        Ảnh BGR.
    kp1, kp2 : List[cv2.KeyPoint]
        Keypoints tương ứng.
    matches : List[cv2.DMatch]
        Danh sách match đã lọc (vd: từ match_descriptors).
    max_draw : int
        Giới hạn số match vẽ cho dễ nhìn.

    Returns
    -------
    vis : np.ndarray
        Ảnh đã ghép cạnh nhau + vẽ line nối match.
    """
    if len(matches) == 0:
        raise ValueError("No matches to draw")

    matches_to_draw = matches[:max_draw]

    vis = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches_to_draw,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return vis


def keypoints_to_numpy(
        kp1: List[cv2.KeyPoint],
        kp2: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tiện ích: lấy ra tọa độ (x, y) của các cặp match để dùng cho RANSAC Homography.

    Returns
    -------
    pts1, pts2 : np.ndarray
        Mảng (N, 2) float32, pts1 là điểm ở ảnh 1, pts2 là điểm tương ứng ở ảnh 2.
    """
    if len(matches) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2
