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
    Ghép N ảnh thành panorama. Tự động:
    - Center-based stitching (ảnh giữa làm chuẩn)
    - Auto-resize ảnh lớn
    - Tăng độ tương phản (tránh lệch màu)
    """
    if not images:
        raise ValueError("Danh sách images trống.")
    if len(images) == 1:
        return images[0]

    # Resize ảnh nếu quá lớn
    MAX_DIM = 1500
    resized_images = []
    
    for img in images:
        h, w = img.shape[:2]
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            resized_images.append(cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA))
            print(f"[INFO] Resize ảnh: {w}x{h} -> {new_w}x{new_h}")
        else:
            resized_images.append(img)
    
    images = resized_images

    # Ghép theo center-based
    if len(images) >= 3:
        print(f"[INFO] Ghép {len(images)} ảnh (ảnh giữa làm chuẩn)...")
        pano = _stitch_center_based(images, ratio, ransac_thresh, matcher_method)
    else:
        print(f"[INFO] Ghép {len(images)} ảnh...")
        pano = _stitch_two_images(images, ratio, ransac_thresh, matcher_method)
    
    # Xử lý hậu kỳ
    print("[INFO] Xử lý hậu kỳ...")
    pano = _postprocess(pano)
    
    return pano


def _stitch_two_images(
    images: List[np.ndarray],
    ratio: float,
    ransac_thresh: float,
    matcher_method: str,
) -> np.ndarray:
    """Ghép 1-2 ảnh đơn giản"""
    if len(images) == 1:
        return images[0]
    
    try:
        return stitch_pair(images[0], images[1], ratio, ransac_thresh, matcher_method)
    except RuntimeError:
        print("[WARN] Thử lại với tham số dễ hơn...")
        return stitch_pair(images[0], images[1], 0.8, ransac_thresh * 1.5, matcher_method)


def _stitch_center_based(
    images: List[np.ndarray],
    ratio: float,
    ransac_thresh: float,
    matcher_method: str,
) -> np.ndarray:
    """
    True Center-based stitching: Ảnh giữa GIỮ NGUYÊN hoàn toàn,
    warp TẤT CẢ ảnh khác về hệ tọa độ ảnh giữa.
    """
    n = len(images)
    center_idx = n // 2
    
    print(f"  → Ảnh #{center_idx + 1} làm chuẩn (GIỮ NGUYÊN)")
    
    # 1. Tính tất cả các Homography về ảnh chuẩn
    homographies = _compute_homographies_to_center(
        images, center_idx, ratio, ransac_thresh, matcher_method
    )
    
    # 2. Tính bounding box chứa tất cả ảnh
    all_corners = []
    for i, img in enumerate(images):
        corners = _get_image_corners(img)
        if i == center_idx:
            # Ảnh chuẩn: không biến đổi
            warped_corners = corners
        else:
            # Warp qua Homography
            warped_corners = cv2.perspectiveTransform(corners, homographies[i])
        all_corners.append(warped_corners)
    
    all_corners = np.concatenate(all_corners, axis=0)
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    # 3. Ma trận translation để tránh tọa độ âm
    translation = [-xmin, -ymin]
    T = np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1],
    ])
    
    print(f"  → Canvas: {xmax - xmin} x {ymax - ymin}")
    
    # 4. Warp tất cả ảnh vào canvas chung
    canvas_shape = (ymax - ymin, xmax - xmin, 3)
    panorama = _blend_images_to_canvas(
        images, homographies, T, canvas_shape, center_idx
    )
    
    return panorama


def _compute_homographies_to_center(
    images: List[np.ndarray],
    center_idx: int,
    ratio: float,
    ransac_thresh: float,
    matcher_method: str,
) -> List[np.ndarray]:
    """
    Tính Homography từ mỗi ảnh về ảnh chuẩn (center).
    
    Returns
    -------
    homographies : List[np.ndarray]
        Ma trận H cho mỗi ảnh. homographies[center_idx] = Identity.
    """
    n = len(images)
    homographies = [None] * n
    homographies[center_idx] = np.eye(3)  # Ảnh chuẩn: Identity matrix
    
    print("  → Tính Homography cho các ảnh bên trái...")
    # Tính H cho các ảnh bên trái
    H_accum = np.eye(3)
    for i in range(center_idx - 1, -1, -1):
        adaptive_ratio = min(ratio + abs(i - center_idx) * 0.05, 0.85)
        try:
            # H: từ images[i] sang images[i+1]
            H, _, _ = estimate_homography(
                images[i+1], images[i],
                adaptive_ratio, ransac_thresh, matcher_method
            )
            H_accum = H_accum @ H  # Tích lũy: từ images[i] về center
            homographies[i] = H_accum.copy()
            print(f"    ✓ Ảnh {i+1} → Center")
        except RuntimeError as e:
            print(f"    ⚠ Ảnh {i+1}: Thử lại với tham số dễ hơn...")
            H, _, _ = estimate_homography(
                images[i+1], images[i],
                0.8, ransac_thresh * 1.5, matcher_method
            )
            H_accum = H_accum @ H
            homographies[i] = H_accum.copy()
    
    print("  → Tính Homography cho các ảnh bên phải...")
    # Tính H cho các ảnh bên phải
    H_accum = np.eye(3)
    for i in range(center_idx + 1, n):
        adaptive_ratio = min(ratio + abs(i - center_idx) * 0.05, 0.85)
        try:
            # H: từ images[i] sang images[i-1]
            H, _, _ = estimate_homography(
                images[i-1], images[i],
                adaptive_ratio, ransac_thresh, matcher_method
            )
            H_accum = H_accum @ H  # Tích lũy: từ images[i] về center
            homographies[i] = H_accum.copy()
            print(f"    ✓ Ảnh {i+1} → Center")
        except RuntimeError as e:
            print(f"    ⚠ Ảnh {i+1}: Thử lại với tham số dễ hơn...")
            H, _, _ = estimate_homography(
                images[i-1], images[i],
                0.8, ransac_thresh * 1.5, matcher_method
            )
            H_accum = H_accum @ H
            homographies[i] = H_accum.copy()
    
    return homographies


def _get_image_corners(img: np.ndarray) -> np.ndarray:
    """Lấy 4 góc của ảnh dưới dạng array (4, 1, 2)"""
    h, w = img.shape[:2]
    return np.float32([
        [0, 0], [w, 0], [w, h], [0, h]
    ]).reshape(-1, 1, 2)


def _blend_images_to_canvas(
    images: List[np.ndarray],
    homographies: List[np.ndarray],
    T: np.ndarray,
    canvas_shape: tuple,
    center_idx: int,
) -> np.ndarray:
    """
    Warp và blend tất cả ảnh vào canvas chung.
    Ảnh chuẩn (center) chỉ được dịch chuyển, KHÔNG bị warp.
    """
    canvas = np.zeros(canvas_shape, dtype=np.uint8)
    canvas_h, canvas_w = canvas_shape[:2]
    
    print("  → Blending ảnh vào canvas...")
    
    # Warp tất cả ảnh (trừ ảnh chuẩn) vào canvas
    for i, img in enumerate(images):
        if i == center_idx:
            continue  # Xử lý ảnh chuẩn sau cùng
        
        if i < center_idx:
            print(f"    → Warp ảnh {i+1} (bên trái)")
        else:
            print(f"    → Warp ảnh {i+1} (bên phải)")
        
        # Ma trận cuối cùng: Translation + Homography
        H_final = T @ homographies[i]
        
        # Warp ảnh
        warped = cv2.warpPerspective(img, H_final, (canvas_w, canvas_h))
        
        # Blend (simple: overwrite non-black pixels)
        mask = (warped > 0).any(axis=2)
        canvas[mask] = warped[mask]
    
    # Dán ảnh chuẩn (GIỮ NGUYÊN chất lượng)
    print(f"    → Dán ảnh {center_idx+1} (chuẩn - GIỮ NGUYÊN)")
    center_img = images[center_idx]
    h_center, w_center = center_img.shape[:2]
    
    # Chỉ áp dụng translation, KHÔNG warp
    tx, ty = int(T[0, 2]), int(T[1, 2])
    
    # Dán ảnh chuẩn vào canvas
    canvas[ty : ty + h_center, tx : tx + w_center] = center_img
    
    return canvas


def _postprocess(pano: np.ndarray) -> np.ndarray:
    """Xử lý hậu kỳ: tăng độ tương phản"""
    try:
        from core.image_processing import postprocess_panorama
        return postprocess_panorama(pano)
    except Exception as e:
        print(f"[WARN] Bỏ qua xử lý hậu kỳ: {e}")
        return pano
