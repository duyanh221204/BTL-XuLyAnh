# test_feature_matching
import cv2
from core.sift_features import detect_and_describe
from core.matching import match_descriptors, draw_matches


img1_path = "C:\\Users\\HoaNgo\\Downloads\\2.png"
img2_path = "C:\\Users\\HoaNgo\\Downloads\\1.png"

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

if img1 is None or img2 is None:
    raise FileNotFoundError("Không đọc được ảnh, hãy kiểm tra lại đường dẫn.")

# 1. Detect + describe
kp1, desc1 = detect_and_describe(img1, n_features=0)
kp2, desc2 = detect_and_describe(img2, n_features=0)

print(f"[INFO] Ảnh 1: {len(kp1)} keypoints, descriptors shape = {desc1.shape}")
print(f"[INFO] Ảnh 2: {len(kp2)} keypoints, descriptors shape = {desc2.shape}")

# 2. Matching
good_matches1 = match_descriptors(desc1, desc2, ratio=0.6, method="bf")
good_matches2 = match_descriptors(desc1, desc2, ratio=0.6, method="flann")

print(f"[INFO] Số match tốt (sau Lowe's ratio test): {len(good_matches1)} (BF), {len(good_matches2)} (FLANN)")

# 3. Vẽ match để debug / chụp slide
vis = draw_matches(img1, kp1, img2, kp2, good_matches1, max_draw=50)
vis2 = draw_matches(img1, kp1, img2, kp2, good_matches2, max_draw=50)

out_path = "outputs/matches_debug_bf.jpg"
import os
os.makedirs("outputs", exist_ok=True)
cv2.imwrite(out_path, vis)
print(f"[INFO] Đã lưu ảnh match: {out_path}")

out_path2 = "outputs/matches_debug_flann.jpg"
cv2.imwrite(out_path2, vis2)
print(f"[INFO] Đã lưu ảnh match: {out_path2}")
