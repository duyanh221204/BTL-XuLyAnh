"""Xử lý hậu kỳ panorama"""
import cv2


def postprocess_panorama(pano):
    """Tăng độ tương phản (CLAHE) để tránh lệch màu"""
    print("  ✨ Tăng độ tương phản...")
    
    lab = cv2.cvtColor(pano, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE chỉ trên kênh L (luminance) - giữ nguyên màu sắc
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    print("[OK] Hoàn tất!")
    return result

