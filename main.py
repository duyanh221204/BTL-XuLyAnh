"""
main.py

Chạy tool ghép ảnh panorama bằng đường dẫn ảnh.

Hai chế độ:
1) Ghép 1 panorama từ danh sách path:
   python main.py --images img1.jpg img2.jpg img3.jpg --output outputs/pano1.jpg

2) Ghép N panorama từ file cấu hình JSON:
   python main.py --config config.json --output_dir outputs/

   Trong đó config.json có dạng:

   {
     "sets": [
       {
         "name": "pano1",
         "images": ["data/p1_1.jpg", "data/p1_2.jpg", "data/p1_3.jpg"]
       },
       {
         "name": "pano2",
         "images": ["data/p2_1.jpg", "data/p2_2.jpg"]
       }
     ]
   }
"""

import argparse
import json
import os

import cv2

from core.stitching import stitch_from_paths


def run_single_pano(image_paths, output_path):
    """
    Ghép 1 panorama từ danh sách path.
    """
    print("[INFO] Ghép 1 panorama...")
    print("      Ảnh đầu vào:")
    for p in image_paths:
        print("       -", p)

    pano = stitch_from_paths(image_paths)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, pano)
    print(f"[OK] Đã lưu panorama: {output_path}")


def run_multi_from_config(config_path, output_dir):
    """
    Ghép nhiều panorama từ 1 file cấu hình JSON.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    sets = cfg.get("sets", [])
    if not sets:
        raise ValueError("config.json không có key 'sets' hoặc danh sách rỗng.")

    os.makedirs(output_dir, exist_ok=True)

    for idx, s in enumerate(sets, start=1):
        name = s.get("name") or f"pano_{idx}"
        image_paths = s.get("images", [])
        if len(image_paths) < 2:
            print(f"[WARN] Bộ '{name}' < 2 ảnh, bỏ qua.")
            continue

        print(f"[INFO] Đang ghép bộ '{name}' với {len(image_paths)} ảnh...")
        pano = stitch_from_paths(image_paths)

        out_path = os.path.join(output_dir, f"{name}.jpg")
        cv2.imwrite(out_path, pano)
        print(f"[OK]   -> Đã lưu: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tool ghép ảnh panorama dựa trên SIFT + Homography."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--images",
        nargs="+",
        help="Danh sách đường dẫn ảnh cho 1 panorama (theo thứ tự trái -> phải).",
    )
    group.add_argument(
        "--config",
        help="Đường dẫn tới file JSON mô tả nhiều bộ ảnh panorama.",
    )

    parser.add_argument(
        "--output",
        help="File output cho chế độ --images (mặc định: outputs/panorama.jpg).",
        default="outputs/panorama.jpg",
    )
    parser.add_argument(
        "--output_dir",
        help="Thư mục output cho chế độ --config (mặc định: outputs/).",
        default="outputs",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.images:
        run_single_pano(args.images, args.output)
    elif args.config:
        run_multi_from_config(args.config, args.output_dir)
