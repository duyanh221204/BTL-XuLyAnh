"""
Script chạy GUI của công cụ ghép ảnh Panorama

Sử dụng:
    python run_gui.py
"""

from gui.panorama_app import main

if __name__ == "__main__":
    print("=" * 60)
    print("    CÔNG CỤ GHÉP ẢNH PANORAMA")
    print("    Sử dụng SIFT + Homography")
    print("=" * 60)
    print()
    print("Đang khởi động giao diện...")
    print()
    
    main()

