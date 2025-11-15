"""
GUI Application cho công cụ ghép ảnh Panorama
Sử dụng Tkinter để tạo giao diện người dùng thân thiện
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import threading
from typing import List, Optional

# Import các module core
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.stitching import stitch_sequence
from core.sift_features import detect_and_describe
from core.matching import match_descriptors, draw_matches


class PanoramaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Công cụ ghép ảnh Panorama - SIFT + Homography")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        
        # Dữ liệu
        self.image_paths: List[str] = []
        self.images: List[np.ndarray] = []
        self.panorama: Optional[np.ndarray] = None
        self.is_processing = False
        self.matching_images: List[np.ndarray] = []  # Lưu ảnh matching visualization
        
        # Tạo UI
        self.create_widgets()
        
    def create_widgets(self):
        """Tạo các widget chính của giao diện"""
        
        # ===== HEADER =====
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="🏞️ Công cụ Ghép ảnh Panorama",
            font=("Arial", 18, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title_label.pack(pady=15)
        
        # ===== MAIN CONTAINER =====
        main_container = tk.Frame(self.root, bg="#f0f0f0")
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ===== LEFT PANEL - Controls =====
        left_panel = tk.Frame(main_container, bg="white", width=350, relief=tk.RAISED, borderwidth=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5))
        left_panel.pack_propagate(False)
        
        # Nút chọn ảnh
        btn_frame = tk.Frame(left_panel, bg="white")
        btn_frame.pack(pady=15, padx=15, fill=tk.X)
        
        self.btn_select = tk.Button(
            btn_frame,
            text="📂 Chọn ảnh",
            command=self.select_images,
            font=("Arial", 12, "bold"),
            bg="#3498db",
            fg="white",
            relief=tk.FLAT,
            cursor="hand2",
            height=2
        )
        self.btn_select.pack(fill=tk.X)
        
        # Danh sách ảnh đã chọn
        list_frame = tk.LabelFrame(
            left_panel,
            text="Ảnh đã chọn",
            font=("Arial", 11, "bold"),
            bg="white",
            fg="#2c3e50"
        )
        list_frame.pack(pady=10, padx=15, fill=tk.BOTH, expand=True)
        
        # Scrollbar cho listbox
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.listbox_images = tk.Listbox(
            list_frame,
            font=("Arial", 9),
            yscrollcommand=scrollbar.set,
            selectmode=tk.SINGLE,
            bg="#ecf0f1"
        )
        self.listbox_images.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.config(command=self.listbox_images.yview)
        self.listbox_images.bind('<<ListboxSelect>>', self.on_image_select)
        
        # Nút xóa ảnh
        btn_remove = tk.Button(
            list_frame,
            text="🗑️ Xóa ảnh đã chọn",
            command=self.remove_selected_image,
            font=("Arial", 9),
            bg="#e74c3c",
            fg="white",
            relief=tk.FLAT,
            cursor="hand2"
        )
        btn_remove.pack(pady=5, padx=5, fill=tk.X)
        
        # Tham số
        param_frame = tk.LabelFrame(
            left_panel,
            text="Tham số ghép ảnh",
            font=("Arial", 11, "bold"),
            bg="white",
            fg="#2c3e50"
        )
        param_frame.pack(pady=10, padx=15, fill=tk.X)
        
        # Ratio threshold
        tk.Label(
            param_frame,
            text="Ratio Threshold:",
            font=("Arial", 9),
            bg="white"
        ).grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        
        self.ratio_var = tk.DoubleVar(value=0.6)
        self.ratio_scale = tk.Scale(
            param_frame,
            from_=0.4,
            to=0.9,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.ratio_var,
            bg="white",
            length=150
        )
        self.ratio_scale.grid(row=0, column=1, padx=10, pady=5)
        
        # RANSAC threshold
        tk.Label(
            param_frame,
            text="RANSAC Threshold:",
            font=("Arial", 9),
            bg="white"
        ).grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        
        self.ransac_var = tk.DoubleVar(value=4.0)
        self.ransac_scale = tk.Scale(
            param_frame,
            from_=1.0,
            to=10.0,
            resolution=0.5,
            orient=tk.HORIZONTAL,
            variable=self.ransac_var,
            bg="white",
            length=150
        )
        self.ransac_scale.grid(row=1, column=1, padx=10, pady=5)
        
        # Matcher method
        tk.Label(
            param_frame,
            text="Phương pháp match:",
            font=("Arial", 9),
            bg="white"
        ).grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        
        self.matcher_var = tk.StringVar(value="bf")
        matcher_combo = ttk.Combobox(
            param_frame,
            textvariable=self.matcher_var,
            values=["bf", "flann"],
            state="readonly",
            width=15
        )
        matcher_combo.grid(row=2, column=1, padx=10, pady=5)
        
        # Info label
        info_label = tk.Label(
            param_frame,
            text="✨ Tự động: ảnh giữa làm chuẩn,\ntăng độ tương phản (tránh lệch màu)",
            font=("Arial", 8),
            bg="white",
            fg="#7f8c8d",
            justify=tk.LEFT
        )
        info_label.grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=10, pady=(10, 5))
        
        # Nút ghép ảnh
        self.btn_stitch = tk.Button(
            left_panel,
            text="✨ Ghép ảnh Panorama",
            command=self.stitch_panorama,
            font=("Arial", 12, "bold"),
            bg="#27ae60",
            fg="white",
            relief=tk.FLAT,
            cursor="hand2",
            height=2,
            state=tk.DISABLED
        )
        self.btn_stitch.pack(pady=15, padx=15, fill=tk.X)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            left_panel,
            variable=self.progress_var,
            maximum=100,
            mode='indeterminate'
        )
        self.progress_bar.pack(pady=5, padx=15, fill=tk.X)
        
        self.status_label = tk.Label(
            left_panel,
            text="Sẵn sàng",
            font=("Arial", 9),
            bg="white",
            fg="#7f8c8d"
        )
        self.status_label.pack(pady=5)
        
        # Nút lưu kết quả
        self.btn_save = tk.Button(
            left_panel,
            text="💾 Lưu kết quả",
            command=self.save_result,
            font=("Arial", 11, "bold"),
            bg="#9b59b6",
            fg="white",
            relief=tk.FLAT,
            cursor="hand2",
            state=tk.DISABLED
        )
        self.btn_save.pack(pady=10, padx=15, fill=tk.X)
        
        # ===== RIGHT PANEL - Preview =====
        right_panel = tk.Frame(main_container, bg="white", relief=tk.RAISED, borderwidth=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Tabs cho preview
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 1: Preview ảnh đầu vào
        self.input_frame = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.input_frame, text="📷 Ảnh đầu vào")
        
        self.input_canvas = tk.Canvas(self.input_frame, bg="#ecf0f1")
        self.input_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 2: Preview kết quả
        self.output_frame = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.output_frame, text="🏞️ Kết quả Panorama")
        
        self.output_canvas = tk.Canvas(self.output_frame, bg="#ecf0f1")
        self.output_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 3: Feature Matching
        self.matching_frame = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.matching_frame, text="🔗 Feature Matching")
        
        # Frame cho controls
        matching_control = tk.Frame(self.matching_frame, bg="white", height=50)
        matching_control.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(
            matching_control,
            text="Chọn cặp ảnh:",
            font=("Arial", 10, "bold"),
            bg="white"
        ).pack(side=tk.LEFT, padx=5)
        
        self.matching_combo = ttk.Combobox(
            matching_control,
            state="readonly",
            font=("Arial", 10),
            width=30
        )
        self.matching_combo.pack(side=tk.LEFT, padx=5)
        self.matching_combo.bind("<<ComboboxSelected>>", self.on_matching_select)
        
        self.matching_info_label = tk.Label(
            matching_control,
            text="",
            font=("Arial", 9),
            bg="white",
            fg="#7f8c8d"
        )
        self.matching_info_label.pack(side=tk.LEFT, padx=10)
        
        # Canvas để hiển thị matching
        self.matching_canvas = tk.Canvas(self.matching_frame, bg="#ecf0f1")
        self.matching_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Bind resize event
        self.root.bind('<Configure>', self.on_window_resize)
        
    def select_images(self):
        """Chọn nhiều ảnh từ file dialog"""
        filetypes = (
            ("Ảnh", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("Tất cả", "*.*")
        )
        
        paths = filedialog.askopenfilenames(
            title="Chọn ảnh để ghép Panorama (theo thứ tự trái -> phải)",
            filetypes=filetypes
        )
        
        if paths:
            self.image_paths = list(paths)
            self.load_images()
            self.update_image_list()
            self.generate_matching_visualization()
            
            if len(self.image_paths) >= 2:
                self.btn_stitch.config(state=tk.NORMAL)
            else:
                messagebox.showwarning(
                    "Cảnh báo",
                    "Cần ít nhất 2 ảnh để ghép Panorama!"
                )
    
    def load_images(self):
        """Đọc các ảnh đã chọn"""
        self.images = []
        for path in self.image_paths:
            img = cv2.imread(path)
            if img is not None:
                self.images.append(img)
            else:
                messagebox.showerror("Lỗi", f"Không đọc được ảnh: {path}")
    
    def update_image_list(self):
        """Cập nhật danh sách ảnh trong listbox"""
        self.listbox_images.delete(0, tk.END)
        for i, path in enumerate(self.image_paths, 1):
            filename = os.path.basename(path)
            self.listbox_images.insert(tk.END, f"{i}. {filename}")
        
        # Tự động chọn ảnh đầu tiên
        if self.image_paths:
            self.listbox_images.select_set(0)
            self.on_image_select(None)
    
    def on_image_select(self, event):
        """Hiển thị preview khi chọn ảnh trong list"""
        selection = self.listbox_images.curselection()
        if selection and self.images:
            idx = selection[0]
            if idx < len(self.images):
                self.display_image(self.images[idx], self.input_canvas)
    
    def remove_selected_image(self):
        """Xóa ảnh đã chọn khỏi danh sách"""
        selection = self.listbox_images.curselection()
        if selection:
            idx = selection[0]
            del self.image_paths[idx]
            del self.images[idx]
            self.update_image_list()
            
            if len(self.image_paths) < 2:
                self.btn_stitch.config(state=tk.DISABLED)
    
    def stitch_panorama(self):
        """Ghép ảnh panorama trong thread riêng"""
        if len(self.images) < 2:
            messagebox.showwarning("Cảnh báo", "Cần ít nhất 2 ảnh!")
            return
        
        if self.is_processing:
            return
        
        # Disable buttons
        self.btn_stitch.config(state=tk.DISABLED)
        self.btn_select.config(state=tk.DISABLED)
        self.btn_save.config(state=tk.DISABLED)
        self.is_processing = True
        
        # Start progress bar
        self.progress_bar.start(10)
        self.status_label.config(text="Đang ghép ảnh...", fg="#e67e22")
        
        # Run in thread
        thread = threading.Thread(target=self._stitch_worker)
        thread.daemon = True
        thread.start()
    
    def _stitch_worker(self):
        """Worker thread để ghép ảnh"""
        try:
            ratio = self.ratio_var.get()
            ransac_thresh = self.ransac_var.get()
            matcher_method = self.matcher_var.get()
            
            # Ghép ảnh (tự động center-based + post-processing)
            self.panorama = stitch_sequence(
                self.images,
                ratio=ratio,
                ransac_thresh=ransac_thresh,
                matcher_method=matcher_method
            )
            
            # Update UI in main thread
            self.root.after(0, self._stitch_complete_success)
            
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda: self._stitch_complete_error(error_msg))
    
    def _stitch_complete_success(self):
        """Callback khi ghép ảnh thành công"""
        self.progress_bar.stop()
        self.status_label.config(text="✓ Ghép ảnh thành công!", fg="#27ae60")
        
        # Hiển thị kết quả
        self.notebook.select(1)  # Chuyển sang tab kết quả
        self.display_image(self.panorama, self.output_canvas)
        
        # Enable buttons
        self.btn_stitch.config(state=tk.NORMAL)
        self.btn_select.config(state=tk.NORMAL)
        self.btn_save.config(state=tk.NORMAL)
        self.is_processing = False
        
        messagebox.showinfo("Thành công", "Đã ghép ảnh Panorama thành công!")
    
    def _stitch_complete_error(self, error_msg):
        """Callback khi ghép ảnh lỗi"""
        self.progress_bar.stop()
        self.status_label.config(text="✗ Lỗi khi ghép ảnh", fg="#e74c3c")
        
        # Enable buttons
        self.btn_stitch.config(state=tk.NORMAL)
        self.btn_select.config(state=tk.NORMAL)
        self.is_processing = False
        
        messagebox.showerror("Lỗi", f"Không thể ghép ảnh:\n{error_msg}")
    
    def display_image(self, img: np.ndarray, canvas: tk.Canvas):
        """Hiển thị ảnh OpenCV trên canvas"""
        if img is None:
            return
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to fit canvas
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas chưa được render, dùng default size
            canvas_width = 800
            canvas_height = 600
        
        h, w = img_rgb.shape[:2]
        scale = min(canvas_width / w, canvas_height / h) * 0.95
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Convert to PIL Image
        img_pil = Image.fromarray(img_resized)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        # Store reference to prevent garbage collection
        canvas.image = img_tk
        
        # Clear canvas và vẽ ảnh
        canvas.delete("all")
        x = (canvas_width - new_w) // 2
        y = (canvas_height - new_h) // 2
        canvas.create_image(x, y, anchor=tk.NW, image=img_tk)
    
    def on_window_resize(self, event):
        """Callback khi resize cửa sổ"""
        # Refresh image displays
        if hasattr(self, 'listbox_images'):
            selection = self.listbox_images.curselection()
            if selection and self.images:
                idx = selection[0]
                if idx < len(self.images):
                    self.root.after(100, lambda: self.display_image(self.images[idx], self.input_canvas))
        
        if self.panorama is not None:
            self.root.after(100, lambda: self.display_image(self.panorama, self.output_canvas))
        
        # Refresh matching display
        if hasattr(self, 'matching_combo') and self.matching_images:
            selection = self.matching_combo.current()
            if selection >= 0 and selection < len(self.matching_images):
                self.root.after(100, lambda: self.display_image(self.matching_images[selection], self.matching_canvas))
    
    def generate_matching_visualization(self):
        """Tạo ảnh visualization cho feature matching giữa các cặp ảnh"""
        if len(self.images) < 2:
            return
        
        self.matching_images = []
        self.matching_combo['values'] = []
        combo_items = []
        
        # Hiển thị progress
        self.status_label.config(text="⏳ Đang tính toán feature matching...", fg="#f39c12")
        self.root.update()
        
        try:
            # Tính toán matches cho mỗi cặp ảnh liên tiếp
            for i in range(len(self.images) - 1):
                img1 = self.images[i]
                img2 = self.images[i + 1]
                
                # Detect features
                kp1, desc1 = detect_and_describe(img1)
                kp2, desc2 = detect_and_describe(img2)
                
                # Match descriptors (dùng tham số từ GUI)
                ratio = self.ratio_var.get()
                method = self.matcher_var.get()
                matches = match_descriptors(desc1, desc2, ratio, method)
                
                # Vẽ matches (giới hạn 100 matches để dễ nhìn)
                if len(matches) > 0:
                    match_img = draw_matches(img1, kp1, img2, kp2, matches, max_draw=100)
                    self.matching_images.append(match_img)
                    
                    # Thêm vào combobox
                    filename1 = os.path.basename(self.image_paths[i])
                    filename2 = os.path.basename(self.image_paths[i + 1])
                    combo_items.append(f"Cặp {i+1}: {filename1} ↔ {filename2}")
                else:
                    # Nếu không có match, tạo ảnh thông báo
                    h1, w1 = img1.shape[:2]
                    h2, w2 = img2.shape[:2]
                    blank = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
                    cv2.putText(blank, "No matches found", (w1//2, max(h1, h2)//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    self.matching_images.append(blank)
                    
                    filename1 = os.path.basename(self.image_paths[i])
                    filename2 = os.path.basename(self.image_paths[i + 1])
                    combo_items.append(f"Cặp {i+1}: {filename1} ↔ {filename2} (0 matches)")
            
            # Cập nhật combobox
            self.matching_combo['values'] = combo_items
            if combo_items:
                self.matching_combo.current(0)
                self.display_image(self.matching_images[0], self.matching_canvas)
                
                # Hiển thị thông tin
                kp1, desc1 = detect_and_describe(self.images[0])
                kp2, desc2 = detect_and_describe(self.images[1])
                ratio = self.ratio_var.get()
                method = self.matcher_var.get()
                matches = match_descriptors(desc1, desc2, ratio, method)
                self.matching_info_label.config(text=f"✓ {len(matches)} matches")
            
            self.status_label.config(text="✓ Feature matching hoàn tất!", fg="#27ae60")
        
        except Exception as e:
            self.status_label.config(text=f"✗ Lỗi: {str(e)}", fg="#e74c3c")
            print(f"Error generating matching: {e}")
    
    def on_matching_select(self, event):
        """Callback khi chọn cặp ảnh trong combobox"""
        selection = self.matching_combo.current()
        if selection >= 0 and selection < len(self.matching_images):
            self.display_image(self.matching_images[selection], self.matching_canvas)
            
            # Tính toán lại số matches cho cặp này
            try:
                img1 = self.images[selection]
                img2 = self.images[selection + 1]
                
                kp1, desc1 = detect_and_describe(img1)
                kp2, desc2 = detect_and_describe(img2)
                
                ratio = self.ratio_var.get()
                method = self.matcher_var.get()
                matches = match_descriptors(desc1, desc2, ratio, method)
                
                self.matching_info_label.config(text=f"✓ {len(matches)} matches | {len(kp1)} + {len(kp2)} keypoints")
            except Exception as e:
                self.matching_info_label.config(text="")
    
    def save_result(self):
        """Lưu kết quả panorama"""
        if self.panorama is None:
            messagebox.showwarning("Cảnh báo", "Chưa có kết quả để lưu!")
            return
        
        filetypes = (
            ("JPEG", "*.jpg"),
            ("PNG", "*.png"),
            ("Tất cả", "*.*")
        )
        
        filepath = filedialog.asksaveasfilename(
            title="Lưu Panorama",
            defaultextension=".jpg",
            filetypes=filetypes,
            initialfile="panorama.jpg"
        )
        
        if filepath:
            try:
                cv2.imwrite(filepath, self.panorama)
                messagebox.showinfo("Thành công", f"Đã lưu ảnh:\n{filepath}")
                self.status_label.config(text=f"✓ Đã lưu: {os.path.basename(filepath)}", fg="#27ae60")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể lưu ảnh:\n{str(e)}")


def main():
    """Entry point của ứng dụng"""
    root = tk.Tk()
    app = PanoramaApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

