import cv2
import customtkinter as ctk
from ultralytics import YOLO
from PIL import Image, ImageTk
import threading
import time

# Load YOLO once globally and warm it up
model = YOLO("best.pt")
model.predict(source=cv2.imread("warmup.jpg") if cv2.haveImageReader("warmup.jpg") else None, imgsz=640, conf=0.5, verbose=False)

# UI theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class AcneDetectionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("YOLOv11 - Acne Detection (Fast)")
        self.geometry("1200x850")

        # Display area
        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.pack(pady=20)

        # Buttons
        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.pack(pady=10)

        self.start_btn = ctk.CTkButton(self.button_frame, text="Start Detection", command=self.start_detection)
        self.start_btn.grid(row=0, column=0, padx=20)

        self.stop_btn = ctk.CTkButton(self.button_frame, text="Stop Detection", command=self.stop_detection, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=20)

        self.info_label = ctk.CTkLabel(self, text="", font=("Arial", 16))
        self.info_label.pack(pady=10)

        # Variables
        self.cap = None
        self.running = False
        self.last_frame = None
        self.detection_thread = None
        self.display_width = 900
        self.display_height = 600

    def start_detection(self):
        """Start webcam detection"""
        if self.running:
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.info_label.configure(text="Error: Cannot open webcam!")
            return

        self.running = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.info_label.configure(text="Starting detection...")

        # Start video thread
        self.detection_thread = threading.Thread(target=self.update_frame, daemon=True)
        self.detection_thread.start()

    def stop_detection(self):
        """Stop detection & show results"""
        if not self.running:
            return

        self.running = False
        time.sleep(0.3)

        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

        if self.last_frame is not None:
            results = model(self.last_frame, imgsz=640, conf=0.5)
            annotated = results[0].plot()
            boxes = results[0].boxes.xyxy
            num_acnes = len(boxes)

            # Save acne sizes to text file
            with open("acne_sizes.txt", "w") as f:
                f.write("Acne Detection Results\n")
                f.write("======================\n\n")
                for i, box in enumerate(boxes, start=1):
                    x1, y1, x2, y2 = box[:4]
                    width = int(x2 - x1)
                    height = int(y2 - y1)
                    f.write(f"Acne {i}: Width = {width}px, Height = {height}px\n")
                f.write(f"\nTotal acne(s) detected: {num_acnes}\n")

            cv2.imwrite("acne_result.jpg", annotated)

            # Show the final frame
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(annotated_rgb)
            img = img.resize((self.display_width, self.display_height))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.configure(image=imgtk)
            self.video_label.image = imgtk

            self.info_label.configure(
                text=f"Detection stopped. {num_acnes} acne(s) detected.\n"
                     f"Saved as 'acne_result.jpg' and 'acne_sizes.txt'."
            )
        else:
            self.info_label.configure(text="No frame captured.")

    def update_frame(self):
        """Continuously capture & display frames faster"""
        frame_skip = 2  # Run detection every Nth frame for speed
        frame_count = 0

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.last_frame = frame.copy()
            frame_count += 1

            if frame_count % frame_skip == 0:
                results = model(frame, imgsz=640, conf=0.5, verbose=False)
                annotated = results[0].plot()
            else:
                annotated = frame

            # Convert to RGB and show
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((self.display_width, self.display_height))
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.configure(image=imgtk)
            self.video_label.image = imgtk

        # Cleanup camera
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None


if __name__ == "__main__":
    app = AcneDetectionApp()
    app.mainloop()
