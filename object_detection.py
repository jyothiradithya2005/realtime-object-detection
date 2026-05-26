import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
from collections import Counter
import threading
import time

# ── Settings ──────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

GREEN = "#00C851"
DARK  = "#181818"

# ── Load YOLO ─────────────────────────────────────────────────────────────────
model = YOLO("yolov8m.pt")

# ── Main App ──────────────────────────────────────────────────────────────────
class SmartDetectionApp:

    def __init__(self):

        self.root = ctk.CTk()
        self.root.geometry("1500x850")
        self.root.title("Smart Detection System")
        self.root.configure(fg_color=DARK)

        self.camera_running = False
        self.mode           = "person"
        self.start_time     = time.time()
        self.camera         = None

        # Person counting mode: "live" = resets when no one, "cumulative" = keeps increasing
        self.person_count_mode = "live"

        # Cumulative count — total unique appearances over the session
        self.cumulative_count  = 0
        self.last_frame_count  = 0   # how many people were visible last frame

        # ── Left side: camera ─────────────────────────────────────────────
        self.left_frame = ctk.CTkFrame(self.root, fg_color="#101010", corner_radius=15)
        self.left_frame.pack(side="left", fill="both", expand=True, padx=15, pady=15)

        self.title_label = ctk.CTkLabel(
            self.left_frame,
            text="SMART DETECTION SYSTEM",
            font=("Arial", 28, "bold"),
            text_color=GREEN
        )
        self.title_label.pack(pady=15)

        self.video_label = ctk.CTkLabel(self.left_frame, text="")
        self.video_label.pack(padx=10, pady=10)

        # ── Right side: controls ──────────────────────────────────────────
        self.right_frame = ctk.CTkFrame(self.root, width=320, fg_color="#1E1E1E", corner_radius=15)
        self.right_frame.pack(side="right", fill="y", padx=15, pady=15)

        ctk.CTkLabel(
            self.right_frame,
            text="DETECTION SETTINGS",
            font=("Arial", 22, "bold"),
            text_color=GREEN
        ).pack(pady=20)

        # Detection mode (person / object)
        self.mode_selector = ctk.CTkOptionMenu(
            self.right_frame,
            values=["person", "object"],
            command=self.change_mode,
            width=220, height=40,
            fg_color=GREEN, button_color="#009944"
        )
        self.mode_selector.pack(pady=10)

        # ── Person counting mode selector (shown only in person mode) ─────
        self.count_mode_label = ctk.CTkLabel(
            self.right_frame,
            text="Person Count Mode:",
            font=("Arial", 13),
            text_color="gray"
        )
        self.count_mode_label.pack(pady=(15, 2))

        self.count_mode_selector = ctk.CTkSegmentedButton(
            self.right_frame,
            values=["Live Count", "Cumulative"],
            command=self.change_count_mode,
            width=220,
            font=("Arial", 13, "bold"),
            selected_color=GREEN,
            selected_hover_color="#009944"
        )
        self.count_mode_selector.set("Live Count")
        self.count_mode_selector.pack(pady=5)

        # Small description under the toggle
        self.count_mode_desc = ctk.CTkLabel(
            self.right_frame,
            text="Resets when no person visible",
            font=("Arial", 11),
            text_color="#777777"
        )
        self.count_mode_desc.pack(pady=(2, 10))

        # Reset cumulative button
        self.reset_button = ctk.CTkButton(
            self.right_frame,
            text="↺  Reset Count",
            command=self.reset_cumulative,
            width=220, height=35,
            font=("Arial", 13),
            fg_color="#333333",
            hover_color="#444444"
        )
        self.reset_button.pack(pady=5)

        # Start button
        ctk.CTkButton(
            self.right_frame,
            text="▶ START CAMERA",
            command=self.start_camera,
            width=220, height=45,
            font=("Arial", 16, "bold"),
            fg_color=GREEN, hover_color="#009944"
        ).pack(pady=15)

        # Stop button
        ctk.CTkButton(
            self.right_frame,
            text="■ STOP CAMERA",
            command=self.stop_camera,
            width=220, height=45,
            font=("Arial", 16, "bold"),
            fg_color="#D32F2F", hover_color="#B71C1C"
        ).pack(pady=5)

        # Session timer
        self.timer_label = ctk.CTkLabel(
            self.right_frame,
            text="Session: 00:00",
            font=("Arial", 18),
            text_color="white"
        )
        self.timer_label.pack(pady=25)

        # Crowd status
        self.status_label = ctk.CTkLabel(
            self.right_frame,
            text="Status: Waiting...",
            font=("Arial", 18, "bold"),
            text_color="orange"
        )
        self.status_label.pack(pady=10)

        # Live counts title
        ctk.CTkLabel(
            self.right_frame,
            text="LIVE COUNTS",
            font=("Arial", 22, "bold"),
            text_color=GREEN
        ).pack(pady=10)

        # Counts textbox
        self.count_box = ctk.CTkTextbox(
            self.right_frame,
            width=260, height=200,
            font=("Consolas", 16),
            fg_color="#111111"
        )
        self.count_box.pack(pady=10)

        # Exit button
        ctk.CTkButton(
            self.right_frame,
            text="EXIT",
            command=self.close_app,
            width=220, height=45,
            font=("Arial", 16, "bold"),
            fg_color="#444444"
        ).pack(pady=20)

        self.root.protocol("WM_DELETE_WINDOW", self.close_app)
        self.root.mainloop()

    # ── Change detection mode (person / object) ───────────────────────────
    def change_mode(self, value):
        self.mode = value

        # Show/hide person count mode controls depending on mode
        if value == "person":
            self.count_mode_label.pack(pady=(15, 2))
            self.count_mode_selector.pack(pady=5)
            self.count_mode_desc.pack(pady=(2, 10))
            self.reset_button.pack(pady=5)
        else:
            self.count_mode_label.pack_forget()
            self.count_mode_selector.pack_forget()
            self.count_mode_desc.pack_forget()
            self.reset_button.pack_forget()

    # ── Change person counting mode (live / cumulative) ───────────────────
    def change_count_mode(self, value):
        if value == "Live Count":
            self.person_count_mode = "live"
            self.count_mode_desc.configure(text="Resets when no person visible")
        else:
            self.person_count_mode = "cumulative"
            self.count_mode_desc.configure(text="Keeps increasing as people appear")

    # ── Reset cumulative count ─────────────────────────────────────────────
    def reset_cumulative(self):
        self.cumulative_count = 0
        self.last_frame_count = 0

    # ── Start camera ──────────────────────────────────────────────────────
    def start_camera(self):
        if self.camera_running:
            return
        self.camera_running = True
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        threading.Thread(target=self.update_camera, daemon=True).start()

    # ── Stop camera ───────────────────────────────────────────────────────
    def stop_camera(self):
        self.camera_running = False
        if self.camera:
            self.camera.release()

    # ── Main detection loop ───────────────────────────────────────────────
    def update_camera(self):

        while self.camera_running:
            success, frame = self.camera.read()
            if not success:
                continue

            results = model(frame, conf=0.4, iou=0.3, imgsz=960, verbose=False)[0]

            object_counts = Counter()

            for box in results.boxes:
                class_id    = int(box.cls[0])
                confidence  = float(box.conf[0])
                object_name = model.names[class_id]

                # Filter by mode
                if self.mode == "person" and object_name != "person":
                    continue
                if self.mode == "object" and object_name == "person":
                    continue

                object_counts[object_name] += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)

                label = f"{object_name} {confidence:.0%}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), (0, 200, 80), -1)
                cv2.putText(frame, label, (x1 + 3, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # ── Person count logic ────────────────────────────────────────
            current_frame_count = object_counts.get("person", 0)

            if self.mode == "person" and self.person_count_mode == "cumulative":
                # If more people visible now than last frame, add the difference
                if current_frame_count > self.last_frame_count:
                    self.cumulative_count += (current_frame_count - self.last_frame_count)
                self.last_frame_count = current_frame_count
                display_count = self.cumulative_count
            else:
                # Live mode — just show what's visible right now
                display_count = current_frame_count

            total = sum(object_counts.values()) if self.mode == "object" else display_count

            # ── Update count box ──────────────────────────────────────────
            self.count_box.delete("1.0", "end")

            if self.mode == "person":
                mode_tag = "CUMULATIVE TOTAL" if self.person_count_mode == "cumulative" else "LIVE COUNT"
                self.count_box.insert("end", f"{'PERSON':15} : {display_count}\n")
                self.count_box.insert("end", "\n-----------------------\n")
                self.count_box.insert("end", f"{mode_tag} : {display_count}")
            else:
                for name, count in object_counts.items():
                    low = "  ⚠ LOW" if count < 3 else ""
                    self.count_box.insert("end", f"{name.upper():15} : {count}{low}\n")
                self.count_box.insert("end", "\n-----------------------\n")
                self.count_box.insert("end", f"TOTAL OBJECTS : {total}")

            # ── Crowd status (person mode only) ───────────────────────────
            if self.mode == "person":
                if   current_frame_count == 0:  status, color = "EMPTY",          "gray"
                elif current_frame_count <= 3:   status, color = "LOW FOOTFALL",   "green"
                elif current_frame_count <= 7:   status, color = "MODERATE CROWD", "orange"
                else:                            status, color = "HIGH DENSITY",   "red"
                self.status_label.configure(text=f"Status: {status}", text_color=color)
            else:
                self.status_label.configure(text="Mode: Object Counting", text_color=GREEN)

            # ── Session timer ─────────────────────────────────────────────
            elapsed    = int(time.time() - self.start_time)
            mins, secs = divmod(elapsed, 60)
            self.timer_label.configure(text=f"Session: {mins:02d}:{secs:02d}")

            # ── Show video frame ──────────────────────────────────────────
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame).resize((1000, 700))
            photo = ImageTk.PhotoImage(image=image)
            self.video_label.configure(image=photo)
            self.video_label.image = photo

        if self.camera:
            self.camera.release()

    # ── Close app ─────────────────────────────────────────────────────────
    def close_app(self):
        self.camera_running = False
        if self.camera:
            self.camera.release()
        self.root.destroy()

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    SmartDetectionApp()
