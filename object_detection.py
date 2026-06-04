import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
from collections import Counter
import threading
import time
import requests

# ── Settings ──────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

GREEN = "#00C851"
DARK  = "#181818"

# ── Ntfy Topics ───────────────────────────────────────────────────────────────
# Both laptops will receive the alert at the same time
# Open these URLs on each laptop to receive notifications:
# Your laptop   → https://ntfy.sh/smartdetection_mylaptop
# Other laptop  → https://ntfy.sh/smartdetection_otherlaptop

MY_LAPTOP_TOPIC    = "smartdetection_mylaptop"
OTHER_LAPTOP_TOPIC = "smartdetection_otherlaptop"

LOW_STOCK_LIMIT = 3

# ── Load YOLO ─────────────────────────────────────────────────────────────────
model = YOLO("yolov8m.pt")

# ── Send notification to a single topic ───────────────────────────────────────
def send_to_topic(topic, item_name, count):
    try:
        response = requests.post(
            f"https://ntfy.sh/{topic}",
            data=(
                f"LOW STOCK ALERT!\n"
                f"Item  : {item_name.upper()}\n"
                f"Count : {count} remaining\n"
                f"Action: Order needs to be placed\n"
                f"Time  : {time.strftime('%H:%M:%S')}"
            ).encode("utf-8"),
            headers={
                "Title":    "Low Stock Detected",
                "Priority": "urgent",
                "Tags":     "warning,package"
            },
            timeout=10
        )
        response.raise_for_status()
        print(f"Alert sent to: {topic}")
        return True
    except Exception as e:
        print(f"Alert failed for {topic}: {e}")
        return False

# ── Send to BOTH laptops at the same time ─────────────────────────────────────
def send_notification(item_name, count, on_done=None):
    results = []

    def worker(topic):
        results.append(send_to_topic(topic, item_name, count))

    t1 = threading.Thread(target=worker, args=(MY_LAPTOP_TOPIC,),    daemon=True)
    t2 = threading.Thread(target=worker, args=(OTHER_LAPTOP_TOPIC,), daemon=True)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    if on_done:
        on_done(all(results), item_name, count)

# ── Main App ──────────────────────────────────────────────────────────────────
class SmartDetectionApp:

    def __init__(self):

        self.root = ctk.CTk()
        self.root.geometry("1500x850")
        self.root.title("Smart Detection System")
        self.root.configure(fg_color=DARK)

        self.camera_running    = False
        self.mode              = "person"
        self.start_time        = time.time()
        self.camera            = None
        self.count_mode          = "live"
        self.cumulative_count    = 0
        self.last_frame_count    = 0
        self.object_cumulative   = Counter()
        self.last_object_counts  = Counter()
        self.alert_sent_for      = {}
        self.ALERT_COOLDOWN    = 60

        # ── Left: camera ──────────────────────────────────────────────────
        self.left_frame = ctk.CTkFrame(self.root, fg_color="#101010", corner_radius=15)
        self.left_frame.pack(side="left", fill="both", expand=True, padx=15, pady=15)

        ctk.CTkLabel(
            self.left_frame,
            text="SMART DETECTION SYSTEM",
            font=("Arial", 28, "bold"),
            text_color=GREEN
        ).pack(pady=15)

        self.video_label = ctk.CTkLabel(self.left_frame, text="")
        self.video_label.pack(padx=10, pady=10)

        # ── Right: controls ───────────────────────────────────────────────
        self.right_frame = ctk.CTkFrame(self.root, width=320, fg_color="#1E1E1E", corner_radius=15)
        self.right_frame.pack(side="right", fill="y", padx=15, pady=15)

        ctk.CTkLabel(
            self.right_frame,
            text="DETECTION SETTINGS",
            font=("Arial", 22, "bold"),
            text_color=GREEN
        ).pack(pady=20)

        # Detection mode
        self.mode_selector = ctk.CTkOptionMenu(
            self.right_frame,
            values=["person", "object"],
            command=self.change_mode,
            width=220, height=40,
            fg_color=GREEN, button_color="#009944"
        )
        self.mode_selector.pack(pady=10)

        # Count mode (person / object)
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

        self.count_mode_desc = ctk.CTkLabel(
            self.right_frame,
            text="Resets when no person visible",
            font=("Arial", 11),
            text_color="#777777"
        )
        self.count_mode_desc.pack(pady=(2, 10))

        self.reset_button = ctk.CTkButton(
            self.right_frame,
            text="↺  Reset Count",
            command=self.reset_cumulative,
            width=220, height=35,
            fg_color="#333333", hover_color="#444444"
        )
        self.reset_button.pack(pady=5)

        ctk.CTkLabel(
            self.right_frame,
            text="LIVE COUNTS",
            font=("Arial", 20, "bold"),
            text_color=GREEN
        ).pack(pady=(12, 4))

        self.count_display_label = ctk.CTkLabel(
            self.right_frame,
            text="—",
            font=("Arial", 36, "bold"),
            text_color="white"
        )
        self.count_display_label.pack(pady=4)

        self.count_box = ctk.CTkTextbox(
            self.right_frame,
            width=260, height=180,
            font=("Consolas", 14),
            fg_color="#111111",
            text_color="#EEEEEE"
        )
        self.count_box.pack(pady=5)
        self.count_box.insert("end", "Press START CAMERA to begin counting.")
        self.count_box.configure(state="disabled")

        # Start / Stop
        ctk.CTkButton(
            self.right_frame,
            text="▶ START CAMERA",
            command=self.start_camera,
            width=220, height=45,
            font=("Arial", 16, "bold"),
            fg_color=GREEN, hover_color="#009944"
        ).pack(pady=15)

        ctk.CTkButton(
            self.right_frame,
            text="■ STOP CAMERA",
            command=self.stop_camera,
            width=220, height=45,
            font=("Arial", 16, "bold"),
            fg_color="#D32F2F", hover_color="#B71C1C"
        ).pack(pady=5)

        # ── Alert Topics Info ─────────────────────────────────────────────
        ctk.CTkLabel(
            self.right_frame,
            text="─────────────────────",
            text_color="#333333"
        ).pack(pady=5)

        ctk.CTkLabel(
            self.right_frame,
            text="ALERT TOPICS",
            font=("Arial", 12, "bold"),
            text_color="gray"
        ).pack(pady=(5, 2))

        ctk.CTkLabel(
            self.right_frame,
            text=f"🖥 Your laptop:",
            font=("Arial", 11),
            text_color="#888888"
        ).pack()

        ctk.CTkLabel(
            self.right_frame,
            text=f"ntfy.sh/{MY_LAPTOP_TOPIC}",
            font=("Arial", 11),
            text_color="#29B6F6"
        ).pack()

        ctk.CTkLabel(
            self.right_frame,
            text=f"💻 Other laptop:",
            font=("Arial", 11),
            text_color="#888888"
        ).pack(pady=(6, 0))

        ctk.CTkLabel(
            self.right_frame,
            text=f"ntfy.sh/{OTHER_LAPTOP_TOPIC}",
            font=("Arial", 11),
            text_color="#29B6F6"
        ).pack(pady=(0, 8))

        # Session timer
        self.timer_label = ctk.CTkLabel(
            self.right_frame,
            text="Session: 00:00",
            font=("Arial", 18),
            text_color="white"
        )
        self.timer_label.pack(pady=10)

        # Status
        self.status_label = ctk.CTkLabel(
            self.right_frame,
            text="Status: Waiting...",
            font=("Arial", 18, "bold"),
            text_color="orange"
        )
        self.status_label.pack(pady=5)

        # Alert status
        self.alert_label = ctk.CTkLabel(
            self.right_frame,
            text="",
            font=("Arial", 11),
            text_color="#29B6F6",
            wraplength=240
        )
        self.alert_label.pack(pady=2)

        ctk.CTkButton(
            self.right_frame,
            text="EXIT",
            command=self.close_app,
            width=220, height=40,
            font=("Arial", 14, "bold"),
            fg_color="#444444"
        ).pack(pady=10)

        self.root.protocol("WM_DELETE_WINDOW", self.close_app)
        self.root.mainloop()

    def change_mode(self, value):
        self.mode = value
        if value == "person":
            self.count_mode_label.configure(text="Person Count Mode:")
        else:
            self.count_mode_label.configure(text="Object Count Mode:")
        self._update_count_mode_desc()

    def change_count_mode(self, value):
        self.count_mode = "live" if value == "Live Count" else "cumulative"
        self._update_count_mode_desc()

    def _update_count_mode_desc(self):
        if self.mode == "person":
            if self.count_mode == "live":
                self.count_mode_desc.configure(text="Resets when no person visible")
            else:
                self.count_mode_desc.configure(text="Keeps increasing as people appear")
        else:
            if self.count_mode == "live":
                self.count_mode_desc.configure(text="Counts only visible objects")
            else:
                self.count_mode_desc.configure(text="Keeps increasing per item type")

    def reset_cumulative(self):
        self.cumulative_count = 0
        self.last_frame_count = 0
        self.object_cumulative.clear()
        self.last_object_counts.clear()
        self.count_display_label.configure(text="0", text_color="gray")
        self._set_count_box("Counts reset.")

    def _set_count_box(self, text):
        self.count_box.configure(state="normal")
        self.count_box.delete("1.0", "end")
        self.count_box.insert("end", text)
        self.count_box.configure(state="disabled")

    def _on_alert_done(self, success, item_name, count):
        if success:
            self.alert_label.configure(
                text=f"Alert sent to both laptops!\n{item_name} has only {count} left",
                text_color="#29B6F6"
            )
        else:
            self.alert_label.configure(
                text="Alert failed — check internet connection",
                text_color="#FF6B6B"
            )

    def check_and_send_alert(self, item_name, count):
        key = item_name.lower()
        now       = time.time()
        last_sent = self.alert_sent_for.get(key, 0)

        if now - last_sent > self.ALERT_COOLDOWN:
            self.alert_sent_for[key] = now

            def run_alert():
                send_notification(
                    item_name, count,
                    on_done=lambda ok, name, n: self.root.after(
                        0, self._on_alert_done, ok, name, n
                    )
                )

            threading.Thread(target=run_alert, daemon=True).start()

    def start_camera(self):
        if self.camera_running:
            return
        self.camera_running = True
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        threading.Thread(target=self.update_camera, daemon=True).start()

    def stop_camera(self):
        self.camera_running = False
        if self.camera:
            self.camera.release()
        self.count_display_label.configure(text="—", text_color="white")
        self._set_count_box("Camera stopped.")
        self.status_label.configure(text="Status: Waiting...", text_color="orange")

    def _refresh_ui(self, frame, object_counts, display_count, display_object_counts, total, mode_tag):
        if self.mode == "person":
            self.count_display_label.configure(
                text=str(display_count),
                text_color=GREEN if display_count > 0 else "gray"
            )
            lines = [
                f"{'PERSON':15} : {display_count}",
                "",
                "-----------------------",
                f"{mode_tag} : {display_count}",
            ]
        else:
            self.count_display_label.configure(
                text=str(total),
                text_color=GREEN if total > 0 else "gray"
            )
            if display_object_counts:
                lines = [
                    f"{name.upper():13} : {count}"
                    + ("  LOW" if object_counts.get(name, 0) < LOW_STOCK_LIMIT else "")
                    for name, count in sorted(display_object_counts.items())
                ]
            else:
                lines = ["No objects detected"]
            lines += ["", "-----------------------", f"{mode_tag} : {total}"]

        self._set_count_box("\n".join(lines))

        current_frame_count = object_counts.get("person", 0)
        if self.mode == "person":
            if   current_frame_count == 0: status, color = "EMPTY",          "gray"
            elif current_frame_count <= 3:  status, color = "LOW FOOTFALL",   "green"
            elif current_frame_count <= 7:  status, color = "MODERATE CROWD", "orange"
            else:                           status, color = "HIGH DENSITY",   "red"
            self.status_label.configure(text=f"Status: {status}", text_color=color)
        else:
            self.status_label.configure(text=f"Objects visible: {total}", text_color=GREEN)

        elapsed    = int(time.time() - self.start_time)
        mins, secs = divmod(elapsed, 60)
        self.timer_label.configure(text=f"Session: {mins:02d}:{secs:02d}")

        photo = ImageTk.PhotoImage(image=frame)
        self.video_label.configure(image=photo)
        self.video_label.image = photo

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

            if self.mode == "object":
                for item_name, count in object_counts.items():
                    if count < LOW_STOCK_LIMIT:
                        self.check_and_send_alert(item_name, count)

            current_frame_count = object_counts.get("person", 0)
            if self.mode == "person" and self.count_mode == "cumulative":
                if current_frame_count > self.last_frame_count:
                    self.cumulative_count += (current_frame_count - self.last_frame_count)
                self.last_frame_count = current_frame_count
                display_count = self.cumulative_count
            else:
                display_count = current_frame_count

            if self.mode == "object" and self.count_mode == "cumulative":
                for name, count in object_counts.items():
                    last = self.last_object_counts.get(name, 0)
                    if count > last:
                        self.object_cumulative[name] += count - last
                self.last_object_counts = object_counts.copy()
                display_object_counts = self.object_cumulative
            else:
                display_object_counts = object_counts

            total = (
                sum(display_object_counts.values())
                if self.mode == "object"
                else display_count
            )

            mode_tag = "CUMULATIVE TOTAL" if self.count_mode == "cumulative" else "LIVE COUNT"

            summary = f"Total: {total}" if self.mode == "object" else f"People: {display_count}"
            cv2.putText(
                frame, summary, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 100), 3
            )

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb).resize((1000, 700))

            self.root.after(
                0,
                self._refresh_ui,
                image,
                dict(object_counts),
                display_count,
                dict(display_object_counts),
                total,
                mode_tag,
            )

        if self.camera:
            self.camera.release()

    def close_app(self):
        self.camera_running = False
        if self.camera:
            self.camera.release()
        self.root.destroy()

if __name__ == "__main__":
    SmartDetectionApp()