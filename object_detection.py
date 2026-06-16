import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
from collections import Counter
import threading
import time
import requests

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

GREEN = "#00C851"
DARK  = "#181818"

MY_LAPTOP_TOPIC    = "smartdetection_mylaptop"
OTHER_LAPTOP_TOPIC = "smartdetection_otherlaptop"
LOW_STOCK_LIMIT    = 3

model = YOLO("yolov8m.pt")


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
            headers={"Title": "Low Stock Detected", "Priority": "urgent", "Tags": "warning,package"},
            timeout=10
        )
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Alert failed for {topic}: {e}")
        return False


def send_notification(item_name, count, on_done=None):
    results = []
    def worker(topic):
        results.append(send_to_topic(topic, item_name, count))
    t1 = threading.Thread(target=worker, args=(MY_LAPTOP_TOPIC,),    daemon=True)
    t2 = threading.Thread(target=worker, args=(OTHER_LAPTOP_TOPIC,), daemon=True)
    t1.start(); t2.start()
    t1.join();  t2.join()
    if on_done:
        on_done(all(results), item_name, count)


class StockTable(ctk.CTkFrame):
    """
    Visible grid table:  Item | Before | → | After | Diff | Status
    Uses grid() with column weights so cells actually appear.
    """
    HEADERS   = ["Item",  "Before", "→", "After", "Diff", "Status"]
    COL_W     = [7,        5,        1,    5,        5,      8      ]  # relative weights

    def __init__(self, master, **kw):
        super().__init__(master, fg_color="#111111", corner_radius=8, **kw)

        # configure column weights so grid fills the frame
        for c, w in enumerate(self.COL_W):
            self.columnconfigure(c, weight=w, minsize=10)

        self._draw_header()
        self._data_widgets = []   # rows × cols of CTkLabel

    # ── header ──────────────────────────────────────────────────────────────
    def _draw_header(self):
        for c, h in enumerate(self.HEADERS):
            ctk.CTkLabel(
                self, text=h,
                font=("Arial", 11, "bold"),
                text_color="#888888",
                fg_color="#1A1A1A",
                corner_radius=0
            ).grid(row=0, column=c, sticky="ew", padx=1, pady=(4, 2))

        # separator line
        sep = ctk.CTkFrame(self, fg_color="#333333", height=1, corner_radius=0)
        sep.grid(row=1, column=0, columnspan=len(self.HEADERS), sticky="ew", padx=4)

    # ── clear data rows ──────────────────────────────────────────────────────
    def _clear_data(self):
        for row in self._data_widgets:
            for w in row:
                w.destroy()
        self._data_widgets.clear()

    # ── public refresh ───────────────────────────────────────────────────────
    def refresh(self, before_stock: dict, after_stock: dict,
                alert_sent_for: dict, cooldown: int):
        self._clear_data()

        if not after_stock:
            lbl = ctk.CTkLabel(
                self, text="No objects tracked yet.",
                font=("Arial", 11), text_color="#555555"
            )
            lbl.grid(row=2, column=0, columnspan=len(self.HEADERS), pady=10)
            self._data_widgets.append([lbl])
            return

        now = time.time()
        for i, name in enumerate(sorted(after_stock.keys())):
            before = before_stock.get(name, after_stock[name])
            after  = after_stock[name]
            diff   = after - before

            alert_active = (
                name.lower() in alert_sent_for and
                (now - alert_sent_for[name.lower()]) < cooldown
            )

            # colours
            if alert_active:
                after_col  = "#29B6F6"
                status_txt = "ALERT SENT"
                status_col = "#29B6F6"
            elif after < LOW_STOCK_LIMIT:
                after_col  = "#FF4444"
                status_txt = "LOW"
                status_col = "#FF4444"
            elif after == LOW_STOCK_LIMIT:
                after_col  = "#FFA500"
                status_txt = "WATCH"
                status_col = "#FFA500"
            else:
                after_col  = GREEN
                status_txt = "OK"
                status_col = GREEN

            diff_col  = "#FF4444" if diff < 0 else "#888888"
            diff_str  = f"{diff:+d}" if diff != 0 else "0"
            grid_row  = (i * 2) + 2   # leave room for separator rows

            row_cells = [
                (name.upper(),   "w",  "#FFFFFF", ("Arial", 11, "bold")),
                (str(before),    "center", "#888888", ("Arial", 11)),
                ("→",            "center", "#444444", ("Arial", 11)),
                (str(after),     "center", after_col,  ("Arial", 11, "bold")),
                (diff_str,       "center", diff_col,   ("Arial", 11)),
                (status_txt,     "center", status_col, ("Arial", 10, "bold")),
            ]

            row_widgets = []
            for c, (txt, anchor, color, font) in enumerate(row_cells):
                lbl = ctk.CTkLabel(
                    self, text=txt,
                    font=font, text_color=color,
                    fg_color="#161616" if i % 2 == 0 else "#111111",
                    corner_radius=0,
                    anchor=anchor
                )
                lbl.grid(row=grid_row, column=c, sticky="ew", padx=1, pady=2)
                row_widgets.append(lbl)

            # thin row separator
            sep = ctk.CTkFrame(self, fg_color="#222222", height=1, corner_radius=0)
            sep.grid(row=grid_row + 1, column=0,
                     columnspan=len(self.HEADERS), sticky="ew", padx=6)
            row_widgets.append(sep)

            self._data_widgets.append(row_widgets)


# ── Main App ───────────────────────────────────────────────────────────────────
class SmartDetectionApp:

    def __init__(self):
        self.root = ctk.CTk()
        self.root.geometry("1500x900")
        self.root.title("Smart Detection System")
        self.root.configure(fg_color=DARK)

        self.camera_running   = False
        self.mode             = "person"
        self.start_time       = time.time()
        self.camera           = None
        self.count_mode       = "live"
        self.cumulative_count = 0
        self.last_frame_count = 0
        self.object_cumulative  = Counter()
        self.last_object_counts = Counter()
        self.alert_sent_for   = {}
        self.ALERT_COOLDOWN   = 60
        self.before_stock     = {}
        self.after_stock      = {}

        # ── Left: camera ───────────────────────────────────────────────────
        self.left_frame = ctk.CTkFrame(self.root, fg_color="#101010", corner_radius=15)
        self.left_frame.pack(side="left", fill="both", expand=True, padx=15, pady=15)

        ctk.CTkLabel(
            self.left_frame, text="SMART DETECTION SYSTEM",
            font=("Arial", 28, "bold"), text_color=GREEN
        ).pack(pady=15)

        self.video_label = ctk.CTkLabel(self.left_frame, text="")
        self.video_label.pack(padx=10, pady=10)

        # ── Right: controls (scrollable so nothing gets cut off) ───────────
        right_outer = ctk.CTkFrame(self.root, width=360, fg_color="#1E1E1E", corner_radius=15)
        right_outer.pack(side="right", fill="y", padx=15, pady=15)

        self.right_frame = ctk.CTkScrollableFrame(
            right_outer, width=330, fg_color="#1E1E1E", corner_radius=0
        )
        self.right_frame.pack(fill="both", expand=True, padx=0, pady=0)

        ctk.CTkLabel(
            self.right_frame, text="DETECTION SETTINGS",
            font=("Arial", 20, "bold"), text_color=GREEN
        ).pack(pady=14)

        self.mode_selector = ctk.CTkOptionMenu(
            self.right_frame, values=["person", "object"],
            command=self.change_mode, width=220, height=38,
            fg_color=GREEN, button_color="#009944"
        )
        self.mode_selector.pack(pady=6)

        self.count_mode_label = ctk.CTkLabel(
            self.right_frame, text="Person Count Mode:",
            font=("Arial", 12), text_color="gray"
        )
        self.count_mode_label.pack(pady=(10, 2))

        self.count_mode_selector = ctk.CTkSegmentedButton(
            self.right_frame, values=["Live Count", "Cumulative"],
            command=self.change_count_mode, width=220,
            font=("Arial", 12, "bold"),
            selected_color=GREEN, selected_hover_color="#009944"
        )
        self.count_mode_selector.set("Live Count")
        self.count_mode_selector.pack(pady=3)

        self.count_mode_desc = ctk.CTkLabel(
            self.right_frame, text="Resets when no person visible",
            font=("Arial", 10), text_color="#777777"
        )
        self.count_mode_desc.pack(pady=(1, 6))

        ctk.CTkButton(
            self.right_frame, text="↺  Reset Count",
            command=self.reset_cumulative,
            width=220, height=32,
            fg_color="#333333", hover_color="#444444"
        ).pack(pady=3)

        ctk.CTkLabel(
            self.right_frame, text="LIVE COUNTS",
            font=("Arial", 16, "bold"), text_color=GREEN
        ).pack(pady=(10, 2))

        self.count_display_label = ctk.CTkLabel(
            self.right_frame, text="—",
            font=("Arial", 30, "bold"), text_color="white"
        )
        self.count_display_label.pack(pady=2)

        self.count_box = ctk.CTkTextbox(
            self.right_frame, width=290, height=90,
            font=("Consolas", 11), fg_color="#111111", text_color="#EEEEEE"
        )
        self.count_box.pack(pady=4)
        self.count_box.insert("end", "Press START CAMERA to begin.")
        self.count_box.configure(state="disabled")

        # ── Stock table section ────────────────────────────────────────────
        ctk.CTkLabel(
            self.right_frame, text="─────────────────────",
            text_color="#333333"
        ).pack(pady=2)

        ctk.CTkLabel(
            self.right_frame, text="STOCK  —  BEFORE / AFTER",
            font=("Arial", 13, "bold"), text_color=GREEN
        ).pack(pady=(4, 6))

        self.stock_table = StockTable(self.right_frame)
        self.stock_table.pack(fill="x", padx=8, pady=(0, 4))

        ctk.CTkButton(
            self.right_frame, text="↺  Reset Stock Baseline",
            command=self.reset_stock_baseline,
            width=220, height=30,
            fg_color="#333333", hover_color="#444444",
            font=("Arial", 11)
        ).pack(pady=(4, 8))

        ctk.CTkLabel(
            self.right_frame, text="─────────────────────",
            text_color="#333333"
        ).pack(pady=2)

        # Start / Stop
        ctk.CTkButton(
            self.right_frame, text="▶ START CAMERA",
            command=self.start_camera,
            width=220, height=42,
            font=("Arial", 14, "bold"),
            fg_color=GREEN, hover_color="#009944"
        ).pack(pady=6)

        ctk.CTkButton(
            self.right_frame, text="■ STOP CAMERA",
            command=self.stop_camera,
            width=220, height=42,
            font=("Arial", 14, "bold"),
            fg_color="#D32F2F", hover_color="#B71C1C"
        ).pack(pady=4)

        ctk.CTkLabel(self.right_frame, text="─────────────────────", text_color="#333333").pack(pady=4)
        ctk.CTkLabel(self.right_frame, text="ALERT TOPICS", font=("Arial", 11, "bold"), text_color="gray").pack()
        ctk.CTkLabel(self.right_frame, text="🖥 Your laptop:",       font=("Arial", 10), text_color="#888888").pack()
        ctk.CTkLabel(self.right_frame, text=f"ntfy.sh/{MY_LAPTOP_TOPIC}",    font=("Arial", 10), text_color="#29B6F6").pack()
        ctk.CTkLabel(self.right_frame, text="💻 Other laptop:",      font=("Arial", 10), text_color="#888888").pack(pady=(4,0))
        ctk.CTkLabel(self.right_frame, text=f"ntfy.sh/{OTHER_LAPTOP_TOPIC}", font=("Arial", 10), text_color="#29B6F6").pack(pady=(0,6))

        self.timer_label = ctk.CTkLabel(self.right_frame, text="Session: 00:00", font=("Arial", 15), text_color="white")
        self.timer_label.pack(pady=4)

        self.status_label = ctk.CTkLabel(self.right_frame, text="Status: Waiting...", font=("Arial", 15, "bold"), text_color="orange")
        self.status_label.pack(pady=2)

        self.alert_label = ctk.CTkLabel(self.right_frame, text="", font=("Arial", 10), text_color="#29B6F6", wraplength=280)
        self.alert_label.pack(pady=2)

        ctk.CTkButton(
            self.right_frame, text="EXIT",
            command=self.close_app,
            width=220, height=36,
            font=("Arial", 13, "bold"),
            fg_color="#444444"
        ).pack(pady=8)

        self.root.protocol("WM_DELETE_WINDOW", self.close_app)
        self.root.mainloop()

    # ── mode helpers ───────────────────────────────────────────────────────────
    def change_mode(self, value):
        self.mode = value
        self.count_mode_label.configure(
            text="Person Count Mode:" if value == "person" else "Object Count Mode:"
        )
        self._update_count_mode_desc()

    def change_count_mode(self, value):
        self.count_mode = "live" if value == "Live Count" else "cumulative"
        self._update_count_mode_desc()

    def _update_count_mode_desc(self):
        descs = {
            ("person", "live"):       "Resets when no person visible",
            ("person", "cumulative"): "Keeps increasing as people appear",
            ("object", "live"):       "Counts only visible objects",
            ("object", "cumulative"): "Keeps increasing per item type",
        }
        self.count_mode_desc.configure(text=descs[(self.mode, self.count_mode)])

    def reset_cumulative(self):
        self.cumulative_count = 0
        self.last_frame_count = 0
        self.object_cumulative.clear()
        self.last_object_counts.clear()
        self.count_display_label.configure(text="0", text_color="gray")
        self._set_count_box("Counts reset.")

    def reset_stock_baseline(self):
        self.before_stock.clear()
        self.after_stock.clear()
        self.stock_table.refresh({}, {}, {}, self.ALERT_COOLDOWN)

    def _set_count_box(self, text):
        self.count_box.configure(state="normal")
        self.count_box.delete("1.0", "end")
        self.count_box.insert("end", text)
        self.count_box.configure(state="disabled")

    def _on_alert_done(self, success, item_name, count):
        msg = (f"Alert sent!\n{item_name} has only {count} left"
               if success else "Alert failed — check internet")
        self.alert_label.configure(text=msg, text_color="#29B6F6" if success else "#FF6B6B")
        self.stock_table.refresh(
            self.before_stock, self.after_stock,
            self.alert_sent_for, self.ALERT_COOLDOWN
        )

    def check_and_send_alert(self, item_name, count):
        key = item_name.lower()
        now = time.time()
        if now - self.alert_sent_for.get(key, 0) > self.ALERT_COOLDOWN:
            self.alert_sent_for[key] = now
            def run_alert():
                send_notification(
                    item_name, count,
                    on_done=lambda ok, n, c: self.root.after(0, self._on_alert_done, ok, n, c)
                )
            threading.Thread(target=run_alert, daemon=True).start()

    # ── camera ─────────────────────────────────────────────────────────────────
    def start_camera(self):
        if self.camera_running:
            return
        self.camera_running = True
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        threading.Thread(target=self.update_camera, daemon=True).start()

    def stop_camera(self):
        self.camera_running = False
        if self.camera:
            self.camera.release()
        self.count_display_label.configure(text="—", text_color="white")
        self._set_count_box("Camera stopped.")
        self.status_label.configure(text="Status: Waiting...", text_color="orange")

    def _refresh_ui(self, frame, object_counts, display_count,
                    display_object_counts, total, mode_tag):
        if self.mode == "person":
            self.count_display_label.configure(
                text=str(display_count),
                text_color=GREEN if display_count > 0 else "gray"
            )
            lines = [f"{'PERSON':15} : {display_count}", "",
                     "-----------------------", f"{mode_tag} : {display_count}"]
        else:
            self.count_display_label.configure(
                text=str(total),
                text_color=GREEN if total > 0 else "gray"
            )
            lines = (
                [f"{n.upper():13} : {c}" +
                 ("  LOW" if object_counts.get(n, 0) < LOW_STOCK_LIMIT else "")
                 for n, c in sorted(display_object_counts.items())]
                if display_object_counts else ["No objects detected"]
            )
            lines += ["", "-----------------------", f"{mode_tag} : {total}"]

        self._set_count_box("\n".join(lines))

        if self.mode == "person":
            fc = object_counts.get("person", 0)
            s, c = (("EMPTY",          "gray")   if fc == 0 else
                    ("LOW FOOTFALL",   "green")  if fc <= 3 else
                    ("MODERATE CROWD", "orange") if fc <= 7 else
                    ("HIGH DENSITY",   "red"))
            self.status_label.configure(text=f"Status: {s}", text_color=c)
        else:
            self.status_label.configure(text=f"Objects visible: {total}", text_color=GREEN)
            self.stock_table.refresh(
                self.before_stock, self.after_stock,
                self.alert_sent_for, self.ALERT_COOLDOWN
            )

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
                # lock "before" on first detection of each item
                for item_name, count in object_counts.items():
                    if item_name not in self.before_stock:
                        self.before_stock[item_name] = count

                # update "after" for ALL ever-seen items:
                # visible items get their current count, missing items get 0
                for item_name in self.before_stock:
                    self.after_stock[item_name] = object_counts.get(item_name, 0)

                # alert based on after count (includes items now at 0)
                for item_name, count in self.after_stock.items():
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

            total    = sum(display_object_counts.values()) if self.mode == "object" else display_count
            mode_tag = "CUMULATIVE TOTAL" if self.count_mode == "cumulative" else "LIVE COUNT"
            summary  = f"Total: {total}" if self.mode == "object" else f"People: {display_count}"
            cv2.putText(frame, summary, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 100), 3)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image     = Image.fromarray(frame_rgb).resize((1000, 700))

            self.root.after(
                0, self._refresh_ui,
                image, dict(object_counts), display_count,
                dict(display_object_counts), total, mode_tag,
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