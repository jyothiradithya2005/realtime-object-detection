# Smart Detection System

A modern AI-powered real-time object and person detection system built using Python, YOLOv8, OpenCV, and CustomTkinter.

---

## Features

- Real-time object detection using YOLOv8
- Person detection mode
- Object counting mode
- Live and cumulative person counting
- Modern dark-themed GUI
- Crowd density analysis
- Session timer
- Camera start/stop controls
- Live detection statistics
- Bounding box visualization
- Smart crowd status monitoring

---

## Technologies Used

- Python
- OpenCV
- YOLOv8
- Ultralytics
- CustomTkinter
- Pillow (PIL)
- NumPy

---

## Why OpenCV and YOLO?

YOLOv8 is used for fast and accurate real-time object detection. It can detect multiple objects simultaneously with high speed, making it ideal for live camera applications.

OpenCV is used for webcam access, video frame processing, drawing bounding boxes, and displaying the live detection output.

Compared to traditional CNN models and TensorFlow Object Detection API, YOLO provides:
- Faster real-time performance
- Easier implementation
- Better efficiency for live detection systems

Together, YOLO and OpenCV make the project fast, accurate, and suitable for real-time smart surveillance and object detection applications.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/smart-detection-system.git
cd smart-detection-system
```

### Install Dependencies

```bash
pip install customtkinter ultralytics opencv-python pillow numpy
```

---

## Run the Project

```bash
python main.py
```

---

## Project Structure

```bash
smart-detection-system/
│
├── main.py
├── README.md
├── requirements.txt
└── yolov8m.pt
```

---

## Detection Modes

### Person Detection
- Detects only people
- Supports:
  - Live Count Mode
  - Cumulative Count Mode
- Crowd density monitoring:
  - Empty
  - Low Footfall
  - Moderate Crowd
  - High Density

### Object Detection
- Detects multiple object classes
- Displays object counts in real time

---

## Use Cases

- Smart surveillance systems
- Crowd monitoring in public places
- Shopping mall footfall analysis
- Classroom attendance monitoring
- Office visitor tracking
- Traffic and pedestrian monitoring
- Security monitoring systems
- Retail analytics
- Smart city applications
- Event crowd management

---

## UI Features

- Modern dark interface
- Green-themed dashboard
- Real-time camera feed
- Detection statistics panel
- Session monitoring
- Interactive controls

---

## YOLO Model Used

This project uses:

```bash
yolov8m.pt
```

from the Ultralytics YOLOv8 model family for accurate and fast detection.

---

## Future Improvements

- Object tracking
- Database logging
- Detection history
- Alert system
- Export reports
- Multi-camera support

---

## Author

Sagiraju V. S. Jyothiradithya 
7013113567 

---

## License

This project is open-source and available for educational and research purposes.
