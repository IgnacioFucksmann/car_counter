# Car Counter - Vehicle Counting System

Vehicle detection, tracking, and counting system using YOLOv8 and ByteTrack.

## 🚀 Features

- **Vehicle detection** using YOLOv8 (car, motorcycle, bus, truck)
- **Object tracking** with ByteTrack to follow vehicles across frames
- **Vehicle counting** using configurable counting line
- **Speed estimation** with adaptive smoothing filter to reduce oscillations
- **Video annotation** with bounding boxes, tracker IDs, speed, and counters

## 📋 Requirements

- Python 3.12+
- CUDA (optional, for GPU acceleration)
- `uv` (Python package manager) or `pip`

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/IgnacioFucksmann/CarCounter.git
cd CarCounter
```

### 2. Create virtual environment

```bash
# With uv (recommended)
uv venv

# Or with standard venv
python -m venv .venv
```

### 3. Activate virtual environment

```bash
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

### 4. Install dependencies

```bash
# With uv
uv pip install -r requirements.txt

# Or with pip
pip install -r requirements.txt
```

### 5. Install ByteTrack

ByteTrack is not available on PyPI, it must be installed from GitHub:

```bash
bash install_bytetrack.sh
```

This script:
- Clones ByteTrack from GitHub
- Fixes dependency versions
- Installs ByteTrack in development mode

**Note:** If you prefer to install it manually, check `install_bytetrack.sh` for the steps.

## 🎯 Usage

### Basic configuration

Edit `main.py` to configure:

```python
input_video = "data/vehicle-counting.mp4"  # Input video
output_video = "output/vehicle-counting-result.mp4"  # Output video
model_name = "yolov8x.pt"  # YOLOv8 model (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
line_start = Point(50, 1500)  # Counting line start point
line_end = Point(3840 - 50, 1500)  # Counting line end point
```

### Run

```bash
python main.py
```

The script:
1. Loads the YOLOv8 model (automatically downloaded the first time)
2. Processes the video frame by frame
3. Detects and tracks vehicles
4. Counts vehicles crossing the line
5. Estimates vehicle speed with adaptive smoothing
6. Saves the processed video with annotations
7. Shows final statistics

### Output

The processed video is saved in `output/vehicle-counting-result.mp4` with:
- Bounding boxes around each vehicle
- Labels with tracker ID, class, and speed (km/h)
- Counting line with vehicle counter

At the end, statistics are shown:
```
Vehicles counted (in):  45
Vehicles counted (out): 42
Total:                  87
```

## 📁 Project Structure

```
car_counter/
├── data/                    # Input videos
│   └── vehicle-counting.mp4
├── output/                  # Processed videos (ignored by Git)
├── utils/                   # Utilities
│   ├── tracking_utils.py    # Tracking functions
│   ├── view_transformer.py  # Perspective transformation for speed estimation
│   └── speed_smoother.py    # Adaptive speed smoothing filter
├── libs/                    # External dependencies
│   └── ByteTrack/           # ByteTrack (ignored by Git)
├── model.py                 # VehicleDetector class
├── processor.py             # VideoProcessor class
├── main.py                  # Main script
├── install_bytetrack.sh     # ByteTrack installation script
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## 🔍 Available YOLOv8 Models

- `yolov8n.pt` - Nano (fastest, less accurate)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - XLarge (most accurate, slowest) - **Default**

Models are automatically downloaded the first time they are used.

## ⚙️ Advanced Configuration

### Change detected classes

In `model.py`, modify `class_ids`:

```python
detector = VehicleDetector(
    model_name="yolov8x.pt",
    class_ids=[2, 3, 5, 7]  # car, motorcycle, bus, truck
)
```

### Adjust ByteTrack parameters

In `processor.py`, modify `BYTETrackerArgs`:

```python
@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25      # Confidence threshold
    track_buffer: int = 30           # Buffer frames
    match_thresh: float = 0.8       # Matching threshold
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
```

### Speed estimation configuration

In `processor.py`, you can configure the speed estimation:

```python
processor = VideoProcessor(
    detector=detector,
    line_start=line_start,
    line_end=line_end,
    source_roi=None,  # Source polygon for perspective transformation
    target_roi=None,  # Target polygon for perspective transformation
    enable_speed_estimation=True,  # Enable/disable speed estimation
)
```

The `AdaptiveSpeedSmoother` uses exponential adaptive filtering:
- **Small changes** (< 3 km/h): High smoothing (alpha ~0.1) to reduce oscillations
- **Large changes** (> 3 km/h): Fast response (alpha up to 0.7) for real speed changes
- Parameters are configurable: `min_alpha`, `max_alpha`, `threshold`, `sensitivity`

## 🐛 Troubleshooting

### Error: "No module named 'yolox'"

ByteTrack is not installed. Run:
```bash
bash install_bytetrack.sh
```

### Error: "No module named 'torch'"

Install PyTorch:
```bash
# With CUDA 12.6
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Or CPU only
uv pip install torch torchvision
```

### Video processing is very slow

- Use a smaller model: `yolov8n.pt` or `yolov8s.pt`
- Make sure you have GPU with CUDA installed
- Reduce the input video resolution

## 📚 Additional Documentation
- `install_bytetrack.sh` - Commented script with explanation of each step

## 🤝 Contributing

Contributions are welcome. Please:
1. Fork the project
2. Create a branch for your feature (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is under the MIT License. See `LICENSE` for more details.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLOv8
- [FoundationVision](https://github.com/FoundationVision/ByteTrack) - ByteTrack
- [Roboflow Supervision](https://github.com/roboflow/supervision) - Video utilities

## 📧 Contact

For questions or suggestions, open an issue in the repository.
