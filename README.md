# 🚦 Smart Traffic Signal Optimizer

A comprehensive traffic management system that uses **Computer Vision**, **Machine Learning**, and a **Web Dashboard** to optimize traffic flow at intersections. The system detects vehicles, identifies ambulances, and dynamically adjusts traffic signal timings based on real-time traffic conditions.

## ✨ Features

- 🎥 **Real-time Vehicle Detection** - YOLOv8-based computer vision for accurate vehicle counting
- 🤖 **ML-Powered Optimization** - TensorFlow models predict optimal signal timings
- 🚑 **Emergency Vehicle Priority** - Automatic ambulance detection with priority override
- 🌐 **Web Dashboard** - Real-time monitoring and control interface
- 🔄 **Hybrid Mode** - Combine computer vision with manual input
- 📊 **Live Updates** - WebSocket-based real-time state synchronization
- 🎯 **Graceful Degradation** - System continues working even if components fail

## 🏗️ Architecture

```
┌─────────────────┐
│   Dashboard     │ ← User Interface (HTML/JS)
│ (Web Browser)   │
└────────┬────────┘
         │ HTTP/WebSocket
         ↓
┌─────────────────┐
│    main.py      │ ← FastAPI Backend (API Layer)
│  (FastAPI)      │
└────────┬────────┘
         │ Delegates to
         ↓
┌─────────────────┐
│traffic_controller│ ← Business Logic (Central Coordination)
│      .py        │   • ML Prediction
└────────┬────────┘   • File Reading
         ↑            • Signal Timing
         │
    ┌────┴────┐
    │         │
    ↓         ↓
┌─────┐  ┌──────────────┐
│ ML  │  │vehicle_count │ ← CV Output
│Models│  │  _south.txt  │
└─────┘  └──────┬───────┘
                ↑
         ┌──────┴───────┐
         │vehicle_      │ ← YOLO Detection
         │detection.py  │
         └──────────────┘
```

## 📦 Components

### 1. **traffic_dashboard.html** - Web Interface
- Real-time traffic visualization with animated intersection
- Manual control sliders for all four directions
- Computer vision mode activation
- Image upload for YOLO detection
- Live countdown timer and phase indicator
- WebSocket-based real-time updates

### 2. **main.py** - FastAPI Backend
- RESTful API endpoints for dashboard
- WebSocket server for real-time updates
- Coordinates between dashboard and traffic controller
- Manages background tasks (simulation, CV polling)
- Handles image upload and YOLO detection

### 3. **traffic_controller.py** - Central Coordination Layer
- Reads vehicle count data from file
- Manages ML model initialization
- Predicts signal timings (ML or formula fallback)
- Handles ambulance override logic
- Provides clean API for main.py

### 4. **vehicle_detection.py** - Computer Vision
- YOLOv8-based vehicle detection
- Counts vehicles by type (car, motorcycle, bus, truck)
- Detects ambulances for emergency override
- Writes counts to `vehicle_count_south.txt`
- Displays annotated video feed

### 5. **trainthemodel.py** - ML Model Training
- Trains TensorFlow models for each direction
- Uses time of day, day of week, and vehicle counts
- Outputs trained models (.h5 files)
- Includes StandardScaler for feature normalization

### 6. **datasetcreate.py** - Dataset Generation
- Generates synthetic training data
- Creates realistic traffic scenarios
- Saves dataset to CSV for model training

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam or IP camera (for CV mode)
- Modern web browser

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/smart-traffic-optimizer.git
   cd smart-traffic-optimizer
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLOv8 model** (if not included):
   ```bash
   # The yolov8n.pt file should be in the project root
   # If missing, it will be downloaded automatically on first run
   ```

### Training the Models

1. **Generate training dataset:**
   ```bash
   python datasetcreate.py
   ```
   This creates `traffic_signal_data_directions.csv`

2. **Train ML models:**
   ```bash
   python trainthemodel.py
   ```
   This creates four model files:
   - `traffic_signal_model_north.h5`
   - `traffic_signal_model_south.h5`
   - `traffic_signal_model_east.h5`
   - `traffic_signal_model_west.h5`

### Running the System

1. **Start the FastAPI server:**
   ```bash
   uvicorn main:app --reload --port 8000
   ```

2. **Open the dashboard:**
   ```
   http://localhost:8000
   ```

3. **(Optional) Start vehicle detection for CV mode:**
   ```bash
   python vehicle_detection.py
   ```
   Or click "Auto CV" in the dashboard to launch it automatically.

## 🎮 Usage

### Manual Mode

1. Adjust the sliders for each direction (North, South, East, West)
2. Click **"▶ Optimize (Manual)"** to calculate timings
3. Watch the simulation run with optimized signal timings

### Computer Vision Mode (Hybrid)

1. Click **"Auto CV"** to start vehicle detection
2. South direction count comes from camera feed
3. Adjust North, East, West sliders manually
4. System combines CV and manual input for optimization

### Image Upload Mode

1. Click **"Upload Image"** and select a traffic image
2. YOLO detects vehicles in the image
3. System updates the selected direction's count
4. Timings are recalculated automatically

### Ambulance Detection

- When an ambulance is detected (via CV or image upload):
  - 🚨 Emergency override activates
  - South direction gets 120s green time
  - Other directions get 0s green time
  - Red ambulance badge appears in dashboard

## 📡 API Endpoints

### GET `/`
Returns the dashboard HTML interface

### GET `/api/info`
Returns system information (ML status, YOLO status)

### GET `/api/state`
Returns current system state (vehicle counts, timings, mode)

### POST `/api/update`
Manual mode optimization with vehicle counts

### POST `/api/set_counts`
Update vehicle counts without starting simulation

### POST `/api/cv/start`
Start computer vision mode (hybrid input)

### POST `/api/detect/upload`
Upload image for YOLO detection

### POST `/api/reset`
Reset system to idle state

### WebSocket `/ws`
Real-time state updates and countdown timer

## ⚙️ Configuration

### Camera Configuration
Edit `vehicle_detection.py`:
```python
# For webcam
cap = cv2.VideoCapture(0)

# For IP camera (ESP32-CAM)
cap = cv2.VideoCapture("http://192.168.1.100:81/stream")
```

### ML Model Settings
Edit `traffic_controller.py`:
```python
# Green time constraints (seconds)
MIN_GREEN = 10
MAX_GREEN = 90

# Ambulance override time
AMBULANCE_GREEN = 120
```

### Server Configuration
```bash
# Change port
uvicorn main:app --port 8080

# Enable auto-reload for development
uvicorn main:app --reload

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 🧪 Testing

The system includes graceful degradation:

- **ML models missing?** → Uses formula-based fallback
- **YOLO unavailable?** → Image upload disabled, CV mode still works
- **Camera disconnected?** → Manual mode continues working
- **WebSocket drops?** → Auto-reconnects every 3 seconds

## 📊 System Requirements

- **CPU**: Multi-core processor (for YOLO inference)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 500MB for models and dependencies
- **Camera**: 720p or higher for best detection accuracy
- **Browser**: Chrome, Firefox, Safari, or Edge (latest versions)

## 🔧 Troubleshooting

### ML models not loading
```bash
# Retrain the models
python trainthemodel.py
```

### YOLO detection not working
```bash
# Check if yolov8n.pt exists
ls -la yolov8n.pt

# Reinstall ultralytics
pip install --upgrade ultralytics
```

### WebSocket connection fails
- Check if port 8000 is available
- Ensure firewall allows WebSocket connections
- Try a different browser

### Camera not detected
```bash
# List available cameras (Linux)
ls /dev/video*

# Test camera with OpenCV
python -c "import cv2; print(cv2.VideoCapture(0).read())"
```

## 🛣️ Roadmap

- [ ] Multi-intersection coordination
- [ ] Historical data analysis and reporting
- [ ] Pedestrian detection and crosswalk timing
- [ ] Weather-based timing adjustments
- [ ] Mobile app for remote monitoring
- [ ] Cloud deployment support
- [ ] Advanced ML models (reinforcement learning)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributors

- [Gokulakrishnan K](https://github.com/Gokulakrishnan610)

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [TensorFlow](https://www.tensorflow.org/) - Machine learning
- [OpenCV](https://opencv.org/) - Computer vision

## 📞 Support

For issues, questions, or contributions, please open an issue on GitHub.

---

**Made with ❤️ for smarter cities**
