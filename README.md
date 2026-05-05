# IOT - Smart Traffic Management System

A comprehensive traffic management solution that uses computer vision, machine learning, and IoT to optimize traffic flow at intersections. The system detects vehicles, identifies ambulances, and dynamically adjusts traffic signal timings based on real-time traffic conditions.

## Overview

This project integrates several components to create an intelligent traffic signal control system:

1. **Computer Vision System**: Uses YOLOv8 to detect and count vehicles in real-time from traffic cameras
2. **Machine Learning Model**: Predicts optimal traffic signal timings based on traffic density and patterns
3. **ESP32 Controller**: Hardware controller that manages the physical traffic signals
4. **Emergency Vehicle Priority**: Automatically detects ambulances and gives them priority

## Components

### 1. Vehicle Detection (`vehicle_detection.py`)

This module uses YOLOv8 to detect and count vehicles from camera feeds:

- Counts vehicles by type (car, motorcycle, bus, truck, bicycle)
- Detects ambulances and triggers emergency protocols
- Provides visual monitoring with bounding boxes around detected vehicles
- Outputs vehicle counts to a text file for the controller to use

### 2. Traffic Controller (`traffic_controller.py`)

The central intelligence of the system:

- Reads vehicle counts from all directions
- Uses pre-trained machine learning models to predict optimal signal timings
- Implements emergency vehicle override protocol
- Sends timing commands to the ESP32 hardware controller

### 3. Model Training (`trainthemodel.py`)

Creates and trains the machine learning models that determine optimal signal timings:

- Trains separate models for each direction (north, south, east, west)
- Uses features like time of day, day of week, and vehicle counts
- Outputs trained models in .h5 format

### 4. Dataset Creation (`datasetcreate.py`)

Generates synthetic training data for the ML models:

- Creates random vehicle count scenarios
- Generates corresponding signal timing data
- Saves dataset to CSV for model training

### 5. ESP32 Controller (ESP32 Code)

Hardware controller that interfaces with physical traffic signals:

- Controls traffic lights for all four directions
- Receives timing updates over WiFi
- Implements timing sequences for each direction

## Setup Instructions

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- OpenCV
- Ultralytics YOLOv8
- ESP32 with Arduino IDE

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/smart-traffic-management.git
   cd smart-traffic-management
   ```

2. Install Python dependencies:
   ```
   pip install ultralytics opencv-python tensorflow scikit-learn pandas numpy requests
   ```

3. Download the YOLOv8 model:
   ```
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
   ```

4. Flash the ESP32 code using Arduino IDE:
   - Install ESP32 board support in Arduino IDE
   - Install required libraries: AsyncTCP, ESPAsyncWebServer, ArduinoJson
   - Update WiFi credentials in the code
   - Upload to your ESP32

### Hardware Setup

Connect the ESP32 pins to traffic light LEDs as specified in the code:

- North: Green (14), Yellow (27), Red (26)
- South: Green (25), Yellow (33), Red (32)
- East: Green (18), Yellow (19), Red (21)
- West: Green (22), Yellow (23), Red (4)

### Running the System

1. First create the dataset and train the models:
   ```
   python datasetcreate.py
   python trainthemodel.py
   ```

2. Start the vehicle detection system:
   ```
   python vehicle_detection.py
   ```

3. Start the traffic controller (in a separate terminal):
   ```
   python traffic_controller.py
   ```

## Configuration

- Update the camera URL in `vehicle_detection.py` to point to your traffic cameras
- Adjust the ESP32_SERVER_URL in `traffic_controller.py` to match your ESP32's IP address
- Modify confidence thresholds in `vehicle_detection.py` for object detection sensitivity

## Future Improvements

- Add pedestrian detection and crosswalk timing
- Implement historical data analysis for predictive traffic management
- Develop a web dashboard for monitoring traffic flow
- Support multiple intersections with coordination
- Add more sophisticated emergency vehicle detection

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributors

- [Gokulakrishnan K](https://github.com/Gokulakrishnan610)
- [Jayaprakash A](https://github.com/A-Jayaprakash)
- [GokulaSarathy S](https://github.com/Gokulasarathy)
  
## Acknowledgments

- YOLOv8 by Ultralytics
- TensorFlow and Keras teams
- ESP32 and Arduino communities
