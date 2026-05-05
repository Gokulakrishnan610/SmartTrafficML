"""Traffic Controller Module

This module serves as the central coordination layer for traffic signal optimization logic.
It provides functions for reading vehicle counts from files, predicting signal timings using
ML models or formula fallback, and handling ambulance emergency overrides.

The module can be used both as an imported module (by main.py) and as a standalone script
for testing ML models independently.

Key Functions:
    - init_ml(): Initialize TensorFlow ML models for signal timing prediction
    - get_vehicle_count_south(): Read vehicle count and ambulance status from file
    - predict_signal_times(): Predict optimal signal timings for all four directions
    - get_user_input(): Console input for standalone mode
    - main(): Standalone execution entry point

Global Variables:
    - ml_initialized: Boolean flag indicating whether ML models loaded successfully
"""
import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ── Global ML Models ───────────────────────────────────────────────────────────
# These models are initialized once at module load time by init_ml().
# If initialization succeeds, ml_initialized is set to True and predictions use ML.
# If initialization fails, ml_initialized is False and predictions use formula fallback.
# Global ML models
model_north = None
model_south = None
model_east = None
model_west = None
scaler = None
ml_initialized = False

def init_ml():
    """Initialize ML models for traffic signal timing prediction.
    
    Loads TensorFlow models for all four directions (north, south, east, west) and
    initializes the StandardScaler using training data. Sets the global ml_initialized
    flag to indicate whether ML models are available.
    
    Returns:
        bool: True if ML models loaded successfully, False otherwise
        
    Side Effects:
        - Sets global variables: model_north, model_south, model_east, model_west, scaler
        - Sets global ml_initialized flag to True on success, False on failure
        - Prints warning messages to console if initialization fails
        
    Notes:
        - Only loads models once (idempotent - subsequent calls return immediately)
        - Requires traffic_signal_model_*.h5 files in the current directory
        - Requires traffic_signal_data_directions.csv for scaler initialization
        - If initialization fails, predict_signal_times() will use formula fallback
    """
    global model_north, model_south, model_east, model_west, scaler, ml_initialized
    try:
        import tensorflow as tf
        from sklearn.preprocessing import StandardScaler
        
        # Only load if not already loaded
        if ml_initialized:
            return True

        model_north = tf.keras.models.load_model('traffic_signal_model_north.h5', compile=False)
        model_south = tf.keras.models.load_model('traffic_signal_model_south.h5', compile=False)
        model_east = tf.keras.models.load_model('traffic_signal_model_east.h5', compile=False)
        model_west = tf.keras.models.load_model('traffic_signal_model_west.h5', compile=False)

        # Load and fit scaler using training data
        # The scaler normalizes input features to improve ML model accuracy
        if os.path.exists('traffic_signal_data_directions.csv'):
            df = pd.read_csv('traffic_signal_data_directions.csv')
            X = df[['time_of_day', 'day_of_week', 'vehicle_count_north', 'vehicle_count_south', 'vehicle_count_east', 'vehicle_count_west']]
            scaler = StandardScaler()
            scaler.fit(X)
            ml_initialized = True
            return True
        else:
            print("⚠️ traffic_signal_data_directions.csv not found.")
            return False
    except Exception as e:
        print(f"⚠️ ML init failed in traffic_controller: {e}")
        return False

# Initialize ML models at module load time
# If this fails, the system will gracefully fall back to formula-based prediction
init_ml()

def get_vehicle_count_south():
    """Read vehicle count and ambulance status from vehicle_count_south.txt.
    
    Parses the vehicle count file written by vehicle_detection.py to extract the
    number of vehicles detected in the South direction and whether an ambulance
    was detected.
    
    Returns:
        tuple[int, bool]: A tuple containing:
            - int: Vehicle count (0 if file missing/malformed)
            - bool: Ambulance detected status (False if file missing/malformed)
            
    Error Handling Strategy:
        - Returns (0, False) if file does not exist
        - Returns (0, False) if file is empty
        - Returns (0, False) if file content is malformed
        - Never raises exceptions - graceful degradation allows system to continue
        - Errors are silently handled; calling code (main.py) manages the zero count
        
    Expected File Format:
        Vehicle<count>,<timestamp> ambulance <true|false>
        Example: "Vehicle15,14:23:45 ambulance false"
        
    Notes:
        - This function provides idempotent reads (reading twice returns same values)
        - Used by both standalone mode and FastAPI backend
        - File is written by vehicle_detection.py during computer vision mode
    """
    try:
        with open("vehicle_count_south.txt", 'r') as f:
            content = f.read().strip()
            if not content:
                return 0, False
            parts = content.split()
            vehicle_info = parts[0]
            vehicle_count = int(vehicle_info.split(',')[0].replace("Vehicle", ""))
            ambulance_status = parts[-1].lower() == 'true' if len(parts) > 1 else False
            return vehicle_count, ambulance_status
    except Exception as e:
        # Error handling: Return safe defaults without crashing
        # This allows the system to continue operating even if CV component fails
        return 0, False

def get_user_input():
    """Get vehicle count input from console for standalone mode.
    
    Prompts the user to enter vehicle counts for North, East, and West directions
    via console input. Reads the South direction count from vehicle_count_south.txt.
    Automatically detects ambulance status from the South direction file.
    
    Returns:
        tuple[pd.DataFrame, Optional[str]]: A tuple containing:
            - pd.DataFrame: User data with columns:
                - time_of_day: float (current hour + minute/60)
                - day_of_week: int (1-7, Monday-Sunday)
                - vehicle_count_north: float
                - vehicle_count_south: float (from file)
                - vehicle_count_east: float
                - vehicle_count_west: float
            - Optional[str]: Ambulance direction ("south" if detected, None otherwise)
            
    Notes:
        - Used only when traffic_controller.py is run as a standalone script
        - Automatically includes current time and day of week for ML prediction
        - South direction data always comes from vehicle_count_south.txt
    """
    now = datetime.now()
    time_of_day = now.hour + now.minute / 60
    day_of_week = now.isoweekday()

    vehicle_count_north = float(input("Enter vehicle count (north): "))
    vehicle_count_south, ambulance_detected = get_vehicle_count_south()
    vehicle_count_east = float(input("Enter vehicle count (east): "))
    vehicle_count_west = float(input("Enter vehicle count (west): "))

    ambulance_direction = "south" if ambulance_detected else None

    user_data = pd.DataFrame({
        'time_of_day': [time_of_day],
        'day_of_week': [day_of_week],
        'vehicle_count_north': [vehicle_count_north],
        'vehicle_count_south': [vehicle_count_south],
        'vehicle_count_east': [vehicle_count_east],
        'vehicle_count_west': [vehicle_count_west]
    })

    return user_data, ambulance_direction

def predict_signal_times(user_data, ambulance_direction):
    """Predict optimal signal timings for all four traffic directions.
    
    Uses ML models (if available) or proportional formula fallback to predict green
    light durations for each direction. Handles ambulance emergency override logic.
    
    Parameters:
        user_data (pd.DataFrame): DataFrame with one row containing:
            - time_of_day: float (hour + minute/60, range 0.0-23.99)
            - day_of_week: int (1-7, Monday-Sunday)
            - vehicle_count_north: float (0-99)
            - vehicle_count_south: float (0-99)
            - vehicle_count_east: float (0-99)
            - vehicle_count_west: float (0-99)
        ambulance_direction (Optional[str]): Direction of ambulance detection
            ("north", "south", "east", "west", or None)
            
    Returns:
        dict: Signal timings for all directions with structure:
            {
                "north": {"green": int, "yellow": int, "red": int},
                "south": {"green": int, "yellow": int, "red": int},
                "east": {"green": int, "yellow": int, "red": int},
                "west": {"green": int, "yellow": int, "red": int}
            }
            All times are in seconds.
            
    Behavior:
        - If ml_initialized is True: Uses TensorFlow models for prediction
        - If ml_initialized is False: Uses proportional formula fallback
        - If ambulance_direction is set: Overrides with emergency timing
          (120s green for ambulance direction, 0s for others)
        - Green times are clamped to [10, 90] seconds (except ambulance override)
        - Yellow time is fixed at 5 seconds for all directions
        - Red time is calculated as 120 - (green + yellow) seconds
        
    Formula Fallback:
        green_time = 30 + (vehicle_count / max_vehicles) * 60
        Ensures proportional allocation based on relative traffic density.
        
    Notes:
        - South direction gets +5s buffer time (except during ambulance override)
        - Total cycle time is 120 seconds per direction
        - Ambulance override takes precedence over all other logic
    """
    yellow = 5.0
    total_cycle = 120.0

    # ML Prediction Path: Use TensorFlow models if initialized successfully
    if ml_initialized and scaler is not None:
        scaled = scaler.transform(user_data)
        g_north = float(model_north.predict(scaled, verbose=0)[0][0])
        g_south = float(model_south.predict(scaled, verbose=0)[0][0])
        g_east = float(model_east.predict(scaled, verbose=0)[0][0])
        g_west = float(model_west.predict(scaled, verbose=0)[0][0])
    else:
        # Formula Fallback Path: Use proportional allocation when ML unavailable
        # This ensures the system continues to function even without ML models
        max_vehicles = max(1, user_data.iloc[0]['vehicle_count_north'], user_data.iloc[0]['vehicle_count_south'], 
                              user_data.iloc[0]['vehicle_count_east'], user_data.iloc[0]['vehicle_count_west'])
        g_north = 30 + (user_data.iloc[0]['vehicle_count_north'] / max_vehicles) * 60
        g_south = 30 + (user_data.iloc[0]['vehicle_count_south'] / max_vehicles) * 60
        g_east = 30 + (user_data.iloc[0]['vehicle_count_east'] / max_vehicles) * 60
        g_west = 30 + (user_data.iloc[0]['vehicle_count_west'] / max_vehicles) * 60

    # Handle ambulance override for any direction
    # Emergency override: Ambulance direction gets full 120s green, all others get 0s
    if ambulance_direction:
        print(f"\n🚨 Ambulance detected on {ambulance_direction.upper()} side. Giving full green cycle to {ambulance_direction} direction.")
        if ambulance_direction == 'north':
            g_north = total_cycle
            g_south = g_east = g_west = 0
        elif ambulance_direction == 'south':
            g_south = total_cycle
            g_north = g_east = g_west = 0
        elif ambulance_direction == 'east':
            g_east = total_cycle
            g_north = g_south = g_west = 0
        elif ambulance_direction == 'west':
            g_west = total_cycle
            g_north = g_south = g_east = 0
    else:
        # No ambulance - apply buffer time and clamping
        # South gets +5s buffer time to account for CV detection latency
        g_south += 5.0  # Buffer time for south
        # Clamp green times to realistic range [10, 90] seconds
        g_north = min(90.0, max(10.0, g_north))
        g_south = min(90.0, max(10.0, g_south))
        g_east = min(90.0, max(10.0, g_east))
        g_west = min(90.0, max(10.0, g_west))

    return {
        "north": {"green": int(g_north), "yellow": int(yellow), "red": int(total_cycle - (g_north + yellow))},
        "south": {"green": int(g_south), "yellow": int(yellow), "red": int(total_cycle - (g_south + yellow))},
        "east": {"green": int(g_east), "yellow": int(yellow), "red": int(total_cycle - (g_east + yellow))},
        "west": {"green": int(g_west), "yellow": int(yellow), "red": int(total_cycle - (g_west + yellow))}
    }



def main():
    """Main entry point for standalone execution.
    
    Runs the traffic controller as a standalone script for testing ML models
    independently of the web interface. Prompts for vehicle counts via console,
    predicts signal timings, and displays results.
    
    Workflow:
        1. Prompt user for North, East, and West vehicle counts
        2. Read South vehicle count from vehicle_count_south.txt
        3. Detect ambulance status from file
        4. Predict signal timings using ML or formula fallback
        5. Display predicted timings to console
        
    Notes:
        - Only executed when script is run directly (not imported)
        - Useful for testing ML model predictions without starting web server
        - Requires vehicle_count_south.txt for South direction data
    """
    user_data, ambulance_direction = get_user_input()
    timings = predict_signal_times(user_data, ambulance_direction)
    print("Final timings:", timings)

if __name__ == "__main__":
    main()
