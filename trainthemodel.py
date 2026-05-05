import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models

def train_models():
    # Load the dataset
    if not os.path.exists('traffic_signal_data_directions.csv'):
        print("Dataset not found!")
        return False
        
    df = pd.read_csv('traffic_signal_data_directions.csv')

    # Define features and targets for each direction
    X = df[['time_of_day', 'day_of_week', 'vehicle_count_north', 'vehicle_count_south', 'vehicle_count_east', 'vehicle_count_west']]

    # Split features and green light time targets for each direction
    y_north = df['green_light_north']
    y_south = df['green_light_south']
    y_east = df['green_light_east']
    y_west = df['green_light_west']

    # Split data into training and test sets for each direction
    X_train, X_test, y_train_north, y_test_north = train_test_split(X, y_north, test_size=0.2, random_state=42)
    _, _, y_train_south, y_test_south = train_test_split(X, y_south, test_size=0.2, random_state=42)
    _, _, y_train_east, y_test_east = train_test_split(X, y_east, test_size=0.2, random_state=42)
    _, _, y_train_west, y_test_west = train_test_split(X, y_west, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define a model architecture
    def create_model():
        model = models.Sequential([
            layers.Input(shape=(X_train_scaled.shape[1],)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)  # Output is the predicted green light time
        ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model

    # Train separate models for each direction
    model_north = create_model()
    model_south = create_model()
    model_east = create_model()
    model_west = create_model()

    # Train the models
    model_north.fit(X_train_scaled, y_train_north, epochs=100, batch_size=16, validation_split=0.2, verbose=1)
    model_south.fit(X_train_scaled, y_train_south, epochs=100, batch_size=16, validation_split=0.2, verbose=1)
    model_east.fit(X_train_scaled, y_train_east, epochs=100, batch_size=16, validation_split=0.2, verbose=1)
    model_west.fit(X_train_scaled, y_train_west, epochs=100, batch_size=16, validation_split=0.2, verbose=1)

    # Save the trained models
    model_north.save('traffic_signal_model_north.h5')
    model_south.save('traffic_signal_model_south.h5')
    model_east.save('traffic_signal_model_east.h5')
    model_west.save('traffic_signal_model_west.h5')
    print("Models saved.")
    return True

if __name__ == '__main__':
    train_models()
