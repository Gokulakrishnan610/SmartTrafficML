import pandas as pd
import numpy as np

# Generate random synthetic data
np.random.seed(42)

# Create a dataset with random vehicle counts and times
data = {
    'time_of_day': np.random.randint(0, 24, 1000),
    'day_of_week': np.random.randint(1, 8, 1000),
    'vehicle_count_north': np.random.randint(0, 100, 1000),
    'vehicle_count_south': np.random.randint(0, 100, 1000),
    'vehicle_count_east': np.random.randint(0, 100, 1000),
    'vehicle_count_west': np.random.randint(0, 100, 1000)
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Define green light duration for each direction (30 to 90 seconds based on vehicle count)
df['green_light_north'] = 30 + (df['vehicle_count_north'] / df['vehicle_count_north'].max()) * 60
df['green_light_south'] = 30 + (df['vehicle_count_south'] / df['vehicle_count_south'].max()) * 60
df['green_light_east'] = 30 + (df['vehicle_count_east'] / df['vehicle_count_east'].max()) * 60
df['green_light_west'] = 30 + (df['vehicle_count_west'] / df['vehicle_count_west'].max()) * 60

# Assign fixed yellow light duration (3 to 6 seconds)
df['yellow_light_north'] = 5  # Fixed yellow light
df['yellow_light_south'] = 5
df['yellow_light_east'] = 5
df['yellow_light_west'] = 5

# Calculate red light time as the remainder of a cycle (assuming a fixed total cycle of 120 seconds)
df['red_light_north'] = 120 - (df['green_light_north'] + df['yellow_light_north'] + df['green_light_south'] + df['green_light_east'] + df['green_light_west'])
df['red_light_south'] = 120 - (df['green_light_south'] + df['yellow_light_south'] + df['green_light_north'] + df['green_light_east'] + df['green_light_west'])
df['red_light_east'] = 120 - (df['green_light_east'] + df['yellow_light_east'] + df['green_light_north'] + df['green_light_south'] + df['green_light_west'])
df['red_light_west'] = 120 - (df['green_light_west'] + df['yellow_light_west'] + df['green_light_north'] + df['green_light_south'] + df['green_light_east'])

# Save dataset to CSV
df.to_csv('traffic_signal_data_directions.csv', index=False)
print("Dataset saved to 'traffic_signal_data_directions.csv'")
