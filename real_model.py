import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import datetime


# Load data from CSV file
print("Loading data...")
df = pd.read_csv('smart_home_data.csv')

# Convert dates to useful features
df['month'] = pd.to_datetime(df['date']).dt.month
df['day_of_week_num'] = pd.to_datetime(df['date']).dt.dayofweek
df['hour'] = pd.to_datetime(df['date']).dt.hour
df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)

# Convert text variables to numbers
day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
               'Friday': 4, 'Saturday': 5, 'Sunday': 6}
df['day_of_week'] = df['day_of_week'].map(day_mapping)

time_mapping = {'morning': 0, 'afternoon': 1, 'evening': 2, 'night': 3}
df['time_of_day'] = df['time_of_day'].map(time_mapping)

condition_mapping = {'sleeping': 0, 'at_work': 1, 'at_home': 2, 
                    'out': 3, 'getting_ready': 4, 'awake': 5}
df['person_condition'] = df['person_condition'].map(condition_mapping)

mood_mapping = {'peaceful': 0, 'focused': 1, 'tired': 2, 'stressed': 3, 
               'happy': 4, 'calm': 5, 'energetic': 6}
df['mood'] = df['mood'].map(mood_mapping)

def handle_unknown_values(input_value, value_mapping, default_value=None):
    """
    Handle unknown values by mapping them to a default value or the closest known value
    """
    if input_value in value_mapping:
        return value_mapping[input_value]
    
    # If default value is provided, use it
    if default_value is not None:
        return value_mapping.get(default_value, 0)
    
    # If no default, find the closest value by string similarity
    from difflib import get_close_matches
    matches = get_close_matches(input_value, value_mapping.keys(), n=1)
    if matches:
        return value_mapping[matches[0]]
    
    # If no close matches found, return a neutral value (middle of the range)
    return len(value_mapping) // 2

def analyze_data():
    print("\nData Analysis:")
    print("\n1. Person condition distribution by time of day:")
    print(pd.crosstab(df['time_of_day'], df['person_condition']))
    
    print("\n2. Average temperature by month:")
    print(df.groupby('month')['temperature'].mean())

# Prepare input features
input_features = ['mood', 'person_condition', 'time_of_day', 'at_home', 'is_holiday']
X = df[input_features]

# Prepare output features
device_columns = ['tv_status', 'smart_locks', 'security_cameras', 
                 'security_system', 'water_heater_status']
light_columns = ['bedroom_light', 'living_room_light', 'bathroom_light', 'kitchen_light']

# Combine all outputs
y_devices = df[device_columns]
y_lights = df[light_columns]
y_music = df['music_type']

# Create and train models
device_model = RandomForestClassifier(n_estimators=100, random_state=42)
light_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
music_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train models
device_model.fit(X, y_devices)
light_model.fit(X, y_lights)

# For music type, we need to encode the categorical values
music_encoder = LabelEncoder()
y_music_encoded = music_encoder.fit_transform(y_music)
music_model.fit(X, y_music_encoded)

def predict_home_automation(mood, person_condition, time_of_day, at_home, is_holiday):
    """
    Predict home automation settings based on input parameters
    """
    try:
        # Prepare input data with handling for unknown values
        input_data = np.array([[
            handle_unknown_values(mood, mood_mapping, 'happy'),
            handle_unknown_values(person_condition, condition_mapping, 'at_home'),
            handle_unknown_values(time_of_day, time_mapping, 'evening'),
            at_home,
            is_holiday
        ]])
        
        # Make predictions
        device_predictions = device_model.predict(input_data)[0]
        light_predictions = light_model.predict(input_data)[0]
        music_prediction = music_encoder.inverse_transform(music_model.predict(input_data))[0]
        
        # Create recommendations
        recommendations = []
        
        # Device recommendations
        devices = dict(zip(device_columns, device_predictions))
        for device, status in devices.items():
            device_name = device.replace('_', ' ').title()
            if status == 1:
                recommendations.append(f"Turn ON {device_name}")
            else:
                recommendations.append(f"Turn OFF {device_name}")
        
        # Light recommendations with brightness levels
        lights = dict(zip(light_columns, light_predictions))
        
        # MODIFIED HERE: Force all lights to be off (brightness 0) when person is sleeping
        is_sleeping = person_condition == 'sleeping'
        
        for light, brightness in lights.items():
            room = light.replace('_light', '').replace('_', ' ').title()
            # Ensure brightness is between 0 and 100
            brightness = max(0, min(100, int(brightness)))
            
            if at_home == 0 or is_sleeping:
                brightness = 0
            
            recommendations.append(f"Set {room} Light brightness to {brightness}%")
        
        # Music recommendation
        if at_home == 1 and not is_sleeping:
            recommendations.append(f"Suggested Music Type: {music_prediction}")
        else:
            recommendations.append("Music: NONE (Not at home or sleeping)")
        
        return recommendations
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return None

def evaluate_models():
    """
    Evaluate model performance
    """
    try:
        # Split data for evaluation
        X_train, X_test, y_devices_train, y_devices_test, y_lights_train, y_lights_test, y_music_train, y_music_test = train_test_split(
            X, y_devices, y_lights, y_music, test_size=0.2, random_state=42
        )
        
        print("\nModel Evaluation:")
        print("-" * 50)
        
        # Evaluate device predictions
        device_predictions = device_model.predict(X_test)
        print("\nDevice Control Accuracy:")
        for i, col in enumerate(device_columns):
            accuracy = accuracy_score(y_devices_test[col], device_predictions[:, i])
            print(f"{col.replace('_', ' ').title()}: {accuracy:.2f}")
        
        # Evaluate light predictions using MSE
        light_predictions = light_model.predict(X_test)
        print("\nLight Control Mean Squared Error:")
        for i, col in enumerate(light_columns):
            mse = mean_squared_error(y_lights_test[col], light_predictions[:, i])
            print(f"{col.replace('_', ' ').title()}: {mse:.2f}")
        
        # Evaluate music predictions
        y_music_test_encoded = music_encoder.transform(y_music_test)
        music_predictions = music_model.predict(X_test)
        music_accuracy = accuracy_score(y_music_test_encoded, music_predictions)
        print(f"\nMusic Type Prediction Accuracy: {music_accuracy:.2f}")
    except Exception as e:
        print(f"Error evaluating models: {str(e)}")

def display_model_accuracy():
    """
    Display model accuracy metrics in a clear format
    """
    # Split data for evaluation
    X_train, X_test, y_devices_train, y_devices_test, y_lights_train, y_lights_test, y_music_train, y_music_test = train_test_split(
        X, y_devices, y_lights, y_music, test_size=0.2, random_state=42
    )
    
    print("\n=== Model Accuracy Report ===")
    print("-" * 50)
    
    # Device predictions accuracy
    device_predictions = device_model.predict(X_test)
    print("\nDevice Control Accuracy:")
    for i, col in enumerate(device_columns):
        accuracy = accuracy_score(y_devices_test[col], device_predictions[:, i])
        print(f"{col.replace('_', ' ').title()}: {accuracy:.2f}")
    
    # Light predictions accuracy
    light_predictions = light_model.predict(X_test)
    print("\nLight Control Accuracy:")
    for i, col in enumerate(light_columns):
        accuracy = accuracy_score(y_lights_test[col], light_predictions[:, i])
        print(f"{col.replace('_', ' ').title()}: {accuracy:.2f}")
    
    # Music predictions accuracy
    y_music_test_encoded = music_encoder.transform(y_music_test)
    music_predictions = music_model.predict(X_test)
    music_accuracy = accuracy_score(y_music_test_encoded, music_predictions)
    print("\nMusic Type Prediction Accuracy:")
    print(f"Accuracy: {music_accuracy:.2f}")

def validate_input(current_time, current_date, person_condition, at_home, 
                  is_holiday, guests_present, mood, temperature, light_level):
    """
    Validate inputs before prediction
    """
    try:
        # Validate date and time
        if not isinstance(current_time, datetime.datetime):
            raise ValueError("current_time must be datetime type")
        if not isinstance(current_date, datetime.date):
            raise ValueError("current_date must be date type")
            
        # Validate person condition
        if person_condition not in condition_mapping:
            raise ValueError(f"Invalid person condition. Allowed values: {list(condition_mapping.keys())}")
            
        # Validate mood
        if mood not in mood_mapping:
            raise ValueError(f"Invalid mood. Allowed values: {list(mood_mapping.keys())}")
            
        # Validate numeric values
        if not isinstance(at_home, (int, bool)) or at_home not in [0, 1]:
            raise ValueError("at_home must be 0 or 1")
        if not isinstance(is_holiday, (int, bool)) or is_holiday not in [0, 1]:
            raise ValueError("is_holiday must be 0 or 1")
        if not isinstance(guests_present, (int, bool)) or guests_present not in [0, 1]:
            raise ValueError("guests_present must be 0 or 1")
            
        # Validate temperature and light level
        if not isinstance(temperature, (int, float)) or temperature < -10 or temperature > 50:
            raise ValueError("Temperature must be between -10 and 50")
        if not isinstance(light_level, (int, float)) or light_level < 0 or light_level > 100:
            raise ValueError("Light level must be between 0 and 100")
            
        return True
        
    except Exception as e:
        print(f"Input validation error: {str(e)}")
        return False

def track_model_performance(predictions, actual_values, timestamp=None):
    """
    Track model performance and save results
    """
    if timestamp is None:
        timestamp = datetime.datetime.now()
    
    performance_data = {
        'timestamp': timestamp,
        'device_accuracy': {},
        'light_accuracy': {},
        'music_accuracy': {}
    }
    
    # Calculate device predictions accuracy
    for i, col in enumerate(device_columns):
        accuracy = accuracy_score(actual_values[col], predictions[:, i])
        performance_data['device_accuracy'][col] = accuracy
    
    # Calculate light predictions accuracy
    for i, col in enumerate(light_columns):
        accuracy = accuracy_score(actual_values[col], predictions[:, i])
        performance_data['light_accuracy'][col] = accuracy
    
    # Calculate music predictions accuracy
    y_music_test_encoded = music_encoder.transform(actual_values['music_type'])
    music_predictions = predictions[:, -1]
    music_accuracy = accuracy_score(y_music_test_encoded, music_predictions)
    performance_data['music_accuracy'] = music_accuracy
    
    # Save performance data
    try:
        performance_history = pd.read_csv('model_performance_history.csv')
    except FileNotFoundError:
        performance_history = pd.DataFrame(columns=['timestamp'] + 
                                        [f'device_accuracy_{col}' for col in device_columns] +
                                        [f'light_accuracy_{col}' for col in light_columns] +
                                        ['music_accuracy'])
    
    new_row = {
        'timestamp': timestamp
    }
    new_row.update({f'device_accuracy_{k}': v for k, v in performance_data['device_accuracy'].items()})
    new_row.update({f'light_accuracy_{k}': v for k, v in performance_data['light_accuracy'].items()})
    new_row['music_accuracy'] = performance_data['music_accuracy']
    
    performance_history = performance_history.append(new_row, ignore_index=True)
    performance_history.to_csv('model_performance_history.csv', index=False)
    
    return performance_data

def train_models():
    print("\nTraining models...")
    try:
        device_model.fit(X, y_devices)
        light_model.fit(X, y_lights)
        music_model.fit(X, y_music_encoded)
        print("Models trained successfully!")
        return True
    except Exception as e:
        print(f"Error training models: {str(e)}")
        return False

def load_models():
    """
    Load trained models from files
    """
    global device_model, light_model, music_model
    try:
        print("Loading and preprocessing data...")
        analyze_data()
        
        print("\nTraining models...")
        if train_models():
            print("Models loaded successfully!")
            return True
        else:
            print("Failed to load models!")
            return False
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        print("Starting program...")
        analyze_data()
        
        if train_models():
            evaluate_models()
            
            print("\nMaking sample prediction...")
            sample_prediction = predict_home_automation(
                mood='happy',
                person_condition='sleeping',  # Testing the sleeping condition
                time_of_day='night',
                at_home=1,
                is_holiday=0
            )
            
            if sample_prediction:
                print("\nRecommended Actions:")
                for i, action in enumerate(sample_prediction, 1):
                    print(f"{i}. {action}")
        else:
            print("Model training failed.")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()