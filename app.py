"""
Gazer - Webcam-based Eye Tracking Application

This version preserves the original working logic while improving code organization.
"""

import json
import logging
import sys

from flask import Flask, render_template
from flask_socketio import SocketIO
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ============================================================================
# Flask App Setup
# ============================================================================

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# ============================================================================
# ML Models (Global - same as original)
# ============================================================================

# Base SVR models for pupil and head tracking
base_model = MultiOutputRegressor(SVR(kernel='poly', C=150, degree=3, epsilon=0.1))
base_model2 = MultiOutputRegressor(SVR(kernel='poly', C=150, degree=3, epsilon=0.1))

# Stacking models (Linear Regression)
stacking_model = LinearRegression()
stacking_model2 = LinearRegression()

# Validation correction model
validation_model = LinearRegression()

# ============================================================================
# State Variables (Global - same as original)
# ============================================================================

accumulated_calibration_data = []
accumulated_validation_data = pd.DataFrame(columns=['screen_x', 'screen_y', 'predicted_x', 'predicted_y'])
model_trained = False
validation_Done = False
mouse_positions = []
MAX_HISTORY_POINTS = 10

# ============================================================================
# Model Training Functions
# ============================================================================

def update(data):
    """Train the gaze estimation models on calibration data."""
    global accumulated_calibration_data, model_trained
    
    logger.info(f"Training model with {len(data)} calibration points")
    
    # Create numpy array for screen coordinates (used by stacking model)
    screen_coordinates = np.empty((0, 2))
    
    # Extract pupil features
    calibration_data = pd.DataFrame({
        'left_x': [item['leftX'] for item in data],
        'left_y': [item['leftY'] for item in data],
        'right_x': [item['rightX'] for item in data],
        'right_y': [item['rightY'] for item in data]
    })
    
    # Extract head features
    head_data = pd.DataFrame({
        'nose_x': [item['noseX'] for item in data],
        'nose_y': [item['noseY'] for item in data],
        'centroid_x': [item['centroidX'] for item in data],
        'centroid_y': [item['centroidY'] for item in data]
    })
    
    # Build screen coordinates numpy array (IMPORTANT: must be numpy for stacking)
    for item in data:
        screen_point = (item['screenX'], item['screenY'])
        screen_coordinates = np.vstack((screen_coordinates, screen_point))
    
    # Also create DataFrame version for base model training
    screen_coords_df = pd.DataFrame({
        'screen_x': [item['screenX'] for item in data],
        'screen_y': [item['screenY'] for item in data]
    })
    
    logger.info("Fitting base models...")
    
    # Train base models
    base_model.fit(calibration_data, screen_coords_df)
    base_model2.fit(head_data, screen_coords_df)
    
    logger.info("Generating base model predictions for stacking...")
    
    # Get predictions for stacking
    base_model_predictions = base_model.predict(calibration_data)
    base_model_head_pred = base_model2.predict(head_data)
    
    logger.info("Fitting stacking models...")
    
    # Train stacking models (IMPORTANT: use numpy array screen_coordinates)
    stacking_model.fit(base_model_predictions, screen_coordinates)
    stacking_model2.fit(base_model_head_pred, screen_coordinates)
    
    model_trained = True
    logger.info("Model training complete!")
    
    socketio.emit('modelTrained', namespace='/')
    accumulated_calibration_data.clear()


def updateValidation():
    """Train the validation correction model."""
    global accumulated_validation_data, validation_Done
    
    data = accumulated_validation_data
    initial_rows = data.shape[0]
    data.dropna(inplace=True)
    
    logger.info(f"Training validation model with {len(data)} points (dropped {initial_rows - len(data)} NaN rows)")
    
    validation_model.fit(
        data[['predicted_x', 'predicted_y']], 
        data[['screen_x', 'screen_y']]
    )
    
    validation_Done = True
    logger.info("Validation model trained!")
    
    socketio.emit('validationStatus', namespace='/')


def predict_new(real_pupil, real_head):
    """Generate gaze prediction from pupil and head data."""
    global validation_Done
    
    # Get base model predictions
    base_model_predictions_pupil = base_model.predict(real_pupil)
    base_model_predictions_head = base_model2.predict(real_head)
    
    # Get stacking model predictions
    final_predictions_pupil = stacking_model.predict(base_model_predictions_pupil)
    final_predictions_head = stacking_model2.predict(base_model_predictions_head)
    
    # Weighted combination (1.5 * pupil + 0.5 * head)
    final_predictions = 1.5 * final_predictions_pupil + 0.5 * final_predictions_head
    
    actual_predicted = final_predictions
    
    # Apply validation correction if available
    if validation_Done:
        validation_predictions = validation_model.predict(final_predictions)
        return validation_predictions
    else:
        return actual_predicted


# ============================================================================
# Routes
# ============================================================================

@app.route('/')
def index():
    """Serve instructions page."""
    return render_template('instructions.html')


@app.route('/eyetracking')
def eyetracking():
    """Serve eye tracking interface."""
    return render_template('eyetracking.html')


@app.route('/health')
def health():
    """Health check endpoint."""
    return {"status": "healthy", "model_trained": model_trained, "validation_done": validation_Done}


# ============================================================================
# WebSocket Event Handlers
# ============================================================================

@socketio.on('calibrationDataOneByOneUpdate')
def handle_calibration_data(data):
    """Receive calibration data point by point."""
    global accumulated_calibration_data
    
    js_array = json.loads(data)
    
    # Calculate centroid (same as original)
    js_array["centroidX"] = (js_array["leftX"] + js_array["rightX"] + js_array["noseX"]) / 3
    js_array["centroidY"] = (js_array["leftY"] + js_array["rightY"] + js_array["noseY"]) / 3
    
    accumulated_calibration_data.append(js_array)


@socketio.on('calibrationStatus')
def handle_calibration_status(data):
    """Handle calibration completion signal."""
    if data == True:
        logger.info("Calibration complete, starting model training...")
        update(accumulated_calibration_data)


@socketio.on('realTimeData')
def handle_real_time_data(data):
    """Handle real-time tracking data and return predictions."""
    global mouse_positions
    
    js_array = json.loads(data)
    
    # Calculate centroid
    js_array["centroidX"] = (js_array["leftX"] + js_array["rightX"] + js_array["noseX"]) / 3
    js_array["centroidY"] = (js_array["leftY"] + js_array["rightY"] + js_array["noseY"]) / 3
    
    # Prepare pupil data DataFrame
    real_time_data = pd.DataFrame({
        'left_x': js_array['leftX'],
        'left_y': js_array['leftY'],
        'right_x': js_array['rightX'],
        'right_y': js_array['rightY']
    }, index=[0])
    
    # Prepare head data DataFrame
    real_time_data_head = pd.DataFrame({
        'nose_x': js_array['noseX'],
        'nose_y': js_array['noseY'],
        'centroid_x': js_array['centroidX'],
        'centroid_y': js_array['centroidY'],
    }, index=[0])
    
    # Get prediction
    predictedScreenCoordinates = predict_new(real_time_data, real_time_data_head)
    
    # === ORIGINAL SMOOTHING LOGIC (CRITICAL - DO NOT CHANGE) ===
    screen_prev_position = (0, 0)
    
    if mouse_positions:
        screen_prev_position = mouse_positions[-1]
    
    screen_coordinates = predictedScreenCoordinates[0]
    
    # Average with previous position
    coordinates = (
        (screen_coordinates[0] + screen_prev_position[0]) / 2,
        (screen_coordinates[1] + screen_prev_position[1]) / 2
    )
    
    # Append current and (0,0) - THIS IS THE ORIGINAL PATTERN
    mouse_positions.append(tuple(coordinates))
    mouse_positions.append((0, 0))
    mouse_positions = mouse_positions[-MAX_HISTORY_POINTS:]
    # === END ORIGINAL SMOOTHING LOGIC ===
    
    # Send response
    data_json = json.dumps(coordinates)
    socketio.emit('data_response', data_json)


@socketio.on('validationData')
def handle_validation_data(data):
    """Receive validation data."""
    global accumulated_validation_data
    
    js_array = json.loads(data)
    new_data = pd.DataFrame(js_array, index=[0])
    
    accumulated_validation_data = pd.concat(
        [accumulated_validation_data, new_data], 
        ignore_index=True
    )


@socketio.on('validationStatus')
def handle_validation_status(data):
    """Handle validation completion signal."""
    global accumulated_validation_data
    
    if data == True:
        logger.info("Validation complete, training correction model...")
        updateValidation()


@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info("Client connected")


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info("Client disconnected")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    logger.info("Starting Gazer server on 0.0.0.0:3226")
    socketio.run(app, host='0.0.0.0', port=3226, debug=True)