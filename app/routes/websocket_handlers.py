"""
WebSocket event handlers for real-time eye tracking communication.

This module handles all Socket.IO events for:
- Calibration data collection
- Validation data collection
- Real-time gaze tracking
- State management signals
"""

import json
import logging
from typing import TYPE_CHECKING

from flask_socketio import emit

if TYPE_CHECKING:
    from flask_socketio import SocketIO
    from app.models import GazeTracker

logger = logging.getLogger(__name__)


def register_handlers(socketio: "SocketIO", tracker: "GazeTracker") -> None:
    """
    Register all WebSocket event handlers.
    
    Args:
        socketio: Flask-SocketIO instance
        tracker: GazeTracker instance for managing tracking state
    """
    
    @socketio.on("calibrationDataOneByOneUpdate")
    def handle_calibration_data(data: str) -> None:
        """
        Handle incoming calibration data point.
        
        Expected data format (JSON string):
        {
            "screenX": float,
            "screenY": float,
            "leftX": float,
            "leftY": float,
            "rightX": float,
            "rightY": float,
            "noseX": float,
            "noseY": float,
            "blink": int (0 or 1)
        }
        """
        try:
            point_data = json.loads(data)
            tracker.add_calibration_point(point_data)
            
            if tracker.calibration_point_count % 100 == 0:
                logger.debug("Calibration points collected: %d", 
                           tracker.calibration_point_count)
                
        except json.JSONDecodeError as e:
            logger.error("Invalid calibration data JSON: %s", e)
        except KeyError as e:
            logger.error("Missing required field in calibration data: %s", e)
    
    @socketio.on("calibrationStatus")
    def handle_calibration_status(status: bool) -> None:
        """
        Handle calibration completion signal.
        
        Args:
            status: True when calibration data collection is complete
        """
        if status:
            logger.info("Calibration status received, training model...")
            success = tracker.finish_calibration()
            
            if success:
                emit("modelTrained", namespace="/")
                logger.info("Model trained, emitted modelTrained event")
            else:
                emit("calibrationError", {"error": "Insufficient data"}, namespace="/")
    
    @socketio.on("validationData")
    def handle_validation_data(data: str) -> None:
        """
        Handle incoming validation data point.
        
        Expected data format (JSON string):
        {
            "screen_x": float,
            "screen_y": float,
            "predicted_x": float,
            "predicted_y": float
        }
        """
        try:
            point_data = json.loads(data)
            tracker.add_validation_point(
                screen_x=point_data["screen_x"],
                screen_y=point_data["screen_y"],
                predicted_x=point_data["predicted_x"],
                predicted_y=point_data["predicted_y"],
            )
        except json.JSONDecodeError as e:
            logger.error("Invalid validation data JSON: %s", e)
        except KeyError as e:
            logger.error("Missing required field in validation data: %s", e)
    
    @socketio.on("validationStatus")
    def handle_validation_status(status: bool) -> None:
        """
        Handle validation completion signal.
        
        Args:
            status: True when validation data collection is complete
        """
        if status:
            logger.info("Validation status received, training correction model...")
            tracker.finish_validation()
            emit("validationStatus", namespace="/")
            logger.info("Validation complete, emitted validationStatus event")
    
    @socketio.on("realTimeData")
    def handle_real_time_data(data: str) -> None:
        """
        Handle real-time tracking data and emit predictions.
        
        Expected data format (JSON string):
        {
            "leftX": float,
            "leftY": float,
            "rightX": float,
            "rightY": float,
            "noseX": float,
            "noseY": float,
            "blink": int,
            "validationDone": bool
        }
        
        Emits:
            data_response: JSON string with [x, y] coordinates
        """
        try:
            tracking_data = json.loads(data)
            
            # Skip if model not ready
            if not tracker.is_calibrated:
                return
            
            # Get smoothed prediction
            coordinates = tracker.track(tracking_data)
            
            # Emit response
            response = json.dumps(coordinates)
            emit("data_response", response)
            
        except json.JSONDecodeError as e:
            logger.error("Invalid real-time data JSON: %s", e)
        except RuntimeError as e:
            logger.error("Prediction error: %s", e)
    
    @socketio.on("connect")
    def handle_connect() -> None:
        """Handle client connection."""
        logger.info("Client connected")
    
    @socketio.on("disconnect")
    def handle_disconnect() -> None:
        """Handle client disconnection."""
        logger.info("Client disconnected")
    
    @socketio.on("reset")
    def handle_reset() -> None:
        """Handle tracker reset request."""
        tracker.reset()
        emit("resetComplete", namespace="/")
        logger.info("Tracker reset by client request")
    
    logger.info("WebSocket handlers registered")
