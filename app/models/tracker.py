"""
Gaze Tracker - Real-time state management for eye tracking.

This module manages the state of the eye tracking session, including:
- Accumulating calibration data
- Managing position history for smoothing
- Coordinating between calibration, validation, and tracking phases
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, List, Optional, Tuple

import pandas as pd

from app.config import config
from app.models.gaze_estimator import CalibrationPoint, GazeEstimator, GazePoint

logger = logging.getLogger(__name__)


class TrackerState(Enum):
    """Current state of the eye tracker."""
    IDLE = auto()
    CALIBRATING = auto()
    VALIDATING = auto()
    TRACKING = auto()


@dataclass
class ValidationData:
    """Accumulated validation data point."""
    screen_x: float
    screen_y: float
    predicted_x: float
    predicted_y: float


class GazeTracker:
    """
    Manages the complete eye tracking session lifecycle.
    
    Handles:
    - State transitions (idle → calibrating → validating → tracking)
    - Data accumulation during calibration/validation
    - Real-time prediction with smoothing
    - Position history management
    
    Thread Safety:
        This class is NOT thread-safe. In production, consider using
        locks or a thread-safe queue for data accumulation.
    
    Example:
        tracker = GazeTracker()
        
        # During calibration
        tracker.add_calibration_point(data)
        tracker.finish_calibration()
        
        # During tracking
        gaze_point = tracker.track(pupil_data, head_data)
    """
    
    def __init__(self):
        """Initialize the gaze tracker."""
        self._estimator = GazeEstimator()
        self._state = TrackerState.IDLE
        
        # Calibration data accumulator
        self._calibration_data: List[CalibrationPoint] = []
        
        # Validation data accumulator
        self._validation_data: List[ValidationData] = []
        
        # Position history for smoothing
        self._position_history: Deque[Tuple[float, float]] = deque(
            maxlen=config.tracking.max_history_points
        )
        
        logger.info("GazeTracker initialized")
    
    @property
    def state(self) -> TrackerState:
        """Current tracker state."""
        return self._state
    
    @property
    def is_calibrated(self) -> bool:
        """Whether calibration is complete."""
        return self._estimator.is_calibrated
    
    @property
    def is_validated(self) -> bool:
        """Whether validation is complete."""
        return self._estimator.is_validated
    
    @property
    def calibration_point_count(self) -> int:
        """Number of accumulated calibration points."""
        return len(self._calibration_data)
    
    def start_calibration(self) -> None:
        """Begin the calibration phase."""
        self._state = TrackerState.CALIBRATING
        self._calibration_data.clear()
        logger.info("Calibration started")
    
    def add_calibration_point(self, data: dict) -> None:
        """
        Add a calibration data point.
        
        Args:
            data: Dictionary with keys:
                - screenX, screenY: Target screen coordinates
                - leftX, leftY: Left pupil coordinates
                - rightX, rightY: Right pupil coordinates
                - noseX, noseY: Nose tip coordinates
        """
        # Calculate centroid
        centroid_x = (data["leftX"] + data["rightX"] + data["noseX"]) / 3
        centroid_y = (data["leftY"] + data["rightY"] + data["noseY"]) / 3
        
        point = CalibrationPoint(
            screen_x=data["screenX"],
            screen_y=data["screenY"],
            left_x=data["leftX"],
            left_y=data["leftY"],
            right_x=data["rightX"],
            right_y=data["rightY"],
            nose_x=data["noseX"],
            nose_y=data["noseY"],
            centroid_x=centroid_x,
            centroid_y=centroid_y,
        )
        
        self._calibration_data.append(point)
    
    def finish_calibration(self) -> bool:
        """
        Complete calibration and train the model.
        
        Returns:
            True if calibration successful, False otherwise
        """
        try:
            self._estimator.calibrate(self._calibration_data)
            self._state = TrackerState.VALIDATING
            logger.info("Calibration finished successfully with %d points", 
                       len(self._calibration_data))
            self._calibration_data.clear()
            return True
        except ValueError as e:
            logger.error("Calibration failed: %s", e)
            return False
    
    def start_validation(self) -> None:
        """Begin the validation phase."""
        self._state = TrackerState.VALIDATING
        self._validation_data.clear()
        logger.info("Validation started")
    
    def add_validation_point(
        self,
        screen_x: float,
        screen_y: float,
        predicted_x: float,
        predicted_y: float
    ) -> None:
        """
        Add a validation data point.
        
        Points with large prediction errors are filtered out to improve
        validation model quality.
        
        Args:
            screen_x, screen_y: Actual target coordinates
            predicted_x, predicted_y: Model predictions
        """
        # Filter out points with large errors
        threshold = config.calibration.validation_distance_threshold
        if (abs(predicted_x - screen_x) < threshold or 
            abs(predicted_y - screen_y) < threshold):
            self._validation_data.append(ValidationData(
                screen_x=screen_x,
                screen_y=screen_y,
                predicted_x=predicted_x,
                predicted_y=predicted_y,
            ))
    
    def finish_validation(self) -> bool:
        """
        Complete validation and train the correction model.
        
        Returns:
            True if validation successful, False otherwise
        """
        if not self._validation_data:
            logger.warning("No validation data collected")
            self._state = TrackerState.TRACKING
            return False
        
        # Convert to DataFrame for model training
        validation_df = pd.DataFrame([
            {
                "screen_x": v.screen_x,
                "screen_y": v.screen_y,
                "predicted_x": v.predicted_x,
                "predicted_y": v.predicted_y,
            }
            for v in self._validation_data
        ])
        
        self._estimator.train_validation(validation_df)
        self._state = TrackerState.TRACKING
        self._validation_data.clear()
        
        logger.info("Validation finished")
        return True
    
    def track(self, data: dict) -> Tuple[float, float]:
        """
        Get smoothed gaze prediction for real-time tracking.
        
        Args:
            data: Dictionary with pupil and head tracking data
            
        Returns:
            Tuple of (x, y) smoothed screen coordinates
        """
        # Calculate centroid
        centroid_x = (data["leftX"] + data["rightX"] + data["noseX"]) / 3
        centroid_y = (data["leftY"] + data["rightY"] + data["noseY"]) / 3
        
        # Prepare feature DataFrames
        pupil_data = pd.DataFrame({
            "left_x": [data["leftX"]],
            "left_y": [data["leftY"]],
            "right_x": [data["rightX"]],
            "right_y": [data["rightY"]],
        })
        
        head_data = pd.DataFrame({
            "nose_x": [data["noseX"]],
            "nose_y": [data["noseY"]],
            "centroid_x": [centroid_x],
            "centroid_y": [centroid_y],
        })
        
        # Get prediction
        gaze_point = self._estimator.predict(pupil_data, head_data)
        current_pos = gaze_point.to_tuple()
        
        # Apply smoothing with position history
        smoothed_pos = self._smooth_position(current_pos)
        
        return smoothed_pos
    
    def predict_for_validation(self, data: dict) -> Tuple[float, float]:
        """
        Get raw prediction (without validation correction) for validation phase.
        
        Args:
            data: Dictionary with pupil and head tracking data
            
        Returns:
            Tuple of (x, y) raw predicted coordinates
        """
        centroid_x = (data["leftX"] + data["rightX"] + data["noseX"]) / 3
        centroid_y = (data["leftY"] + data["rightY"] + data["noseY"]) / 3
        
        pupil_data = pd.DataFrame({
            "left_x": [data["leftX"]],
            "left_y": [data["leftY"]],
            "right_x": [data["rightX"]],
            "right_y": [data["rightY"]],
        })
        
        head_data = pd.DataFrame({
            "nose_x": [data["noseX"]],
            "nose_y": [data["noseY"]],
            "centroid_x": [centroid_x],
            "centroid_y": [centroid_y],
        })
        
        prediction = self._estimator.predict_raw(pupil_data, head_data)
        return (prediction[0, 0], prediction[0, 1])
    
    def _smooth_position(self, current: Tuple[float, float]) -> Tuple[float, float]:
        """
        Apply exponential smoothing to reduce jitter.
        
        Args:
            current: Current predicted position
            
        Returns:
            Smoothed position
        """
        if not self._position_history:
            self._position_history.append(current)
            return current
        
        # Get previous position
        prev = self._position_history[-1]
        
        # Simple averaging (could be replaced with Kalman filter)
        alpha = config.tracking.smoothing_factor
        smoothed = (
            alpha * prev[0] + (1 - alpha) * current[0],
            alpha * prev[1] + (1 - alpha) * current[1],
        )
        
        self._position_history.append(smoothed)
        return smoothed
    
    def reset(self) -> None:
        """Reset tracker to initial state."""
        self._estimator.reset()
        self._state = TrackerState.IDLE
        self._calibration_data.clear()
        self._validation_data.clear()
        self._position_history.clear()
        logger.info("GazeTracker reset")
