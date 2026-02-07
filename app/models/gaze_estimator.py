"""
Gaze Estimation Model using Stacked SVR with head position compensation.

This module implements the core machine learning pipeline for webcam-based
eye tracking. It uses a two-stage stacking approach:
1. Base models: MultiOutput SVR for pupil coords and head position
2. Stacking models: Linear Regression to refine predictions

The architecture combines pupil tracking with head pose estimation to
improve accuracy and reduce positional dependency.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

from app.config import config

logger = logging.getLogger(__name__)


@dataclass
class GazePoint:
    """Represents a predicted gaze point on screen."""
    x: float
    y: float
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    def __iter__(self):
        yield self.x
        yield self.y


@dataclass
class CalibrationPoint:
    """Raw calibration data from a single frame."""
    screen_x: float
    screen_y: float
    left_x: float
    left_y: float
    right_x: float
    right_y: float
    nose_x: float
    nose_y: float
    centroid_x: Optional[float] = None
    centroid_y: Optional[float] = None
    
    def __post_init__(self):
        """Calculate centroid if not provided."""
        if self.centroid_x is None:
            self.centroid_x = (self.left_x + self.right_x + self.nose_x) / 3
        if self.centroid_y is None:
            self.centroid_y = (self.left_y + self.right_y + self.nose_y) / 3
    
    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationPoint":
        """Create CalibrationPoint from JavaScript data dictionary."""
        return cls(
            screen_x=data.get("screenX", 0),
            screen_y=data.get("screenY", 0),
            left_x=data["leftX"],
            left_y=data["leftY"],
            right_x=data["rightX"],
            right_y=data["rightY"],
            nose_x=data["noseX"],
            nose_y=data["noseY"],
            centroid_x=data.get("centroidX"),
            centroid_y=data.get("centroidY"),
        )


class GazeEstimator:
    """
    Stacked ensemble model for gaze estimation.
    
    Architecture:
    ```
    Pupil Coords ──► SVR Base Model ──► Linear Stacking ──┐
                                                          ├──► Weighted Average ──► Final Prediction
    Head Coords ───► SVR Base Model ──► Linear Stacking ──┘
    ```
    
    The validation model provides an additional correction layer
    trained on the difference between predicted and actual gaze points.
    
    Attributes:
        is_calibrated: Whether the model has been trained on calibration data
        is_validated: Whether the validation correction model is trained
    """
    
    def __init__(self):
        """Initialize the gaze estimator with configured hyperparameters."""
        self._config = config.model
        
        # Base models for pupil and head tracking
        self._pupil_base_model = self._create_svr_model()
        self._head_base_model = self._create_svr_model()
        
        # Stacking models to refine base predictions
        self._pupil_stacking_model = LinearRegression()
        self._head_stacking_model = LinearRegression()
        
        # Validation model for final correction
        self._validation_model = LinearRegression()
        
        # State flags
        self.is_calibrated = False
        self.is_validated = False
        
        logger.info("GazeEstimator initialized with SVR(kernel=%s, C=%s, degree=%s)",
                    self._config.svr_kernel, self._config.svr_c, self._config.svr_degree)
    
    def _create_svr_model(self) -> MultiOutputRegressor:
        """Create a MultiOutput SVR model with configured hyperparameters."""
        return MultiOutputRegressor(
            SVR(
                kernel=self._config.svr_kernel,
                C=self._config.svr_c,
                degree=self._config.svr_degree,
                epsilon=self._config.svr_epsilon,
            )
        )
    
    def calibrate(self, calibration_points: list[CalibrationPoint]) -> None:
        """
        Train the gaze estimation models on calibration data.
        
        This performs the full training pipeline:
        1. Extract features from calibration points
        2. Train base SVR models on pupil and head data
        3. Train stacking models on base model predictions
        
        Args:
            calibration_points: List of CalibrationPoint objects from calibration phase
            
        Raises:
            ValueError: If insufficient calibration data provided
        """
        if len(calibration_points) < 10:
            raise ValueError(f"Insufficient calibration data: {len(calibration_points)} points")
        
        logger.info("Starting calibration with %d data points", len(calibration_points))
        
        # Extract feature matrices
        pupil_features = pd.DataFrame({
            "left_x": [p.left_x for p in calibration_points],
            "left_y": [p.left_y for p in calibration_points],
            "right_x": [p.right_x for p in calibration_points],
            "right_y": [p.right_y for p in calibration_points],
        })
        
        head_features = pd.DataFrame({
            "nose_x": [p.nose_x for p in calibration_points],
            "nose_y": [p.nose_y for p in calibration_points],
            "centroid_x": [p.centroid_x for p in calibration_points],
            "centroid_y": [p.centroid_y for p in calibration_points],
        })
        
        screen_coords = pd.DataFrame({
            "screen_x": [p.screen_x for p in calibration_points],
            "screen_y": [p.screen_y for p in calibration_points],
        })
        
        screen_coords_array = screen_coords.values
        
        # Train base models
        logger.debug("Training pupil base model...")
        self._pupil_base_model.fit(pupil_features, screen_coords)
        
        logger.debug("Training head base model...")
        self._head_base_model.fit(head_features, screen_coords)
        
        # Get base model predictions for stacking
        pupil_predictions = self._pupil_base_model.predict(pupil_features)
        head_predictions = self._head_base_model.predict(head_features)
        
        # Train stacking models
        logger.debug("Training stacking models...")
        self._pupil_stacking_model.fit(pupil_predictions, screen_coords_array)
        self._head_stacking_model.fit(head_predictions, screen_coords_array)
        
        self.is_calibrated = True
        logger.info("Calibration complete")
    
    def train_validation(self, validation_data: pd.DataFrame) -> None:
        """
        Train the validation correction model.
        
        The validation model learns to correct systematic errors by
        mapping predicted coordinates to actual screen coordinates.
        
        Args:
            validation_data: DataFrame with columns:
                - predicted_x, predicted_y: Model predictions
                - screen_x, screen_y: Actual screen coordinates
        """
        if validation_data.empty:
            logger.warning("Empty validation data, skipping validation training")
            return
        
        # Remove any NaN values
        clean_data = validation_data.dropna()
        
        if len(clean_data) < 5:
            logger.warning("Insufficient validation data after cleaning: %d points", len(clean_data))
            return
        
        logger.info("Training validation model with %d points", len(clean_data))
        
        self._validation_model.fit(
            clean_data[["predicted_x", "predicted_y"]],
            clean_data[["screen_x", "screen_y"]]
        )
        
        self.is_validated = True
        logger.info("Validation model trained")
    
    def predict(self, pupil_data: pd.DataFrame, head_data: pd.DataFrame) -> GazePoint:
        """
        Predict gaze point from pupil and head tracking data.
        
        Args:
            pupil_data: DataFrame with columns [left_x, left_y, right_x, right_y]
            head_data: DataFrame with columns [nose_x, nose_y, centroid_x, centroid_y]
            
        Returns:
            GazePoint with predicted x, y screen coordinates
            
        Raises:
            RuntimeError: If model hasn't been calibrated
        """
        if not self.is_calibrated:
            raise RuntimeError("Model must be calibrated before prediction")
        
        # Base model predictions
        pupil_base_pred = self._pupil_base_model.predict(pupil_data)
        head_base_pred = self._head_base_model.predict(head_data)
        
        # Stacking model predictions
        pupil_final = self._pupil_stacking_model.predict(pupil_base_pred)
        head_final = self._head_stacking_model.predict(head_base_pred)
        
        # Weighted combination
        combined = (
            self._config.pupil_weight * pupil_final + 
            self._config.head_weight * head_final
        )
        
        # Apply validation correction if available
        if self.is_validated:
            combined = self._validation_model.predict(combined)
        
        return GazePoint(x=combined[0, 0], y=combined[0, 1])
    
    def predict_raw(self, pupil_data: pd.DataFrame, head_data: pd.DataFrame) -> np.ndarray:
        """
        Predict gaze point without validation correction.
        
        Used during the validation phase to collect data for training
        the validation model.
        
        Args:
            pupil_data: DataFrame with pupil coordinates
            head_data: DataFrame with head position data
            
        Returns:
            numpy array of shape (1, 2) with [x, y] coordinates
        """
        if not self.is_calibrated:
            raise RuntimeError("Model must be calibrated before prediction")
        
        pupil_base_pred = self._pupil_base_model.predict(pupil_data)
        head_base_pred = self._head_base_model.predict(head_data)
        
        pupil_final = self._pupil_stacking_model.predict(pupil_base_pred)
        head_final = self._head_stacking_model.predict(head_base_pred)
        
        return self._config.pupil_weight * pupil_final + self._config.head_weight * head_final
    
    def reset(self) -> None:
        """Reset the model to untrained state."""
        self.__init__()
        logger.info("GazeEstimator reset to initial state")
