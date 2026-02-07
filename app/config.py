"""
Configuration settings for the Gazer eye-tracking application.

This module centralizes all configurable parameters, making it easy to
tune the system and understand the available options.
"""

import os
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for the gaze estimation ML models."""
    
    # SVR hyperparameters
    svr_kernel: str = "poly"
    svr_c: float = 150.0
    svr_degree: int = 3
    svr_epsilon: float = 0.1
    
    # Ensemble weights for combining pupil and head predictions
    # final = pupil_weight * pupil_pred + head_weight * head_pred
    pupil_weight: float = 1.5
    head_weight: float = 0.5


@dataclass
class CalibrationConfig:
    """Configuration for the calibration process."""
    
    # Number of data points to collect per calibration point
    points_per_calibration: int = 100
    
    # Number of data points for validation phase
    points_per_validation: int = 300
    
    # Number of points for testing accuracy
    points_per_test: int = 400
    
    # Minimum accuracy threshold (below this, calibration is unreliable)
    min_accuracy_threshold: float = 0.65
    
    # Validation point filter: max pixel distance to accept a point
    validation_distance_threshold: int = 100


@dataclass
class TrackingConfig:
    """Configuration for real-time gaze tracking."""
    
    # Number of previous positions to keep for smoothing
    max_history_points: int = 10
    
    # Smoothing factor (0 = no smoothing, 1 = full average with previous)
    smoothing_factor: float = 0.5


@dataclass
class ServerConfig:
    """Configuration for the Flask server."""
    
    host: str = "0.0.0.0"
    port: int = 3226
    debug: bool = True
    
    # CORS settings
    cors_enabled: bool = True
    
    # SSL settings (for HTTPS)
    ssl_enabled: bool = False
    ssl_cert_path: str = "certificate.crt"
    ssl_key_path: str = "private.key"


@dataclass
class Config:
    """Main configuration class aggregating all settings."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    
    @classmethod
    def from_env(cls) -> "Config":
        """
        Create configuration from environment variables.
        
        Environment variables override default values.
        Prefix: GAZER_
        
        Examples:
            GAZER_SERVER_PORT=8080
            GAZER_MODEL_SVR_C=200
            GAZER_CALIBRATION_POINTS_PER_CALIBRATION=150
        """
        config = cls()
        
        # Server settings from env
        config.server.port = int(os.getenv("GAZER_SERVER_PORT", config.server.port))
        config.server.host = os.getenv("GAZER_SERVER_HOST", config.server.host)
        config.server.debug = os.getenv("GAZER_DEBUG", "true").lower() == "true"
        
        # Model settings from env
        config.model.svr_c = float(os.getenv("GAZER_MODEL_SVR_C", config.model.svr_c))
        config.model.pupil_weight = float(
            os.getenv("GAZER_MODEL_PUPIL_WEIGHT", config.model.pupil_weight)
        )
        config.model.head_weight = float(
            os.getenv("GAZER_MODEL_HEAD_WEIGHT", config.model.head_weight)
        )
        
        # Calibration settings from env
        config.calibration.points_per_calibration = int(
            os.getenv("GAZER_CALIBRATION_POINTS", config.calibration.points_per_calibration)
        )
        
        return config


# Default configuration instance
config = Config.from_env()
