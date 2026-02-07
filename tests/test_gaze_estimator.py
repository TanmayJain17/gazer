"""
Unit tests for the GazeEstimator model.

Run with: pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.gaze_estimator import GazeEstimator, CalibrationPoint, GazePoint
from app.models.tracker import GazeTracker, TrackerState


class TestCalibrationPoint:
    """Tests for CalibrationPoint dataclass."""
    
    def test_creation_with_all_fields(self):
        """Test creating a CalibrationPoint with all fields."""
        point = CalibrationPoint(
            screen_x=100.0,
            screen_y=200.0,
            left_x=50.0,
            left_y=60.0,
            right_x=70.0,
            right_y=80.0,
            nose_x=60.0,
            nose_y=100.0,
        )
        
        assert point.screen_x == 100.0
        assert point.screen_y == 200.0
        assert point.centroid_x is not None  # Auto-calculated
        
    def test_centroid_calculation(self):
        """Test automatic centroid calculation."""
        point = CalibrationPoint(
            screen_x=0, screen_y=0,
            left_x=30.0, left_y=30.0,
            right_x=60.0, right_y=60.0,
            nose_x=45.0, nose_y=90.0,
        )
        
        expected_centroid_x = (30.0 + 60.0 + 45.0) / 3
        expected_centroid_y = (30.0 + 60.0 + 90.0) / 3
        
        assert point.centroid_x == pytest.approx(expected_centroid_x)
        assert point.centroid_y == pytest.approx(expected_centroid_y)
    
    def test_from_dict(self):
        """Test creating CalibrationPoint from dictionary."""
        data = {
            "screenX": 500.0,
            "screenY": 300.0,
            "leftX": 100.0,
            "leftY": 110.0,
            "rightX": 120.0,
            "rightY": 130.0,
            "noseX": 110.0,
            "noseY": 150.0,
        }
        
        point = CalibrationPoint.from_dict(data)
        
        assert point.screen_x == 500.0
        assert point.left_x == 100.0


class TestGazePoint:
    """Tests for GazePoint dataclass."""
    
    def test_to_tuple(self):
        """Test converting GazePoint to tuple."""
        point = GazePoint(x=100.5, y=200.5)
        
        assert point.to_tuple() == (100.5, 200.5)
    
    def test_iteration(self):
        """Test iterating over GazePoint."""
        point = GazePoint(x=50.0, y=75.0)
        
        x, y = point
        assert x == 50.0
        assert y == 75.0


class TestGazeEstimator:
    """Tests for GazeEstimator model."""
    
    @pytest.fixture
    def estimator(self):
        """Create a fresh GazeEstimator instance."""
        return GazeEstimator()
    
    @pytest.fixture
    def sample_calibration_data(self):
        """Generate sample calibration data."""
        np.random.seed(42)
        points = []
        
        # Generate 100 sample points
        for _ in range(100):
            screen_x = np.random.uniform(0, 1920)
            screen_y = np.random.uniform(0, 1080)
            
            # Simulate pupil positions (correlated with screen position)
            left_x = 200 + (screen_x / 1920) * 100 + np.random.normal(0, 5)
            left_y = 200 + (screen_y / 1080) * 50 + np.random.normal(0, 5)
            right_x = left_x + 50 + np.random.normal(0, 2)
            right_y = left_y + np.random.normal(0, 2)
            nose_x = (left_x + right_x) / 2
            nose_y = left_y + 50
            
            points.append(CalibrationPoint(
                screen_x=screen_x,
                screen_y=screen_y,
                left_x=left_x,
                left_y=left_y,
                right_x=right_x,
                right_y=right_y,
                nose_x=nose_x,
                nose_y=nose_y,
            ))
        
        return points
    
    def test_initial_state(self, estimator):
        """Test estimator initial state."""
        assert not estimator.is_calibrated
        assert not estimator.is_validated
    
    def test_calibration_insufficient_data(self, estimator):
        """Test calibration fails with insufficient data."""
        few_points = [
            CalibrationPoint(0, 0, 0, 0, 0, 0, 0, 0)
            for _ in range(5)
        ]
        
        with pytest.raises(ValueError, match="Insufficient"):
            estimator.calibrate(few_points)
    
    def test_calibration_success(self, estimator, sample_calibration_data):
        """Test successful calibration."""
        estimator.calibrate(sample_calibration_data)
        
        assert estimator.is_calibrated
        assert not estimator.is_validated
    
    def test_prediction_before_calibration(self, estimator):
        """Test prediction fails before calibration."""
        pupil_data = pd.DataFrame({
            "left_x": [100], "left_y": [100],
            "right_x": [150], "right_y": [100]
        })
        head_data = pd.DataFrame({
            "nose_x": [125], "nose_y": [150],
            "centroid_x": [125], "centroid_y": [117]
        })
        
        with pytest.raises(RuntimeError, match="calibrated"):
            estimator.predict(pupil_data, head_data)
    
    def test_prediction_after_calibration(self, estimator, sample_calibration_data):
        """Test prediction works after calibration."""
        estimator.calibrate(sample_calibration_data)
        
        pupil_data = pd.DataFrame({
            "left_x": [250.0], "left_y": [225.0],
            "right_x": [300.0], "right_y": [225.0]
        })
        head_data = pd.DataFrame({
            "nose_x": [275.0], "nose_y": [275.0],
            "centroid_x": [275.0], "centroid_y": [242.0]
        })
        
        result = estimator.predict(pupil_data, head_data)
        
        assert isinstance(result, GazePoint)
        assert isinstance(result.x, (int, float))
        assert isinstance(result.y, (int, float))
    
    def test_reset(self, estimator, sample_calibration_data):
        """Test model reset."""
        estimator.calibrate(sample_calibration_data)
        assert estimator.is_calibrated
        
        estimator.reset()
        
        assert not estimator.is_calibrated
        assert not estimator.is_validated


class TestGazeTracker:
    """Tests for GazeTracker state manager."""
    
    @pytest.fixture
    def tracker(self):
        """Create a fresh GazeTracker instance."""
        return GazeTracker()
    
    def test_initial_state(self, tracker):
        """Test tracker initial state."""
        assert tracker.state == TrackerState.IDLE
        assert not tracker.is_calibrated
        assert tracker.calibration_point_count == 0
    
    def test_add_calibration_point(self, tracker):
        """Test adding calibration points."""
        data = {
            "screenX": 500, "screenY": 300,
            "leftX": 100, "leftY": 110,
            "rightX": 150, "rightY": 110,
            "noseX": 125, "noseY": 160,
        }
        
        tracker.add_calibration_point(data)
        
        assert tracker.calibration_point_count == 1
    
    def test_state_transitions(self, tracker):
        """Test state transitions through calibration flow."""
        assert tracker.state == TrackerState.IDLE
        
        tracker.start_calibration()
        assert tracker.state == TrackerState.CALIBRATING
        
        tracker.start_validation()
        assert tracker.state == TrackerState.VALIDATING
    
    def test_reset(self, tracker):
        """Test tracker reset."""
        tracker.start_calibration()
        tracker.add_calibration_point({
            "screenX": 0, "screenY": 0,
            "leftX": 0, "leftY": 0,
            "rightX": 0, "rightY": 0,
            "noseX": 0, "noseY": 0,
        })
        
        tracker.reset()
        
        assert tracker.state == TrackerState.IDLE
        assert tracker.calibration_point_count == 0


class TestConfig:
    """Tests for configuration system."""
    
    def test_default_config_values(self):
        """Test default configuration values."""
        from app.config import Config
        
        cfg = Config()
        
        assert cfg.model.svr_kernel == "poly"
        assert cfg.model.svr_c == 150.0
        assert cfg.calibration.points_per_calibration == 100
        assert cfg.server.port == 3226


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
