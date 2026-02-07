"""
Gazer - Webcam-based Eye Tracking Application

A real-time eye tracking system using webcam input and machine learning
for gaze estimation. Uses MediaPipe for face landmark detection and
stacked SVR models for gaze prediction.

Usage:
    python -m app.main
    
    Or with gunicorn:
    gunicorn --worker-class eventlet -w 1 app.main:app
"""

import logging
import sys

from flask import Flask, render_template
from flask_cors import CORS
from flask_socketio import SocketIO

from app.config import config
from app.models import GazeTracker
from app.routes import register_handlers

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if config.server.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """
    Application factory for creating the Flask app.
    
    Returns:
        Configured Flask application instance
    """
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )
    
    # Enable CORS if configured
    if config.server.cors_enabled:
        CORS(app)
        logger.info("CORS enabled")
    
    # Register routes
    @app.route("/")
    def index():
        """Serve the instructions page."""
        return render_template("instructions.html")
    
    @app.route("/eyetracking")
    def eyetracking():
        """Serve the main eye tracking interface."""
        return render_template("eyetracking.html")
    
    @app.route("/health")
    def health():
        """Health check endpoint."""
        return {"status": "healthy", "version": "1.0.0"}
    
    logger.info("Flask app created")
    return app


def create_socketio(app: Flask) -> SocketIO:
    """
    Create and configure SocketIO instance.
    
    Args:
        app: Flask application instance
        
    Returns:
        Configured SocketIO instance
    """
    socketio = SocketIO(
        app,
        cors_allowed_origins="*" if config.server.cors_enabled else None,
        async_mode="eventlet",  # Use eventlet for production
        logger=config.server.debug,
        engineio_logger=config.server.debug,
    )
    
    logger.info("SocketIO configured")
    return socketio


# Create application instances
app = create_app()
socketio = create_socketio(app)

# Create tracker and register handlers
tracker = GazeTracker()
register_handlers(socketio, tracker)


def main():
    """Run the application."""
    logger.info(
        "Starting Gazer server on %s:%d (debug=%s)",
        config.server.host,
        config.server.port,
        config.server.debug,
    )
    
    # Build kwargs for socketio.run()
    run_kwargs = {
        "host": config.server.host,
        "port": config.server.port,
        "debug": config.server.debug,
    }
    
    # Only add SSL if enabled (eventlet has compatibility issues with ssl_context)
    if config.server.ssl_enabled:
        run_kwargs["certfile"] = config.server.ssl_cert_path
        run_kwargs["keyfile"] = config.server.ssl_key_path
        logger.info("SSL enabled")
    
    socketio.run(app, **run_kwargs)
