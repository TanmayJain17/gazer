#!/usr/bin/env python3
"""
Gazer - Eye Tracking Application

Run with: python run.py
Then open: http://localhost:3226
"""

import app as gazer_app

if __name__ == "__main__":
    print("Starting Gazer server...")
    print("Open http://localhost:3226 in your browser")
    gazer_app.socketio.run(gazer_app.app, host='0.0.0.0', port=3226, debug=True)