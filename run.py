#!/usr/bin/env python3
"""
Quick start script for Gazer eye tracking application.

Usage:
    python run.py
    
Or with custom settings:
    GAZER_SERVER_PORT=8080 python run.py
"""

from app.main import main

if __name__ == "__main__":
    main()
