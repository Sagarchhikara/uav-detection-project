#!/usr/bin/env python3
"""
Simple launcher for the Anti-UAV Streamlit app with error handling
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are available"""
    try:
        import streamlit
        import pandas
        import plotly
        import cv2
        import numpy
        print("✅ All dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def main():
    print("🚁 Anti-UAV Detection System - Web Interface Launcher")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("ui/streamlit_app.py").exists():
        print("❌ Error: streamlit_app.py not found!")
        print("Make sure you're running this from the project root directory.")
        return
    
    # Check dependencies
    if not check_dependencies():
        print("Installing missing dependencies...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'streamlit', 'pandas', 'plotly'])
    
    print("\n🚀 Launching Streamlit application...")
    print("📱 The app will open in your browser at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 
            'ui/streamlit_app.py',
            '--server.port', '8501',
            '--server.address', 'localhost',
            '--server.headless', 'false'
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down Anti-UAV Detection System...")
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're in the project root directory")
        print("2. Check that all dependencies are installed: pip install -r requirements.txt")
        print("3. Try running directly: streamlit run ui/streamlit_app.py")

if __name__ == "__main__":
    main()