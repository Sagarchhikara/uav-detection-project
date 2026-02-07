#!/usr/bin/env python3
"""
Quick launcher for the Anti-UAV Detection System
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit application"""
    
    print("🚁 Launching Anti-UAV Detection System...")
    
    # Check if we're in the right directory
    if not Path("ui/streamlit_app.py").exists():
        print("❌ Error: streamlit_app.py not found!")
        print("Make sure you're running this from the project root directory.")
        sys.exit(1)
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("❌ Streamlit not installed!")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 
            'ui/streamlit_app.py',
            '--server.port', '8501',
            '--server.address', 'localhost'
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Error launching app: {e}")

if __name__ == "__main__":
    main()