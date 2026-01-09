#!/usr/bin/env python3
# src/run_gui.py
"""
GUI Chat Application Entry Point
Launches the Gemini Chat GUI with server management and API key rotation.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

# Change to the project root for config file access
project_root = src_dir.parent
os.chdir(project_root)

def main():
    """Main entry point."""
    try:
        from gui_chat.chat_app import main as run_app
        run_app()
    except ImportError as e:
        print(f"Error: Failed to import GUI modules: {e}")
        print("Please ensure all dependencies are installed:")
        print("  poetry install")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
