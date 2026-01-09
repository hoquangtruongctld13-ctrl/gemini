#!/usr/bin/env python3
# run_gui.py
"""
Gemini Chat GUI Application Runner

A simple script to run the GUI chat application.
This script sets up the environment and launches the GUI.

Usage:
    poetry run python run_gui.py
    
    or if dependencies are installed:
    python run_gui.py
"""

import os
import sys


def main():
    """Main entry point."""
    # Get project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(project_root, "src")
    
    # Add src to Python path
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    # Change to project root for config file access
    os.chdir(project_root)
    
    # Check for tkinter availability
    try:
        import tkinter as tk
        del tk
    except ImportError:
        print("Error: tkinter is not installed.")
        print("Please install it using your system package manager:")
        print("  - Ubuntu/Debian: sudo apt-get install python3-tk")
        print("  - Fedora: sudo dnf install python3-tkinter")
        print("  - macOS: tkinter is included with Python from python.org")
        print("  - Windows: tkinter is included with Python installer")
        sys.exit(1)
    
    # Import and run the GUI
    try:
        from gui.chat_app import main as run_gui
        run_gui()
    except ImportError as e:
        print(f"Error importing GUI module: {e}")
        print("Make sure you are running from the project root directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()
