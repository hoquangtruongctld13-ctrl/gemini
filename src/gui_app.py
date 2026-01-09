#!/usr/bin/env python3
# src/gui_app.py
"""
Gemini Chat GUI Application Entry Point

Run this file to start the desktop GUI chat application.
Usage: python src/gui_app.py
"""

import os
import sys

# Ensure we're running from the correct directory
if __name__ == "__main__":
    # Add src to path
    src_dir = os.path.dirname(os.path.abspath(__file__))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    # Change to project root for config file access
    project_root = os.path.dirname(src_dir)
    os.chdir(project_root)
    
    # Import and run the GUI
    from gui.chat_app import main
    main()
