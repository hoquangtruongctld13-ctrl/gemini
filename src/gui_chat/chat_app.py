# src/gui_chat/chat_app.py
"""
Gemini Chat GUI Application
A modern graphical interface for chatting with Gemini models.
Features:
- Server start/stop button
- Model selection
- Chat history
- API key rotation support
- Dark/Light theme
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import asyncio
import threading
import queue
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import json
import time
import logging

# Add the parent directory to sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gui_chat.key_rotator import KeyRotator, AccountRotator

logger = logging.getLogger(__name__)


class Theme(Enum):
    """Application themes."""
    DARK = "dark"
    LIGHT = "light"


@dataclass
class ThemeColors:
    """Color scheme for a theme."""
    bg_primary: str
    bg_secondary: str
    bg_input: str
    fg_primary: str
    fg_secondary: str
    accent: str
    accent_hover: str
    border: str
    success: str
    error: str
    warning: str


# Theme definitions
THEMES = {
    Theme.DARK: ThemeColors(
        bg_primary="#1e1e1e",
        bg_secondary="#252526",
        bg_input="#2d2d2d",
        fg_primary="#ffffff",
        fg_secondary="#cccccc",
        accent="#007acc",
        accent_hover="#0098ff",
        border="#3c3c3c",
        success="#4ec9b0",
        error="#f14c4c",
        warning="#dcdcaa"
    ),
    Theme.LIGHT: ThemeColors(
        bg_primary="#ffffff",
        bg_secondary="#f3f3f3",
        bg_input="#ffffff",
        fg_primary="#1e1e1e",
        fg_secondary="#6e6e6e",
        accent="#0078d4",
        accent_hover="#106ebe",
        border="#d1d1d1",
        success="#107c10",
        error="#d13438",
        warning="#c19c00"
    )
}


class ServerStatus(Enum):
    """Server status states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class GeminiChatApp:
    """Main GUI Chat Application for Gemini models."""
    
    # Supported Gemini models
    MODELS = [
        "gemini-3.0-pro",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
    ]
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Gemini Chat - AI Assistant")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
        
        # State variables
        self.current_theme = Theme.DARK
        self.server_status = ServerStatus.STOPPED
        self.server_process = None
        self.server_thread = None
        self.chat_history = []
        self.selected_model = tk.StringVar(value=self.MODELS[0])
        
        # Key rotator for API keys
        self.key_rotator = KeyRotator()
        self.account_rotator = AccountRotator()
        
        # Queue for thread-safe UI updates
        self.ui_queue = queue.Queue()
        
        # Asyncio event loop for background tasks
        self.loop = None
        self.loop_thread = None
        
        # Configuration
        self.config = {
            "host": "localhost",
            "port": 6969,
            "auto_start_server": False,
            "api_keys_file": "",
            "accounts_file": ""
        }
        
        # Load configuration
        self.load_config()
        
        # Setup UI
        self._setup_styles()
        self._create_ui()
        self._apply_theme()
        
        # Start asyncio event loop in background thread
        self._start_async_loop()
        
        # Start UI update checker
        self._check_ui_queue()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Auto-start server if configured
        if self.config.get("auto_start_server"):
            self.root.after(1000, self._toggle_server)
    
    def _setup_styles(self):
        """Setup ttk styles."""
        self.style = ttk.Style()
        self.style.theme_use('clam')
    
    def _create_ui(self):
        """Create the main UI layout."""
        # Main container
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Create sidebar and main area
        self._create_sidebar()
        self._create_main_area()
    
    def _create_sidebar(self):
        """Create the sidebar with controls."""
        self.sidebar = tk.Frame(self.main_container, width=280)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)
        
        # App title
        title_frame = tk.Frame(self.sidebar)
        title_frame.pack(fill=tk.X, padx=15, pady=(20, 10))
        
        self.title_label = tk.Label(
            title_frame,
            text="✨ Gemini Chat",
            font=("Segoe UI", 18, "bold")
        )
        self.title_label.pack(anchor=tk.W)
        
        self.subtitle_label = tk.Label(
            title_frame,
            text="AI Assistant",
            font=("Segoe UI", 10)
        )
        self.subtitle_label.pack(anchor=tk.W)
        
        # Separator
        ttk.Separator(self.sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=15, pady=10)
        
        # Server control section
        server_frame = tk.Frame(self.sidebar)
        server_frame.pack(fill=tk.X, padx=15, pady=5)
        
        self.server_label = tk.Label(
            server_frame,
            text="🖥️ Server Control",
            font=("Segoe UI", 11, "bold")
        )
        self.server_label.pack(anchor=tk.W)
        
        # Server status
        status_frame = tk.Frame(server_frame)
        status_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.status_indicator = tk.Label(
            status_frame,
            text="●",
            font=("Segoe UI", 12)
        )
        self.status_indicator.pack(side=tk.LEFT)
        
        self.status_text = tk.Label(
            status_frame,
            text="Server Stopped",
            font=("Segoe UI", 10)
        )
        self.status_text.pack(side=tk.LEFT, padx=(5, 0))
        
        # Server toggle button
        self.server_button = tk.Button(
            server_frame,
            text="▶  Start Server",
            font=("Segoe UI", 10),
            command=self._toggle_server,
            cursor="hand2",
            relief=tk.FLAT,
            padx=20,
            pady=8
        )
        self.server_button.pack(fill=tk.X, pady=(10, 0))
        
        # Server URL
        self.url_frame = tk.Frame(server_frame)
        self.url_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.url_label = tk.Label(
            self.url_frame,
            text=f"http://{self.config['host']}:{self.config['port']}",
            font=("Consolas", 9),
            cursor="hand2"
        )
        self.url_label.pack(anchor=tk.W)
        self.url_label.bind("<Button-1>", self._copy_url)
        
        # Separator
        ttk.Separator(self.sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=15, pady=10)
        
        # Model selection
        model_frame = tk.Frame(self.sidebar)
        model_frame.pack(fill=tk.X, padx=15, pady=5)
        
        model_label = tk.Label(
            model_frame,
            text="🤖 Model Selection",
            font=("Segoe UI", 11, "bold")
        )
        model_label.pack(anchor=tk.W)
        
        self.model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.selected_model,
            values=self.MODELS,
            state="readonly",
            font=("Segoe UI", 10)
        )
        self.model_combo.pack(fill=tk.X, pady=(5, 0))
        
        # Separator
        ttk.Separator(self.sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=15, pady=10)
        
        # API Keys section
        keys_frame = tk.Frame(self.sidebar)
        keys_frame.pack(fill=tk.X, padx=15, pady=5)
        
        keys_label = tk.Label(
            keys_frame,
            text="🔑 API Key Rotation",
            font=("Segoe UI", 11, "bold")
        )
        keys_label.pack(anchor=tk.W)
        
        self.keys_status_label = tk.Label(
            keys_frame,
            text="No API keys loaded",
            font=("Segoe UI", 9)
        )
        self.keys_status_label.pack(anchor=tk.W, pady=(5, 0))
        
        keys_buttons = tk.Frame(keys_frame)
        keys_buttons.pack(fill=tk.X, pady=(5, 0))
        
        self.load_keys_btn = tk.Button(
            keys_buttons,
            text="Load Keys",
            font=("Segoe UI", 9),
            command=self._load_api_keys,
            cursor="hand2",
            relief=tk.FLAT,
            padx=10,
            pady=4
        )
        self.load_keys_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        
        self.add_key_btn = tk.Button(
            keys_buttons,
            text="Add Key",
            font=("Segoe UI", 9),
            command=self._add_api_key,
            cursor="hand2",
            relief=tk.FLAT,
            padx=10,
            pady=4
        )
        self.add_key_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))
        
        # Separator
        ttk.Separator(self.sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=15, pady=10)
        
        # Theme toggle
        theme_frame = tk.Frame(self.sidebar)
        theme_frame.pack(fill=tk.X, padx=15, pady=5)
        
        self.theme_button = tk.Button(
            theme_frame,
            text="🌙 Dark Mode",
            font=("Segoe UI", 10),
            command=self._toggle_theme,
            cursor="hand2",
            relief=tk.FLAT,
            padx=10,
            pady=6
        )
        self.theme_button.pack(fill=tk.X)
        
        # Bottom buttons
        bottom_frame = tk.Frame(self.sidebar)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=15)
        
        self.clear_btn = tk.Button(
            bottom_frame,
            text="🗑️ Clear Chat",
            font=("Segoe UI", 10),
            command=self._clear_chat,
            cursor="hand2",
            relief=tk.FLAT,
            padx=10,
            pady=6
        )
        self.clear_btn.pack(fill=tk.X, pady=(0, 5))
        
        self.settings_btn = tk.Button(
            bottom_frame,
            text="⚙️ Settings",
            font=("Segoe UI", 10),
            command=self._open_settings,
            cursor="hand2",
            relief=tk.FLAT,
            padx=10,
            pady=6
        )
        self.settings_btn.pack(fill=tk.X)
    
    def _create_main_area(self):
        """Create the main chat area."""
        self.main_area = tk.Frame(self.main_container)
        self.main_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Chat display
        self.chat_frame = tk.Frame(self.main_area)
        self.chat_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(20, 10))
        
        # Chat text widget with scrollbar
        self.chat_display = scrolledtext.ScrolledText(
            self.chat_frame,
            wrap=tk.WORD,
            font=("Segoe UI", 11),
            state=tk.DISABLED,
            cursor="arrow",
            relief=tk.FLAT,
            padx=15,
            pady=15
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        
        # Configure text tags for formatting
        self.chat_display.tag_configure("user", foreground="#4fc3f7", font=("Segoe UI", 11, "bold"))
        self.chat_display.tag_configure("assistant", foreground="#81c784", font=("Segoe UI", 11, "bold"))
        self.chat_display.tag_configure("system", foreground="#ffb74d", font=("Segoe UI", 10, "italic"))
        self.chat_display.tag_configure("error", foreground="#f44336", font=("Segoe UI", 10))
        
        # Input area
        self.input_frame = tk.Frame(self.main_area)
        self.input_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        # Text input with border frame
        input_container = tk.Frame(self.input_frame, relief=tk.FLAT, bd=1)
        input_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.input_text = tk.Text(
            input_container,
            height=3,
            font=("Segoe UI", 11),
            wrap=tk.WORD,
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.input_text.pack(fill=tk.BOTH, expand=True)
        
        # Bind Enter to send (Shift+Enter for new line)
        self.input_text.bind("<Return>", self._on_enter)
        self.input_text.bind("<Shift-Return>", self._on_shift_enter)
        
        # Send button
        self.send_btn = tk.Button(
            self.input_frame,
            text="Send ➤",
            font=("Segoe UI", 11, "bold"),
            command=self._send_message,
            cursor="hand2",
            relief=tk.FLAT,
            padx=25,
            pady=10
        )
        self.send_btn.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Add welcome message
        self._add_system_message("Welcome to Gemini Chat! Start the server and begin chatting.")
    
    def _apply_theme(self):
        """Apply the current theme to all widgets."""
        colors = THEMES[self.current_theme]
        
        # Configure root window
        self.root.configure(bg=colors.bg_secondary)
        
        # Configure sidebar
        self.sidebar.configure(bg=colors.bg_primary)
        
        # Configure all widgets in sidebar
        for widget in self.sidebar.winfo_children():
            self._apply_theme_to_frame(widget, colors)
        
        # Configure main area
        self.main_area.configure(bg=colors.bg_secondary)
        self.chat_frame.configure(bg=colors.bg_secondary)
        
        # Configure chat display
        self.chat_display.configure(
            bg=colors.bg_input,
            fg=colors.fg_primary,
            insertbackground=colors.fg_primary
        )
        
        # Configure input area
        self.input_frame.configure(bg=colors.bg_secondary)
        self.input_text.configure(
            bg=colors.bg_input,
            fg=colors.fg_primary,
            insertbackground=colors.fg_primary
        )
        
        # Configure buttons
        self._style_button(self.server_button, colors, is_primary=True)
        self._style_button(self.send_btn, colors, is_primary=True)
        self._style_button(self.load_keys_btn, colors)
        self._style_button(self.add_key_btn, colors)
        self._style_button(self.theme_button, colors)
        self._style_button(self.clear_btn, colors)
        self._style_button(self.settings_btn, colors)
        
        # Update status indicator
        self._update_server_status_display()
        
        # Update theme button text
        self.theme_button.configure(
            text="☀️ Light Mode" if self.current_theme == Theme.DARK else "🌙 Dark Mode"
        )
    
    def _apply_theme_to_frame(self, widget, colors: ThemeColors):
        """Recursively apply theme to a frame and its children."""
        widget_type = widget.winfo_class()
        
        if widget_type == "Frame":
            widget.configure(bg=colors.bg_primary)
            for child in widget.winfo_children():
                self._apply_theme_to_frame(child, colors)
        elif widget_type == "Label":
            widget.configure(bg=colors.bg_primary, fg=colors.fg_primary)
        elif widget_type == "Button":
            self._style_button(widget, colors)
    
    def _style_button(self, button: tk.Button, colors: ThemeColors, is_primary: bool = False):
        """Style a button with theme colors."""
        if is_primary:
            button.configure(
                bg=colors.accent,
                fg="#ffffff",
                activebackground=colors.accent_hover,
                activeforeground="#ffffff"
            )
        else:
            button.configure(
                bg=colors.bg_secondary,
                fg=colors.fg_primary,
                activebackground=colors.border,
                activeforeground=colors.fg_primary
            )
    
    def _toggle_theme(self):
        """Toggle between dark and light themes."""
        self.current_theme = Theme.LIGHT if self.current_theme == Theme.DARK else Theme.DARK
        self._apply_theme()
    
    def _toggle_server(self):
        """Start or stop the server."""
        if self.server_status in (ServerStatus.STOPPED, ServerStatus.ERROR):
            self._start_server()
        elif self.server_status == ServerStatus.RUNNING:
            self._stop_server()
    
    def _start_server(self):
        """Start the FastAPI server in a background thread."""
        self.server_status = ServerStatus.STARTING
        self._update_server_status_display()
        
        def run_server():
            try:
                import uvicorn
                from app.main import app as webai_app
                
                # Update UI via queue
                self.ui_queue.put(("server_status", ServerStatus.RUNNING))
                self.ui_queue.put(("log", "Server started successfully"))
                
                # Run the server
                config = uvicorn.Config(
                    webai_app,
                    host=self.config["host"],
                    port=self.config["port"],
                    log_config=None
                )
                server = uvicorn.Server(config)
                self.server_instance = server
                
                asyncio.run(server.serve())
                
            except ImportError as e:
                self.ui_queue.put(("server_status", ServerStatus.ERROR))
                self.ui_queue.put(("error", f"Failed to import server modules: {e}"))
            except Exception as e:
                self.ui_queue.put(("server_status", ServerStatus.ERROR))
                self.ui_queue.put(("error", f"Server error: {e}"))
            finally:
                self.ui_queue.put(("server_status", ServerStatus.STOPPED))
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
    
    def _stop_server(self):
        """Stop the server."""
        self.server_status = ServerStatus.STOPPING
        self._update_server_status_display()
        
        if hasattr(self, 'server_instance') and self.server_instance:
            self.server_instance.should_exit = True
        
        self.server_status = ServerStatus.STOPPED
        self._update_server_status_display()
        self._add_system_message("Server stopped")
    
    def _update_server_status_display(self):
        """Update the server status indicator and button."""
        colors = THEMES[self.current_theme]
        
        status_configs = {
            ServerStatus.STOPPED: ("●", "Server Stopped", colors.fg_secondary, "▶  Start Server"),
            ServerStatus.STARTING: ("●", "Starting...", colors.warning, "Starting..."),
            ServerStatus.RUNNING: ("●", "Server Running", colors.success, "■  Stop Server"),
            ServerStatus.STOPPING: ("●", "Stopping...", colors.warning, "Stopping..."),
            ServerStatus.ERROR: ("●", "Error", colors.error, "▶  Retry")
        }
        
        indicator, text, color, btn_text = status_configs[self.server_status]
        
        self.status_indicator.configure(text=indicator, fg=color)
        self.status_text.configure(text=text, fg=color)
        self.server_button.configure(text=btn_text)
    
    def _send_message(self):
        """Send a message to the chat."""
        message = self.input_text.get("1.0", tk.END).strip()
        if not message:
            return
        
        # Clear input
        self.input_text.delete("1.0", tk.END)
        
        # Add user message to display
        self._add_user_message(message)
        
        # Check server status
        if self.server_status != ServerStatus.RUNNING:
            self._add_error_message("Server is not running. Please start the server first.")
            return
        
        # Send message in background
        self._send_message_async(message)
    
    def _send_message_async(self, message: str):
        """Send message to server asynchronously."""
        def send():
            try:
                import httpx
                
                url = f"http://{self.config['host']}:{self.config['port']}/v1/chat/completions"
                payload = {
                    "model": self.selected_model.get(),
                    "messages": [{"role": "user", "content": message}]
                }
                
                with httpx.Client(timeout=60.0) as client:
                    response = client.post(url, json=payload)
                    response.raise_for_status()
                    
                    data = response.json()
                    assistant_message = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    
                    self.ui_queue.put(("assistant_message", assistant_message))
                    
            except Exception as e:
                self.ui_queue.put(("error", f"Failed to get response: {e}"))
        
        threading.Thread(target=send, daemon=True).start()
    
    def _add_user_message(self, message: str):
        """Add a user message to the chat display."""
        self._append_to_chat(f"You: {message}\n\n", "user")
        self.chat_history.append({"role": "user", "content": message})
    
    def _add_assistant_message(self, message: str):
        """Add an assistant message to the chat display."""
        self._append_to_chat(f"Gemini: {message}\n\n", "assistant")
        self.chat_history.append({"role": "assistant", "content": message})
    
    def _add_system_message(self, message: str):
        """Add a system message to the chat display."""
        self._append_to_chat(f"[System] {message}\n\n", "system")
    
    def _add_error_message(self, message: str):
        """Add an error message to the chat display."""
        self._append_to_chat(f"[Error] {message}\n\n", "error")
    
    def _append_to_chat(self, text: str, tag: str = None):
        """Append text to the chat display."""
        self.chat_display.configure(state=tk.NORMAL)
        if tag:
            self.chat_display.insert(tk.END, text, tag)
        else:
            self.chat_display.insert(tk.END, text)
        self.chat_display.configure(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def _clear_chat(self):
        """Clear the chat history."""
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.configure(state=tk.DISABLED)
        self.chat_history.clear()
        self._add_system_message("Chat cleared")
    
    def _on_enter(self, event):
        """Handle Enter key press."""
        self._send_message()
        return "break"  # Prevent default newline
    
    def _on_shift_enter(self, event):
        """Handle Shift+Enter key press (new line)."""
        return  # Allow default behavior
    
    def _copy_url(self, event=None):
        """Copy server URL to clipboard."""
        url = f"http://{self.config['host']}:{self.config['port']}"
        self.root.clipboard_clear()
        self.root.clipboard_append(url)
        self._add_system_message(f"URL copied: {url}")
    
    def _load_api_keys(self):
        """Load API keys from a file."""
        filepath = filedialog.askopenfilename(
            title="Select API Keys File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filepath:
            count = self.key_rotator.load_keys_from_file(filepath)
            self._update_keys_status()
            if count > 0:
                self._add_system_message(f"Loaded {count} API keys")
            else:
                self._add_error_message("No API keys loaded from file")
    
    def _add_api_key(self):
        """Add a single API key via dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Add API Key")
        dialog.geometry("400x150")
        dialog.transient(self.root)
        dialog.grab_set()
        
        colors = THEMES[self.current_theme]
        dialog.configure(bg=colors.bg_primary)
        
        tk.Label(
            dialog,
            text="Enter Gemini API Key:",
            font=("Segoe UI", 11),
            bg=colors.bg_primary,
            fg=colors.fg_primary
        ).pack(pady=(20, 5))
        
        entry = tk.Entry(dialog, width=50, font=("Segoe UI", 10), show="*")
        entry.pack(pady=5, padx=20)
        entry.focus()
        
        def add():
            key = entry.get().strip()
            if key:
                if self.key_rotator.add_key(key):
                    self._update_keys_status()
                    self._add_system_message("API key added successfully")
                else:
                    messagebox.showwarning("Warning", "API key already exists")
            dialog.destroy()
        
        tk.Button(
            dialog,
            text="Add Key",
            command=add,
            font=("Segoe UI", 10),
            bg=colors.accent,
            fg="#ffffff",
            relief=tk.FLAT,
            padx=20,
            pady=5
        ).pack(pady=10)
    
    def _update_keys_status(self):
        """Update the API keys status display."""
        stats = self.key_rotator.get_stats()
        total = stats["total_keys"]
        active = stats["active_keys"]
        
        if total == 0:
            text = "No API keys loaded"
        else:
            text = f"{active}/{total} keys active"
        
        self.keys_status_label.configure(text=text)
    
    def _open_settings(self):
        """Open the settings dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Settings")
        dialog.geometry("450x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        colors = THEMES[self.current_theme]
        dialog.configure(bg=colors.bg_primary)
        
        # Server settings
        tk.Label(
            dialog,
            text="Server Settings",
            font=("Segoe UI", 12, "bold"),
            bg=colors.bg_primary,
            fg=colors.fg_primary
        ).pack(pady=(20, 10), padx=20, anchor=tk.W)
        
        # Host
        host_frame = tk.Frame(dialog, bg=colors.bg_primary)
        host_frame.pack(fill=tk.X, padx=20, pady=5)
        
        tk.Label(
            host_frame,
            text="Host:",
            font=("Segoe UI", 10),
            bg=colors.bg_primary,
            fg=colors.fg_primary,
            width=12,
            anchor=tk.W
        ).pack(side=tk.LEFT)
        
        host_entry = tk.Entry(host_frame, font=("Segoe UI", 10))
        host_entry.insert(0, self.config["host"])
        host_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Port
        port_frame = tk.Frame(dialog, bg=colors.bg_primary)
        port_frame.pack(fill=tk.X, padx=20, pady=5)
        
        tk.Label(
            port_frame,
            text="Port:",
            font=("Segoe UI", 10),
            bg=colors.bg_primary,
            fg=colors.fg_primary,
            width=12,
            anchor=tk.W
        ).pack(side=tk.LEFT)
        
        port_entry = tk.Entry(port_frame, font=("Segoe UI", 10))
        port_entry.insert(0, str(self.config["port"]))
        port_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Auto-start checkbox
        auto_start_var = tk.BooleanVar(value=self.config["auto_start_server"])
        auto_start_check = tk.Checkbutton(
            dialog,
            text="Auto-start server on launch",
            variable=auto_start_var,
            font=("Segoe UI", 10),
            bg=colors.bg_primary,
            fg=colors.fg_primary,
            selectcolor=colors.bg_secondary,
            activebackground=colors.bg_primary,
            activeforeground=colors.fg_primary
        )
        auto_start_check.pack(pady=10, padx=20, anchor=tk.W)
        
        # Save button
        def save_settings():
            try:
                self.config["host"] = host_entry.get().strip() or "localhost"
                self.config["port"] = int(port_entry.get().strip() or "6969")
                self.config["auto_start_server"] = auto_start_var.get()
                self.save_config()
                
                # Update URL label
                self.url_label.configure(text=f"http://{self.config['host']}:{self.config['port']}")
                
                self._add_system_message("Settings saved")
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Error", "Invalid port number")
        
        tk.Button(
            dialog,
            text="Save Settings",
            command=save_settings,
            font=("Segoe UI", 10),
            bg=colors.accent,
            fg="#ffffff",
            relief=tk.FLAT,
            padx=20,
            pady=8
        ).pack(pady=20)
    
    def _start_async_loop(self):
        """Start the asyncio event loop in a background thread."""
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()
        
        self.loop_thread = threading.Thread(target=run_loop, daemon=True)
        self.loop_thread.start()
    
    def _check_ui_queue(self):
        """Check the UI queue for updates from background threads."""
        try:
            while True:
                msg_type, data = self.ui_queue.get_nowait()
                
                if msg_type == "server_status":
                    self.server_status = data
                    self._update_server_status_display()
                elif msg_type == "assistant_message":
                    self._add_assistant_message(data)
                elif msg_type == "error":
                    self._add_error_message(data)
                elif msg_type == "log":
                    self._add_system_message(data)
                    
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self._check_ui_queue)
    
    def load_config(self):
        """Load configuration from file."""
        config_path = Path(__file__).parent.parent.parent / "gui_config.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    saved_config = json.load(f)
                    self.config.update(saved_config)
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
    
    def save_config(self):
        """Save configuration to file."""
        config_path = Path(__file__).parent.parent.parent / "gui_config.json"
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def _on_close(self):
        """Handle window close event."""
        if self.server_status == ServerStatus.RUNNING:
            if messagebox.askyesno("Confirm", "Server is running. Stop server and exit?"):
                self._stop_server()
            else:
                return
        
        # Save config
        self.save_config()
        
        # Stop asyncio loop
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        
        self.root.destroy()


def main():
    """Main entry point for the GUI application."""
    root = tk.Tk()
    
    # Set DPI awareness on Windows
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    
    app = GeminiChatApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
