# src/gui/chat_app.py
"""
Gemini Chat GUI Application

A desktop GUI application for chatting with Gemini models.
Features:
- Auto-start server with one click
- Real-time chat interface
- Model selection dropdown
- Message history display
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import asyncio
import sys
import os
import time
import queue
from typing import Optional
import httpx

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ServerManager:
    """Manages the FastAPI server process."""
    
    def __init__(self, host: str = "localhost", port: int = 6969):
        self.host = host
        self.port = port
        self.process: Optional[object] = None
        self.is_running = False
        self._stop_event: Optional[object] = None
        
    def start(self) -> bool:
        """Start the server in a background process."""
        if self.is_running:
            return True
            
        try:
            import multiprocessing
            from app.config import load_config
            from app.main import app as webai_app
            from app.services.gemini_client import init_gemini_client
            import uvicorn
            import signal
            
            # Initialize the client in a separate thread to avoid blocking
            # This runs the async initialization in a new event loop
            init_result = [False]
            init_error = [None]
            
            def init_client_sync():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        init_result[0] = loop.run_until_complete(init_gemini_client())
                    finally:
                        loop.close()
                except Exception as e:
                    init_error[0] = str(e)
            
            # Run initialization in current thread (called from background thread)
            init_client_sync()
            
            if init_error[0]:
                print(f"Client initialization error: {init_error[0]}")
                return False
                
            if not init_result[0]:
                return False
            
            self._stop_event = multiprocessing.Event()
            
            def run_server(host, port, stop_event):
                """Server process function."""
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                if sys.platform == "win32":
                    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
                
                config = uvicorn.Config(
                    webai_app, host=host, port=port, log_config=None
                )
                server = uvicorn.Server(config)
                
                def shutdown_monitor():
                    stop_event.wait()
                    server.should_exit = True
                
                monitor_thread = threading.Thread(target=shutdown_monitor, daemon=True)
                monitor_thread.start()
                
                server.run()
            
            self.process = multiprocessing.Process(
                target=run_server,
                args=(self.host, self.port, self._stop_event)
            )
            self.process.start()
            
            # Wait for server to be ready with health check
            if self._wait_for_server():
                self.is_running = True
                return True
            else:
                # Server didn't start properly, clean up
                self.stop()
                return False
            
        except Exception as e:
            print(f"Failed to start server: {e}")
            return False
    
    def _wait_for_server(self, timeout: int = 10, check_interval: float = 0.5) -> bool:
        """Wait for the server to be ready by polling the health endpoint."""
        import httpx
        
        start_time = time.time()
        url = f"http://{self.host}:{self.port}/docs"
        
        while time.time() - start_time < timeout:
            try:
                response = httpx.get(url, timeout=1.0)
                if response.status_code == 200:
                    return True
            except (httpx.ConnectError, httpx.TimeoutException):
                pass
            time.sleep(check_interval)
        
        return False
    
    def stop(self):
        """Stop the server."""
        if self._stop_event:
            self._stop_event.set()
        
        if self.process and self.process.is_alive():
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
        
        self.is_running = False
        self.process = None
        self._stop_event = None
    
    def get_base_url(self) -> str:
        """Get the server base URL."""
        return f"http://{self.host}:{self.port}"


class GeminiChatGUI:
    """Main GUI Application for Gemini Chat."""
    
    # Available models
    MODELS = [
        ("Gemini 3.0 Pro", "gemini-3.0-pro"),
        ("Gemini 2.5 Pro", "gemini-2.5-pro"),
        ("Gemini 2.5 Flash", "gemini-2.5-flash"),
    ]
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Gemini Chat - WebAI-to-API")
        self.root.geometry("900x700")
        self.root.minsize(600, 500)
        
        # Server manager
        self.server_manager = ServerManager()
        
        # Message queue for thread-safe UI updates
        self.message_queue = queue.Queue()
        
        # HTTP client
        self.http_client = httpx.AsyncClient(timeout=120.0)
        
        # Build UI
        self._setup_styles()
        self._build_ui()
        
        # Start message processor
        self._process_messages()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
    def _setup_styles(self):
        """Configure ttk styles."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('TFrame', background='#2b2b2b')
        style.configure('TLabel', background='#2b2b2b', foreground='#ffffff')
        style.configure('TButton', padding=6)
        style.configure('Header.TLabel', font=('Segoe UI', 14, 'bold'))
        style.configure('Status.TLabel', font=('Segoe UI', 10))
        
        # Configure combobox
        style.configure('TCombobox', fieldbackground='#3c3c3c', background='#3c3c3c')
        
    def _build_ui(self):
        """Build the main UI."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        self._build_header(main_frame)
        
        # Server control
        self._build_server_control(main_frame)
        
        # Model selection
        self._build_model_selection(main_frame)
        
        # Chat area
        self._build_chat_area(main_frame)
        
        # Input area
        self._build_input_area(main_frame)
        
    def _build_header(self, parent):
        """Build header section."""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(
            header_frame, 
            text="🤖 Gemini Chat",
            style='Header.TLabel'
        )
        title_label.pack(side=tk.LEFT)
        
        version_label = ttk.Label(
            header_frame,
            text="WebAI-to-API GUI",
            style='Status.TLabel'
        )
        version_label.pack(side=tk.RIGHT)
        
    def _build_server_control(self, parent):
        """Build server control section."""
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Server status indicator
        self.status_indicator = tk.Canvas(
            control_frame, width=12, height=12, 
            bg='#2b2b2b', highlightthickness=0
        )
        self.status_indicator.pack(side=tk.LEFT, padx=(0, 5))
        self._update_status_indicator(False)
        
        self.status_label = ttk.Label(
            control_frame, 
            text="Server: Stopped",
            style='Status.TLabel'
        )
        self.status_label.pack(side=tk.LEFT)
        
        # Control buttons
        self.stop_btn = ttk.Button(
            control_frame,
            text="⏹ Stop Server",
            command=self._stop_server,
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.RIGHT, padx=5)
        
        self.start_btn = ttk.Button(
            control_frame,
            text="▶ Start Server",
            command=self._start_server
        )
        self.start_btn.pack(side=tk.RIGHT)
        
    def _build_model_selection(self, parent):
        """Build model selection section."""
        model_frame = ttk.Frame(parent)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        model_label = ttk.Label(model_frame, text="Model:")
        model_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.model_var = tk.StringVar(value=self.MODELS[0][1])
        self.model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            values=[m[0] for m in self.MODELS],
            state="readonly",
            width=25
        )
        self.model_combo.current(0)
        self.model_combo.pack(side=tk.LEFT)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_change)
        
    def _build_chat_area(self, parent):
        """Build chat display area."""
        chat_frame = ttk.Frame(parent)
        chat_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Chat display
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=('Consolas', 11),
            bg='#1e1e1e',
            fg='#d4d4d4',
            insertbackground='#ffffff',
            selectbackground='#264f78',
            padx=10,
            pady=10,
            state=tk.DISABLED
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        
        # Configure tags for styling
        self.chat_display.tag_configure('user', foreground='#569cd6', font=('Consolas', 11, 'bold'))
        self.chat_display.tag_configure('assistant', foreground='#4ec9b0', font=('Consolas', 11, 'bold'))
        self.chat_display.tag_configure('system', foreground='#ce9178', font=('Consolas', 10, 'italic'))
        self.chat_display.tag_configure('error', foreground='#f14c4c')
        
        # Welcome message
        self._append_message("System", "Welcome to Gemini Chat! Start the server to begin chatting.", 'system')
        
    def _build_input_area(self, parent):
        """Build input area."""
        input_frame = ttk.Frame(parent)
        input_frame.pack(fill=tk.X)
        
        # Message input
        self.message_input = tk.Text(
            input_frame,
            height=3,
            font=('Consolas', 11),
            bg='#3c3c3c',
            fg='#d4d4d4',
            insertbackground='#ffffff',
            selectbackground='#264f78',
            padx=10,
            pady=5
        )
        self.message_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.message_input.bind('<Return>', self._on_enter_key)
        self.message_input.bind('<Shift-Return>', lambda e: None)  # Allow Shift+Enter for new line
        
        # Send button
        self.send_btn = ttk.Button(
            input_frame,
            text="Send",
            command=self._send_message,
            state=tk.DISABLED
        )
        self.send_btn.pack(side=tk.RIGHT, fill=tk.Y)
        
    def _update_status_indicator(self, is_running: bool):
        """Update the status indicator color."""
        self.status_indicator.delete("all")
        color = "#4caf50" if is_running else "#f44336"
        self.status_indicator.create_oval(2, 2, 10, 10, fill=color, outline=color)
        
    def _start_server(self):
        """Start the server in background."""
        self.start_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Server: Starting...")
        self._append_message("System", "Starting server...", 'system')
        
        def start_thread():
            success = self.server_manager.start()
            self.message_queue.put(('server_started', success))
        
        thread = threading.Thread(target=start_thread, daemon=True)
        thread.start()
        
    def _stop_server(self):
        """Stop the server."""
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Server: Stopping...")
        
        def stop_thread():
            self.server_manager.stop()
            self.message_queue.put(('server_stopped', None))
        
        thread = threading.Thread(target=stop_thread, daemon=True)
        thread.start()
        
    def _send_message(self):
        """Send message to the API."""
        if not self.server_manager.is_running:
            messagebox.showwarning("Server Not Running", "Please start the server first.")
            return
            
        message = self.message_input.get("1.0", tk.END).strip()
        if not message:
            return
            
        # Clear input
        self.message_input.delete("1.0", tk.END)
        
        # Display user message
        self._append_message("You", message, 'user')
        
        # Disable input while processing
        self.send_btn.config(state=tk.DISABLED)
        self.message_input.config(state=tk.DISABLED)
        
        # Send request in background
        def send_thread():
            asyncio.run(self._async_send_message(message))
        
        thread = threading.Thread(target=send_thread, daemon=True)
        thread.start()
        
    async def _async_send_message(self, message: str):
        """Async method to send message to API."""
        try:
            # Get selected model
            selected_index = self.model_combo.current()
            model = self.MODELS[selected_index][1]
            
            # Send request
            response = await self.http_client.post(
                f"{self.server_manager.get_base_url()}/gemini",
                json={"message": message, "model": model}
            )
            
            if response.status_code == 200:
                data = response.json()
                self.message_queue.put(('response', data.get('response', 'No response')))
            else:
                error_detail = response.json().get('detail', response.text)
                self.message_queue.put(('error', f"Error {response.status_code}: {error_detail}"))
                
        except Exception as e:
            self.message_queue.put(('error', f"Connection error: {str(e)}"))
            
    def _on_enter_key(self, event):
        """Handle Enter key press."""
        if not event.state & 0x1:  # If Shift is not pressed
            self._send_message()
            return 'break'  # Prevent default newline
            
    def _on_model_change(self, event):
        """Handle model selection change."""
        selected_index = self.model_combo.current()
        model_name, model_value = self.MODELS[selected_index]
        self.model_var.set(model_value)
        self._append_message("System", f"Model changed to: {model_name}", 'system')
        
    def _append_message(self, sender: str, message: str, tag: str):
        """Append a message to the chat display."""
        self.chat_display.config(state=tk.NORMAL)
        
        # Add timestamp
        timestamp = time.strftime("%H:%M")
        
        # Add sender line
        self.chat_display.insert(tk.END, f"\n[{timestamp}] {sender}:\n", tag)
        
        # Add message content
        self.chat_display.insert(tk.END, f"{message}\n")
        
        # Auto-scroll to bottom
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
        
    def _process_messages(self):
        """Process messages from the queue."""
        try:
            while True:
                msg_type, data = self.message_queue.get_nowait()
                
                if msg_type == 'server_started':
                    if data:  # Success
                        self.status_label.config(text="Server: Running")
                        self._update_status_indicator(True)
                        self.start_btn.config(state=tk.DISABLED)
                        self.stop_btn.config(state=tk.NORMAL)
                        self.send_btn.config(state=tk.NORMAL)
                        self._append_message("System", f"Server started at {self.server_manager.get_base_url()}", 'system')
                    else:  # Failed
                        self.status_label.config(text="Server: Failed to start")
                        self._update_status_indicator(False)
                        self.start_btn.config(state=tk.NORMAL)
                        self._append_message(
                            "System", 
                            "Failed to start server. Gemini cookies not found or invalid.\n\n"
                            "To fix this:\n"
                            "1. Open https://gemini.google.com and log in\n"
                            "2. Press F12 > Application > Cookies > gemini.google.com\n"
                            "3. Copy '__Secure-1PSID' and '__Secure-1PSIDTS' values\n"
                            "4. Paste them in config.conf under [Cookies]\n\n"
                            "See TROUBLESHOOTING.md for detailed instructions.", 
                            'error'
                        )
                        
                elif msg_type == 'server_stopped':
                    self.status_label.config(text="Server: Stopped")
                    self._update_status_indicator(False)
                    self.start_btn.config(state=tk.NORMAL)
                    self.stop_btn.config(state=tk.DISABLED)
                    self.send_btn.config(state=tk.DISABLED)
                    self._append_message("System", "Server stopped.", 'system')
                    
                elif msg_type == 'response':
                    self._append_message("Gemini", data, 'assistant')
                    self.send_btn.config(state=tk.NORMAL)
                    self.message_input.config(state=tk.NORMAL)
                    self.message_input.focus()
                    
                elif msg_type == 'error':
                    self._append_message("Error", data, 'error')
                    self.send_btn.config(state=tk.NORMAL)
                    self.message_input.config(state=tk.NORMAL)
                    
        except queue.Empty:
            pass
            
        # Schedule next check
        self.root.after(100, self._process_messages)
        
    def _on_close(self):
        """Handle window close."""
        if self.server_manager.is_running:
            if messagebox.askyesno("Confirm Exit", "Server is running. Stop server and exit?"):
                self.server_manager.stop()
            else:
                return
        
        # Close HTTP client in a separate thread to avoid blocking
        def close_client():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self.http_client.aclose())
                finally:
                    loop.close()
            except Exception:
                pass  # Ignore errors on close
        
        close_thread = threading.Thread(target=close_client, daemon=True)
        close_thread.start()
        close_thread.join(timeout=2)  # Wait up to 2 seconds for cleanup
        
        self.root.destroy()


def main():
    """Main entry point for the GUI application."""
    # Set up multiprocessing for Windows
    import multiprocessing
    multiprocessing.freeze_support()
    
    # Set asyncio policy for Windows
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Create and run the application
    root = tk.Tk()
    app = GeminiChatGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
