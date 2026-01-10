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
import signal
from dataclasses import dataclass
import re
from typing import Optional, TYPE_CHECKING
import httpx
import uvicorn

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event as MultiprocessingEvent

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run_server_process(host: str, port: int, stop_event: "MultiprocessingEvent"):
    """
    Server process function.
    
    This function is defined at module level to be picklable for Windows multiprocessing.
    On Windows, multiprocessing uses 'spawn' which requires all target functions 
    to be importable from the module namespace (not local/nested functions).
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Import app here to avoid circular imports and ensure fresh import in subprocess
    from app.main import app as webai_app
    
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
            from app.services.gemini_client import init_gemini_client
            
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
            
            # Use module-level function for Windows multiprocessing compatibility
            # Local functions cannot be pickled when using 'spawn' start method
            self.process = multiprocessing.Process(
                target=_run_server_process,
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
        self.http_client = httpx.Client(timeout=120.0)
        
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
        
        # Tabs
        tabs = ttk.Notebook(main_frame)
        tabs.pack(fill=tk.BOTH, expand=True)
        
        chat_tab = ttk.Frame(tabs)
        translate_tab = ttk.Frame(tabs)
        
        tabs.add(chat_tab, text="Chat")
        tabs.add(translate_tab, text="Subtitle Translate")
        
        # Chat tab
        self._build_chat_tab(chat_tab)
        
        # Translate tab
        self._build_translate_tab(translate_tab)

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

    def _build_chat_tab(self, parent):
        """Build chat tab UI."""
        self._build_model_selection(parent)
        self._build_chat_area(parent)
        self._build_input_area(parent)
        
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

    @dataclass
    class SubtitleEntry:
        index: int
        timecode: Optional[str]
        content: str

    def _build_translate_tab(self, parent):
        """Build subtitle translate tab UI with table view."""
        # Settings frame
        settings_frame = ttk.Frame(parent)
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        model_label = ttk.Label(settings_frame, text="Model:")
        model_label.grid(row=0, column=0, padx=(0, 8), pady=4, sticky=tk.W)
        
        self.translate_model_var = tk.StringVar(value=self.MODELS[0][1])
        self.translate_model_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.translate_model_var,
            values=[m[0] for m in self.MODELS],
            state="readonly",
            width=20
        )
        self.translate_model_combo.current(0)
        self.translate_model_combo.grid(row=0, column=1, padx=(0, 16), pady=4, sticky=tk.W)
        
        target_label = ttk.Label(settings_frame, text="Target language:")
        target_label.grid(row=0, column=2, padx=(0, 8), pady=4, sticky=tk.W)
        
        self.target_language_var = tk.StringVar(value="Vietnamese")
        target_entry = ttk.Entry(settings_frame, textvariable=self.target_language_var, width=18)
        target_entry.grid(row=0, column=3, padx=(0, 16), pady=4, sticky=tk.W)
        
        source_label = ttk.Label(settings_frame, text="Source language:")
        source_label.grid(row=0, column=4, padx=(0, 8), pady=4, sticky=tk.W)
        
        self.source_language_var = tk.StringVar(value="")
        source_entry = ttk.Entry(settings_frame, textvariable=self.source_language_var, width=18)
        source_entry.grid(row=0, column=5, padx=(0, 16), pady=4, sticky=tk.W)
        
        lines_label = ttk.Label(settings_frame, text="Lines/request:")
        lines_label.grid(row=1, column=0, padx=(0, 8), pady=4, sticky=tk.W)
        
        self.lines_per_request_var = tk.StringVar(value="10")
        lines_entry = ttk.Entry(settings_frame, textvariable=self.lines_per_request_var, width=10)
        lines_entry.grid(row=1, column=1, padx=(0, 16), pady=4, sticky=tk.W)
        
        parallel_label = ttk.Label(settings_frame, text="Parallel requests:")
        parallel_label.grid(row=1, column=2, padx=(0, 8), pady=4, sticky=tk.W)
        
        self.parallel_requests_var = tk.StringVar(value="3")
        parallel_entry = ttk.Entry(settings_frame, textvariable=self.parallel_requests_var, width=10)
        parallel_entry.grid(row=1, column=3, padx=(0, 16), pady=4, sticky=tk.W)
        
        # Instruction frame
        instruction_frame = ttk.Frame(parent)
        instruction_frame.pack(fill=tk.X, pady=(0, 10))
        
        system_label = ttk.Label(instruction_frame, text="System instruction:")
        system_label.pack(anchor=tk.W)
        self.system_instruction_text = scrolledtext.ScrolledText(
            instruction_frame,
            height=3,
            font=('Consolas', 10),
            bg='#1e1e1e',
            fg='#d4d4d4',
            insertbackground='#ffffff',
            padx=8,
            pady=6
        )
        self.system_instruction_text.pack(fill=tk.X, expand=False, pady=(0, 8))
        
        prompt_label = ttk.Label(instruction_frame, text="Prompt template (use {lines}):")
        prompt_label.pack(anchor=tk.W)
        self.prompt_template_text = scrolledtext.ScrolledText(
            instruction_frame,
            height=3,
            font=('Consolas', 10),
            bg='#1e1e1e',
            fg='#d4d4d4',
            insertbackground='#ffffff',
            padx=8,
            pady=6
        )
        self.prompt_template_text.pack(fill=tk.X, expand=False)
        
        # File load and buttons frame
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=(0, 10))
        
        load_btn = ttk.Button(
            btn_frame,
            text="📂 Load SRT File",
            command=self._load_srt_file
        )
        load_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        paste_btn = ttk.Button(
            btn_frame,
            text="📋 Paste Text",
            command=self._paste_subtitle_text
        )
        paste_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        clear_btn = ttk.Button(
            btn_frame,
            text="🗑️ Clear",
            command=self._clear_subtitle_table
        )
        clear_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.translate_btn = ttk.Button(
            btn_frame,
            text="🔄 Translate Subtitles",
            command=self._send_multi_translate
        )
        self.translate_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        export_btn = ttk.Button(
            btn_frame,
            text="💾 Export SRT",
            command=self._export_translated_srt
        )
        export_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Status label
        self.translate_status_label = ttk.Label(btn_frame, text="", style='Status.TLabel')
        self.translate_status_label.pack(side=tk.RIGHT)
        
        # Table frame with Treeview
        table_frame = ttk.Frame(parent)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create Treeview with columns
        columns = ("index", "timecode", "original", "translated")
        self.subtitle_tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            selectmode="extended"
        )
        
        # Define column headings
        self.subtitle_tree.heading("index", text="#")
        self.subtitle_tree.heading("timecode", text="Timecode")
        self.subtitle_tree.heading("original", text="Original")
        self.subtitle_tree.heading("translated", text="Translated")
        
        # Define column widths
        self.subtitle_tree.column("index", width=50, minwidth=40, stretch=False)
        self.subtitle_tree.column("timecode", width=180, minwidth=150, stretch=False)
        self.subtitle_tree.column("original", width=300, minwidth=200)
        self.subtitle_tree.column("translated", width=300, minwidth=200)
        
        # Add scrollbars
        tree_scroll_y = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.subtitle_tree.yview)
        tree_scroll_x = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.subtitle_tree.xview)
        self.subtitle_tree.configure(yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)
        
        # Pack the tree and scrollbars
        self.subtitle_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Configure tree tags for styling
        self.subtitle_tree.tag_configure('missing', foreground='#f14c4c')
        self.subtitle_tree.tag_configure('translated', foreground='#4ec9b0')
        
        # Store subtitle entries for later use
        self._subtitle_entries: list["GeminiChatGUI.SubtitleEntry"] = []
        
        # Legacy text widgets for backward compatibility (hidden)
        self.subtitle_input = None
        self.subtitle_output = None
        
    def _load_srt_file(self):
        """Load an SRT file and populate the table."""
        from tkinter import filedialog
        
        filepath = filedialog.askopenfilename(
            title="Select SRT File",
            filetypes=[("SRT files", "*.srt"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
        
        try:
            # Try different encodings
            content = None
            for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                try:
                    with open(filepath, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                messagebox.showerror("Error", "Could not read file with supported encodings.")
                return
            
            entries, output_format = self._parse_subtitle_input(content)
            if not entries:
                messagebox.showwarning("No Subtitles", "No subtitle lines could be parsed from the file.")
                return
            
            self._populate_subtitle_table(entries)
            self.translate_status_label.config(text=f"Loaded {len(entries)} lines from {os.path.basename(filepath)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def _clear_subtitle_table(self):
        """Clear the subtitle table."""
        for item in self.subtitle_tree.get_children():
            self.subtitle_tree.delete(item)
        self._subtitle_entries = []
        self.translate_status_label.config(text="Cleared")
    
    def _paste_subtitle_text(self):
        """Open a dialog to paste subtitle text."""
        paste_window = tk.Toplevel(self.root)
        paste_window.title("Paste Subtitle Text")
        paste_window.geometry("600x400")
        paste_window.transient(self.root)
        paste_window.grab_set()
        
        label = ttk.Label(paste_window, text="Paste subtitle text (SRT format or index: content):")
        label.pack(anchor=tk.W, padx=10, pady=(10, 5))
        
        text_area = scrolledtext.ScrolledText(
            paste_window,
            wrap=tk.WORD,
            font=('Consolas', 10),
            bg='#1e1e1e',
            fg='#d4d4d4',
            insertbackground='#ffffff',
            padx=8,
            pady=6
        )
        text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        def on_ok():
            content = text_area.get("1.0", tk.END).strip()
            if not content:
                messagebox.showwarning("No Input", "Please paste subtitle text.", parent=paste_window)
                return
            
            entries, output_format = self._parse_subtitle_input(content)
            if not entries:
                messagebox.showwarning("Invalid Input", "No subtitle lines could be parsed.", parent=paste_window)
                return
            
            self._populate_subtitle_table(entries)
            self.translate_status_label.config(text=f"Loaded {len(entries)} lines from pasted text")
            paste_window.destroy()
        
        def on_cancel():
            paste_window.destroy()
        
        btn_frame = ttk.Frame(paste_window)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ok_btn = ttk.Button(btn_frame, text="OK", command=on_ok)
        ok_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        cancel_btn = ttk.Button(btn_frame, text="Cancel", command=on_cancel)
        cancel_btn.pack(side=tk.RIGHT)
    
    def _populate_subtitle_table(self, entries: list["GeminiChatGUI.SubtitleEntry"]):
        """Populate the table with subtitle entries."""
        # Clear existing items
        for item in self.subtitle_tree.get_children():
            self.subtitle_tree.delete(item)
        
        self._subtitle_entries = entries
        
        for entry in entries:
            # Replace newlines in content with visible marker for display
            display_original = entry.content.replace('\n', ' ↵ ')
            timecode = entry.timecode if entry.timecode else ""
            
            self.subtitle_tree.insert(
                "",
                tk.END,
                values=(entry.index, timecode, display_original, "")
            )
    
    def _update_table_translation(self, index: int, translated: str, is_missing: bool = False):
        """Update a single row's translation in the table."""
        for item in self.subtitle_tree.get_children():
            values = self.subtitle_tree.item(item, 'values')
            if values and int(values[0]) == index:
                display_translated = translated.replace('\n', ' ↵ ')
                tag = 'missing' if is_missing else 'translated'
                self.subtitle_tree.item(item, values=(values[0], values[1], values[2], display_translated), tags=(tag,))
                break
    
    def _export_translated_srt(self):
        """Export translated subtitles to an SRT file."""
        from tkinter import filedialog
        
        if not self._subtitle_entries:
            messagebox.showwarning("No Data", "No subtitles to export.")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Save Translated SRT",
            defaultextension=".srt",
            filetypes=[("SRT files", "*.srt"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
        
        try:
            # Collect translations from the table
            translation_map = {}
            for item in self.subtitle_tree.get_children():
                values = self.subtitle_tree.item(item, 'values')
                if values and len(values) >= 4:
                    idx = int(values[0])
                    translated = values[3].replace(' ↵ ', '\n') if values[3] else ""
                    translation_map[idx] = translated
            
            # Build SRT content
            blocks = []
            for entry in self._subtitle_entries:
                translated = translation_map.get(entry.index, entry.content)
                # If no translation, use original
                if not translated or translated == "[API không trả về dòng này]":
                    translated = entry.content
                
                block_lines = [str(entry.index)]
                if entry.timecode:
                    block_lines.append(entry.timecode)
                block_lines.append(translated)
                blocks.append("\n".join(block_lines))
            
            content = "\n\n".join(blocks)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.translate_status_label.config(text=f"Exported to {os.path.basename(filepath)}")
            messagebox.showinfo("Success", f"Exported {len(self._subtitle_entries)} lines to {filepath}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {str(e)}")
    
    def _handle_translate_table_update(self, data: dict):
        """Handle translation results and update the table."""
        entries = data.get("entries", [])
        translations = data.get("translations", [])
        total_lines = data.get("total_lines", 0)
        successful_lines = data.get("successful_lines", 0)
        failed_lines = data.get("failed_lines", 0)
        errors = data.get("errors", [])
        
        # Build translation map
        translation_map = {item.get("index"): item.get("translated") for item in translations}
        
        # Update each row in the table
        translated_indices = set()
        for item in self.subtitle_tree.get_children():
            values = self.subtitle_tree.item(item, 'values')
            if values and len(values) >= 4:
                idx = int(values[0])
                
                if idx in translation_map and translation_map[idx]:
                    translated_text = translation_map[idx].replace('\n', ' ↵ ')
                    self.subtitle_tree.item(
                        item, 
                        values=(values[0], values[1], values[2], translated_text),
                        tags=('translated',)
                    )
                    translated_indices.add(idx)
                else:
                    # Mark as missing translation
                    self.subtitle_tree.item(
                        item, 
                        values=(values[0], values[1], values[2], "[API không trả về dòng này]"),
                        tags=('missing',)
                    )
        
        # Re-enable translate button
        self.translate_btn.config(state=tk.NORMAL)
        
        # Update status
        status_text = f"Done: {successful_lines}/{total_lines} lines translated"
        if failed_lines > 0:
            status_text += f", {failed_lines} failed"
        self.translate_status_label.config(text=status_text)
        
        # Show errors in chat if any
        if errors:
            error_text = f"Translation completed with {len(errors)} error(s):\n" + "\n".join(errors[:5])
            if len(errors) > 5:
                error_text += f"\n... and {len(errors) - 5} more errors"
            self._append_message("System", error_text, 'system')
        
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
            self._send_message_sync(message)
        
        thread = threading.Thread(target=send_thread, daemon=True)
        thread.start()
        
    def _send_message_sync(self, message: str):
        """Send message to API using sync client."""
        try:
            # Get selected model
            selected_index = self.model_combo.current()
            model = self.MODELS[selected_index][1]
            
            # Send request
            response = self.http_client.post(
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

    def _send_multi_translate(self):
        """Send multi-translate request."""
        if not self.server_manager.is_running:
            messagebox.showwarning("Server Not Running", "Please start the server first.")
            return
        
        # Check if we have entries in the table
        if not self._subtitle_entries:
            messagebox.showwarning("No Input", "Please load an SRT file or paste subtitle text first.")
            return
        
        entries = self._subtitle_entries
        
        # Disable translate button during processing
        self.translate_btn.config(state=tk.DISABLED)
        self.translate_status_label.config(text="Translating...")
        
        def send_thread():
            self._send_multi_translate_sync(entries)
        
        thread = threading.Thread(target=send_thread, daemon=True)
        thread.start()

    def _send_multi_translate_sync(self, entries: list["GeminiChatGUI.SubtitleEntry"]):
        """Send multi-translate request using sync client."""
        try:
            selected_index = self.translate_model_combo.current()
            model = self.MODELS[selected_index][1]
            
            lines_per_request = self._safe_int(self.lines_per_request_var, 10)
            parallel_requests = self._safe_int(self.parallel_requests_var, 3)
            
            payload = {
                "lines": [{"index": entry.index, "content": entry.content} for entry in entries],
                "target_language": self.target_language_var.get().strip() or "Vietnamese",
                "model": model,
                "lines_per_request": max(1, min(lines_per_request, 50)),
                "parallel_requests": max(1, min(parallel_requests, 10)),
            }
            
            source_language = self.source_language_var.get().strip()
            if source_language:
                payload["source_language"] = source_language
            
            system_instruction = self.system_instruction_text.get("1.0", tk.END).strip()
            if system_instruction:
                payload["system_instruction"] = system_instruction
            
            prompt_template = self.prompt_template_text.get("1.0", tk.END).strip()
            if prompt_template:
                if "{lines}" not in prompt_template:
                    prompt_template = f"{prompt_template}\n\n{{lines}}"
                payload["prompt_template"] = prompt_template
            
            response = self.http_client.post(
                f"{self.server_manager.get_base_url()}/multi-translate",
                json=payload,
                timeout=300.0  # 5 minute timeout for large batches
            )
            
            if response.status_code == 200:
                data = response.json()
                translations = data.get("translations", [])
                errors = data.get("errors") or []
                
                # Build response data for the queue
                result = {
                    "entries": entries,
                    "translations": translations,
                    "total_lines": data.get("total_lines", len(entries)),
                    "successful_lines": data.get("successful_lines", 0),
                    "failed_lines": data.get("failed_lines", 0),
                    "errors": errors
                }
                self.message_queue.put(('translate_table_update', result))
            else:
                error_detail = response.json().get('detail', response.text)
                self.message_queue.put(('translate_error', f"Translate error {response.status_code}: {error_detail}"))
                
        except Exception as e:
            self.message_queue.put(('translate_error', f"Translate connection error: {str(e)}"))

    def _safe_int(self, variable: tk.Variable, default: int) -> int:
        """Safely parse int from Tk variable."""
        try:
            return int(variable.get())
        except (tk.TclError, ValueError):
            return default

    def _parse_subtitle_input(self, raw_text: str) -> tuple[list["GeminiChatGUI.SubtitleEntry"], str]:
        """Parse subtitle input text into entries."""
        lines = raw_text.splitlines()
        entries: list[GeminiChatGUI.SubtitleEntry] = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            
            if line.isdigit() and i + 1 < len(lines) and "-->" in lines[i + 1]:
                index = int(line)
                timecode = lines[i + 1].strip()
                i += 2
                content_lines = []
                while i < len(lines) and lines[i].strip():
                    content_lines.append(lines[i].rstrip())
                    i += 1
                content = "\n".join(content_lines).strip()
                if content:
                    entries.append(self.SubtitleEntry(index=index, timecode=timecode, content=content))
                continue
            
            match = re.match(r"^(\d+)[:\.\)]\s*(.+)$", line)
            if match:
                entries.append(self.SubtitleEntry(
                    index=int(match.group(1)),
                    timecode=None,
                    content=match.group(2).strip()
                ))
                i += 1
                continue
            
            entries.append(self.SubtitleEntry(
                index=len(entries) + 1,
                timecode=None,
                content=line
            ))
            i += 1
        
        output_format = "srt" if any(entry.timecode for entry in entries) else "simple"
        return entries, output_format

    def _format_translations(
        self,
        entries: list["GeminiChatGUI.SubtitleEntry"],
        translations: list[dict],
        output_format: str
    ) -> str:
        """Format translations for output."""
        translation_map = {item.get("index"): item.get("translated") for item in translations}
        
        if output_format == "srt":
            blocks = []
            for entry in entries:
                translated = translation_map.get(entry.index, entry.content)
                block_lines = [str(entry.index)]
                if entry.timecode:
                    block_lines.append(entry.timecode)
                block_lines.append(translated)
                blocks.append("\n".join(block_lines))
            return "\n\n".join(blocks)
        
        lines = []
        for entry in entries:
            translated = translation_map.get(entry.index, entry.content)
            lines.append(f"{entry.index}: {translated}")
        return "\n".join(lines)
            
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
                            "2. Press F12 > Application (Chrome/Edge) or Storage (Firefox) > Cookies\n"
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
                    
                elif msg_type == 'translate_table_update':
                    # Update the table with translation results
                    self._handle_translate_table_update(data)
                    
                elif msg_type == 'translate_error':
                    self.translate_btn.config(state=tk.NORMAL)
                    self.translate_status_label.config(text="Translation failed")
                    self._append_message("Error", data, 'error')
                    
                elif msg_type == 'translate_response':
                    # Legacy support (not used with new table UI)
                    if self.subtitle_output:
                        self.subtitle_output.config(state=tk.NORMAL)
                        self.subtitle_output.delete("1.0", tk.END)
                        self.subtitle_output.insert(tk.END, data)
                        self.subtitle_output.config(state=tk.DISABLED)
                    
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
        
        try:
            self.http_client.close()
        except Exception:
            pass
        
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
