# src/gui_chat/server_manager.py
"""
Server Manager for GUI Chat Application
Handles server lifecycle, process management, and health monitoring.
"""

import asyncio
import threading
import multiprocessing
import time
import sys
import os
import signal
import logging
from typing import Optional, Callable, Dict, Any
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class ServerState(Enum):
    """Server lifecycle states."""
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "localhost"
    port: int = 6969
    reload: bool = False
    log_level: str = "info"


class ServerManager:
    """
    Manages the WebAI-to-API server lifecycle.
    Supports starting, stopping, and monitoring the server.
    """
    
    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config or ServerConfig()
        self._state = ServerState.IDLE
        self._process: Optional[multiprocessing.Process] = None
        self._stop_event: Optional[multiprocessing.Event] = None
        self._callbacks: Dict[str, list] = {
            "state_changed": [],
            "log": [],
            "error": []
        }
        self._health_check_thread: Optional[threading.Thread] = None
        self._running = False
    
    @property
    def state(self) -> ServerState:
        """Get current server state."""
        return self._state
    
    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._state == ServerState.RUNNING
    
    @property
    def url(self) -> str:
        """Get server URL."""
        return f"http://{self.config.host}:{self.config.port}"
    
    def on_state_changed(self, callback: Callable[[ServerState], None]):
        """Register a callback for state changes."""
        self._callbacks["state_changed"].append(callback)
    
    def on_log(self, callback: Callable[[str], None]):
        """Register a callback for log messages."""
        self._callbacks["log"].append(callback)
    
    def on_error(self, callback: Callable[[str], None]):
        """Register a callback for errors."""
        self._callbacks["error"].append(callback)
    
    def _emit(self, event: str, data: Any):
        """Emit an event to all registered callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def _set_state(self, state: ServerState):
        """Set server state and emit event."""
        old_state = self._state
        self._state = state
        if old_state != state:
            self._emit("state_changed", state)
            logger.info(f"Server state changed: {old_state.value} -> {state.value}")
    
    def start(self) -> bool:
        """
        Start the server in a separate process.
        Returns True if start was initiated successfully.
        """
        if self._state in (ServerState.RUNNING, ServerState.STARTING):
            self._emit("log", "Server is already running or starting")
            return False
        
        self._set_state(ServerState.STARTING)
        self._stop_event = multiprocessing.Event()
        
        # Start server process
        self._process = multiprocessing.Process(
            target=self._run_server_process,
            args=(
                self.config.host,
                self.config.port,
                self.config.reload,
                self._stop_event
            ),
            daemon=True
        )
        
        try:
            self._process.start()
            self._emit("log", f"Starting server at {self.url}")
            
            # Start health check thread
            self._running = True
            self._health_check_thread = threading.Thread(
                target=self._health_check_loop,
                daemon=True
            )
            self._health_check_thread.start()
            
            return True
            
        except Exception as e:
            self._set_state(ServerState.ERROR)
            self._emit("error", f"Failed to start server: {e}")
            return False
    
    def stop(self, timeout: float = 10.0) -> bool:
        """
        Stop the server gracefully.
        Returns True if server stopped successfully.
        """
        if self._state not in (ServerState.RUNNING, ServerState.STARTING):
            self._emit("log", "Server is not running")
            return False
        
        self._set_state(ServerState.STOPPING)
        self._running = False
        
        try:
            # Signal the server to stop
            if self._stop_event:
                self._stop_event.set()
            
            # Wait for process to exit
            if self._process and self._process.is_alive():
                self._process.join(timeout=timeout)
                
                # Force terminate if still running
                if self._process.is_alive():
                    self._emit("log", "Force terminating server process")
                    self._process.terminate()
                    self._process.join(timeout=5.0)
            
            self._set_state(ServerState.IDLE)
            self._emit("log", "Server stopped")
            return True
            
        except Exception as e:
            self._set_state(ServerState.ERROR)
            self._emit("error", f"Error stopping server: {e}")
            return False
    
    def restart(self) -> bool:
        """Restart the server."""
        self.stop()
        time.sleep(1)
        return self.start()
    
    def _health_check_loop(self):
        """Background thread to check server health."""
        import httpx
        
        # Wait for server to start
        time.sleep(2)
        
        consecutive_failures = 0
        max_failures = 3
        
        while self._running:
            try:
                # Quick health check
                with httpx.Client(timeout=5.0) as client:
                    response = client.get(f"{self.url}/docs")
                    if response.status_code == 200:
                        if self._state == ServerState.STARTING:
                            self._set_state(ServerState.RUNNING)
                            self._emit("log", f"Server is ready at {self.url}")
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
            except Exception:
                consecutive_failures += 1
            
            # Check if process is still alive
            if self._process and not self._process.is_alive():
                if self._state == ServerState.RUNNING:
                    self._set_state(ServerState.ERROR)
                    self._emit("error", "Server process terminated unexpectedly")
                break
            
            # Check for consecutive failures
            if consecutive_failures >= max_failures and self._state == ServerState.STARTING:
                # Still trying to start but failing
                pass
            
            time.sleep(5)
    
    @staticmethod
    def _run_server_process(host: str, port: int, reload: bool, stop_event: multiprocessing.Event):
        """
        Server process entry point.
        Runs the FastAPI server using uvicorn.
        """
        # Ignore SIGINT in child process
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        
        # Set event loop policy for Windows
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        try:
            import uvicorn
            from app.main import app as webai_app
            
            config = uvicorn.Config(
                webai_app,
                host=host,
                port=port,
                reload=reload,
                log_config=None
            )
            server = uvicorn.Server(config)
            
            # Shutdown monitor
            def shutdown_monitor():
                stop_event.wait()
                server.should_exit = True
            
            monitor_thread = threading.Thread(target=shutdown_monitor, daemon=True)
            monitor_thread.start()
            
            # Run server
            server.run()
            
        except Exception as e:
            logger.error(f"Server process error: {e}")
        finally:
            logger.info("Server process exited")


class AsyncServerManager:
    """
    Async version of ServerManager for use with asyncio.
    """
    
    def __init__(self, config: Optional[ServerConfig] = None):
        self._sync_manager = ServerManager(config)
    
    @property
    def state(self) -> ServerState:
        return self._sync_manager.state
    
    @property
    def is_running(self) -> bool:
        return self._sync_manager.is_running
    
    @property
    def url(self) -> str:
        return self._sync_manager.url
    
    def on_state_changed(self, callback: Callable[[ServerState], None]):
        self._sync_manager.on_state_changed(callback)
    
    def on_log(self, callback: Callable[[str], None]):
        self._sync_manager.on_log(callback)
    
    def on_error(self, callback: Callable[[str], None]):
        self._sync_manager.on_error(callback)
    
    async def start(self) -> bool:
        """Start the server asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_manager.start)
    
    async def stop(self, timeout: float = 10.0) -> bool:
        """Stop the server asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_manager.stop, timeout)
    
    async def restart(self) -> bool:
        """Restart the server asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_manager.restart)
