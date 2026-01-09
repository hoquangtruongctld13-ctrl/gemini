# src/app/services/session_manager.py
import asyncio
from app.logger import logger
from app.services.gemini_client import get_gemini_client, GeminiClientNotInitializedError

class SessionManager:
    def __init__(self, client):
        self.client = client
        self.session = None
        self.model = None
        self._lock = None  # Lazy initialization to avoid event loop issues

    def _get_lock(self):
        """Get or create lock for current event loop to avoid 'Event loop is closed' errors."""
        try:
            loop = asyncio.get_running_loop()
            # Check if we have a lock and if it's for the current event loop
            if self._lock is None:
                self._lock = asyncio.Lock()
            return self._lock
        except RuntimeError:
            # No running loop, create a new lock
            self._lock = asyncio.Lock()
            return self._lock

    def reset_session(self):
        """Reset the session to force a new chat session on next request."""
        self.session = None
        self.model = None
        logger.info("Session reset - will create new chat session on next request")

    async def get_response(self, model, message, images):
        lock = self._get_lock()
        async with lock:
            # Start a new session if none exists or the model has changed
            if self.session is None or self.model != model:
                if self.session is not None:
                    # Closing the session is handled by the library's internal logic
                    pass
                # If model is an Enum, use its value
                model_value = model.value if hasattr(model, "value") else model
                self.session = self.client.start_chat(model=model_value)
                self.model = model

            try:
                # Send message using the gemini-webapi's ChatSession
                return await self.session.send_message(prompt=message, images=images)
            except Exception as e:
                error_str = str(e).lower()
                # Reset session on critical errors to allow recovery
                if any(err in error_str for err in ['event loop', 'closed', 'connection', 'auth']):
                    logger.warning(f"Critical error detected, resetting session: {e}")
                    self.reset_session()
                logger.error(f"Error in session get_response: {e}", exc_info=True)
                raise

_translate_session_manager = None
_gemini_chat_manager = None

def init_session_managers():
    """
    Initialize session managers for translation and chat
    """
    global _translate_session_manager, _gemini_chat_manager
    try:
        client = get_gemini_client()
        _translate_session_manager = SessionManager(client)
        _gemini_chat_manager = SessionManager(client)
        logger.info("Session managers initialized successfully")
    except GeminiClientNotInitializedError:
        logger.warning("Session managers not initialized: Gemini client not available.")

def reset_all_sessions():
    """Reset all session managers to recover from errors."""
    global _translate_session_manager, _gemini_chat_manager
    if _translate_session_manager:
        _translate_session_manager.reset_session()
    if _gemini_chat_manager:
        _gemini_chat_manager.reset_session()
    logger.info("All sessions reset")

async def reinit_session_managers():
    """
    Reinitialize session managers with refreshed client.
    Use this when client has been reinitialized.
    """
    global _translate_session_manager, _gemini_chat_manager
    try:
        client = get_gemini_client()
        _translate_session_manager = SessionManager(client)
        _gemini_chat_manager = SessionManager(client)
        logger.info("Session managers reinitialized successfully")
        return True
    except GeminiClientNotInitializedError:
        logger.warning("Session managers not reinitialized: Gemini client not available.")
        return False

def get_translate_session_manager():
    return _translate_session_manager

def get_gemini_chat_manager():
    return _gemini_chat_manager
