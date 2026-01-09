# src/app/services/gemini_client.py
from models.gemini import MyGeminiClient
from app.config import CONFIG, load_config
from app.logger import logger
from app.utils.browser import get_cookie_from_browser

# Import the specific exception to handle it gracefully
from gemini_webapi.exceptions import AuthError


class GeminiClientNotInitializedError(Exception):
    """Raised when the Gemini client is not initialized or initialization failed."""
    pass


# Global variable to store the Gemini client instance
_gemini_client = None
_initialization_error = None

async def init_gemini_client(force_reload_config: bool = False) -> bool:
    """
    Initialize and set up the Gemini client based on the configuration.
    Returns True on success, False on failure.
    
    Parameters:
        force_reload_config: If True, reload config file to pick up new cookies
    """
    global _gemini_client, _initialization_error, CONFIG
    _initialization_error = None

    # Reload config if requested (useful when cookies have been updated)
    if force_reload_config:
        from app import config as config_module
        config_module.CONFIG = load_config()
        CONFIG = config_module.CONFIG
        logger.info("Configuration reloaded")

    if CONFIG.getboolean("EnabledAI", "gemini", fallback=True):
        try:
            gemini_cookie_1PSID = CONFIG["Cookies"].get("gemini_cookie_1PSID")
            gemini_cookie_1PSIDTS = CONFIG["Cookies"].get("gemini_cookie_1PSIDTS")
            gemini_proxy = CONFIG["Proxy"].get("http_proxy")

            if not gemini_cookie_1PSID or not gemini_cookie_1PSIDTS:
                cookies = get_cookie_from_browser("gemini")
                if cookies:
                    gemini_cookie_1PSID, gemini_cookie_1PSIDTS = cookies

            if gemini_proxy == "":
                gemini_proxy = None

            if gemini_cookie_1PSID and gemini_cookie_1PSIDTS:
                # Close existing client if any
                if _gemini_client is not None:
                    try:
                        await _gemini_client.close()
                    except Exception:
                        pass  # Ignore errors when closing old client
                
                _gemini_client = MyGeminiClient(secure_1psid=gemini_cookie_1PSID, secure_1psidts=gemini_cookie_1PSIDTS, proxy=gemini_proxy)
                await _gemini_client.init()
                logger.info("Gemini client initialized successfully.")
                return True
            else:
                error_msg = (
                    "Gemini cookies not found or empty. "
                    "To fix this:\n"
                    "  1. Open https://gemini.google.com in your browser and log in\n"
                    "  2. Open DevTools (F12) > Application > Cookies > gemini.google.com\n"
                    "  3. Copy values of '__Secure-1PSID' and '__Secure-1PSIDTS'\n"
                    "  4. Paste them in config.conf under [Cookies] section\n"
                    "  See TROUBLESHOOTING.md for detailed instructions."
                )
                logger.error(error_msg)
                _initialization_error = error_msg
                return False

        except AuthError as e:
            error_msg = f"Gemini authentication failed: {e}. This usually means cookies are expired or invalid."
            logger.error(error_msg)
            _gemini_client = None
            _initialization_error = error_msg
            return False

        except Exception as e:
            error_msg = f"Unexpected error initializing Gemini client: {e}"
            logger.error(error_msg, exc_info=True)
            _gemini_client = None
            _initialization_error = error_msg
            return False
    else:
        error_msg = "Gemini client is disabled in config."
        logger.info(error_msg)
        _initialization_error = error_msg
        return False


async def reinit_gemini_client() -> bool:
    """
    Reinitialize the Gemini client, reloading configuration.
    Useful when cookies have been updated or when recovering from errors.
    """
    logger.info("Reinitializing Gemini client...")
    return await init_gemini_client(force_reload_config=True)


def get_gemini_client():
    """
    Returns the initialized Gemini client instance.

    Raises:
        GeminiClientNotInitializedError: If the client is not initialized.
    """
    if _gemini_client is None:
        error_detail = _initialization_error or "Gemini client was not initialized. Check logs for details."
        raise GeminiClientNotInitializedError(error_detail)
    return _gemini_client

