# src/gui_chat/gemini_proxy_client.py
"""
Gemini Proxy Client with API Key Rotation
Provides a client interface to interact with Gemini API through rotating API keys.
"""

import asyncio
import httpx
from typing import Optional, List, Dict, Any, AsyncGenerator
from dataclasses import dataclass
import json
import logging

from gui_chat.key_rotator import KeyRotator, AccountRotator

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Represents a chat message."""
    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class ChatResponse:
    """Represents a chat response from the API."""
    content: str
    model: str
    finish_reason: str
    usage: Dict[str, int]
    raw_response: Optional[Dict[str, Any]] = None


class GeminiProxyClient:
    """
    Client for interacting with Gemini API through the local server.
    Supports API key rotation for direct API access as well.
    """
    
    # Gemini API base URL for direct access
    GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
    
    def __init__(
        self,
        server_url: str = "http://localhost:6969",
        key_rotator: Optional[KeyRotator] = None,
        timeout: float = 60.0
    ):
        self.server_url = server_url.rstrip("/")
        self.key_rotator = key_rotator or KeyRotator()
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def chat(
        self,
        messages: List[ChatMessage],
        model: str = "gemini-2.5-flash",
        stream: bool = False,
        use_server: bool = True
    ) -> ChatResponse:
        """
        Send a chat request.
        
        Args:
            messages: List of chat messages
            model: Model to use
            stream: Whether to stream the response
            use_server: If True, use local server; otherwise, use direct API with key rotation
        
        Returns:
            ChatResponse with the assistant's reply
        """
        if use_server:
            return await self._chat_via_server(messages, model, stream)
        else:
            return await self._chat_via_api(messages, model, stream)
    
    async def _chat_via_server(
        self,
        messages: List[ChatMessage],
        model: str,
        stream: bool
    ) -> ChatResponse:
        """Send chat request through local server."""
        client = await self._get_client()
        
        url = f"{self.server_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream
        }
        
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            
            return ChatResponse(
                content=data.get("choices", [{}])[0].get("message", {}).get("content", ""),
                model=data.get("model", model),
                finish_reason=data.get("choices", [{}])[0].get("finish_reason", "stop"),
                usage=data.get("usage", {}),
                raw_response=data
            )
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from server: {e}")
            raise
        except Exception as e:
            logger.error(f"Error communicating with server: {e}")
            raise
    
    async def _chat_via_api(
        self,
        messages: List[ChatMessage],
        model: str,
        stream: bool
    ) -> ChatResponse:
        """Send chat request directly to Gemini API with key rotation."""
        client = await self._get_client()
        
        # Get an API key from the rotator
        api_key = await self.key_rotator.get_next_key()
        if not api_key:
            raise RuntimeError("No API keys available")
        
        # Build the request URL
        url = f"{self.GEMINI_API_BASE}/models/{model}:generateContent"
        
        # Build the request payload
        contents = []
        for msg in messages:
            role = "user" if msg.role == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg.content}]
            })
        
        payload = {
            "contents": contents
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        try:
            response = await client.post(
                url,
                json=payload,
                headers=headers,
                params={"key": api_key}
            )
            
            # Handle rate limiting
            if response.status_code == 429:
                await self.key_rotator.report_rate_limit(api_key, cooldown_seconds=60)
                raise RuntimeError("Rate limited. Try again later.")
            
            # Handle auth errors
            if response.status_code in (401, 403):
                await self.key_rotator.report_error(api_key, is_auth_error=True)
                raise RuntimeError("Authentication error with API key")
            
            response.raise_for_status()
            
            # Report success
            await self.key_rotator.report_success(api_key)
            
            data = response.json()
            
            # Extract response text
            content = ""
            if "candidates" in data and data["candidates"]:
                candidate = data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    content = candidate["content"]["parts"][0].get("text", "")
            
            return ChatResponse(
                content=content,
                model=model,
                finish_reason=data.get("candidates", [{}])[0].get("finishReason", "STOP"),
                usage={
                    "prompt_tokens": data.get("usageMetadata", {}).get("promptTokenCount", 0),
                    "completion_tokens": data.get("usageMetadata", {}).get("candidatesTokenCount", 0),
                    "total_tokens": data.get("usageMetadata", {}).get("totalTokenCount", 0)
                },
                raw_response=data
            )
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from Gemini API: {e}")
            await self.key_rotator.report_error(api_key)
            raise
        except Exception as e:
            logger.error(f"Error communicating with Gemini API: {e}")
            await self.key_rotator.report_error(api_key)
            raise
    
    async def simple_chat(
        self,
        message: str,
        model: str = "gemini-2.5-flash",
        use_server: bool = True
    ) -> str:
        """
        Simple chat interface for single-turn conversation.
        
        Args:
            message: User message
            model: Model to use
            use_server: If True, use local server; otherwise, use direct API
        
        Returns:
            Assistant's response text
        """
        messages = [ChatMessage(role="user", content=message)]
        response = await self.chat(messages, model, use_server=use_server)
        return response.content
    
    async def generate_content(
        self,
        prompt: str,
        model: str = "gemini-2.5-flash",
        use_server: bool = True
    ) -> str:
        """
        Generate content using Gemini.
        
        Args:
            prompt: The prompt to generate content from
            model: Model to use
            use_server: If True, use local server
        
        Returns:
            Generated content
        """
        if use_server:
            client = await self._get_client()
            url = f"{self.server_url}/gemini"
            payload = {
                "message": prompt,
                "model": model
            }
            
            response = await client.post(url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            return data.get("response", "")
        else:
            return await self.simple_chat(prompt, model, use_server=False)
    
    async def health_check(self) -> bool:
        """Check if the server is healthy."""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.server_url}/docs")
            return response.status_code == 200
        except Exception:
            return False


class SyncGeminiProxyClient:
    """
    Synchronous wrapper for GeminiProxyClient.
    Useful for integration with synchronous code (e.g., Tkinter callbacks).
    """
    
    def __init__(
        self,
        server_url: str = "http://localhost:6969",
        key_rotator: Optional[KeyRotator] = None,
        timeout: float = 60.0
    ):
        self.server_url = server_url
        self.key_rotator = key_rotator or KeyRotator()
        self.timeout = timeout
    
    def _run_async(self, coro):
        """Run an async coroutine synchronously."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If called from within an async context, use run_coroutine_threadsafe
                import concurrent.futures
                future = asyncio.run_coroutine_threadsafe(coro, loop)
                return future.result(timeout=self.timeout)
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            # No event loop exists, create a new one
            return asyncio.run(coro)
    
    def chat(
        self,
        messages: List[ChatMessage],
        model: str = "gemini-2.5-flash",
        stream: bool = False,
        use_server: bool = True
    ) -> ChatResponse:
        """Send a chat request synchronously."""
        async def _chat():
            async with GeminiProxyClient(
                self.server_url,
                self.key_rotator,
                self.timeout
            ) as client:
                return await client.chat(messages, model, stream, use_server)
        
        return self._run_async(_chat())
    
    def simple_chat(
        self,
        message: str,
        model: str = "gemini-2.5-flash",
        use_server: bool = True
    ) -> str:
        """Simple chat interface synchronously."""
        async def _chat():
            async with GeminiProxyClient(
                self.server_url,
                self.key_rotator,
                self.timeout
            ) as client:
                return await client.simple_chat(message, model, use_server)
        
        return self._run_async(_chat())
    
    def generate_content(
        self,
        prompt: str,
        model: str = "gemini-2.5-flash",
        use_server: bool = True
    ) -> str:
        """Generate content synchronously."""
        async def _generate():
            async with GeminiProxyClient(
                self.server_url,
                self.key_rotator,
                self.timeout
            ) as client:
                return await client.generate_content(prompt, model, use_server)
        
        return self._run_async(_generate())
    
    def health_check(self) -> bool:
        """Check server health synchronously."""
        async def _check():
            async with GeminiProxyClient(
                self.server_url,
                self.key_rotator,
                self.timeout
            ) as client:
                return await client.health_check()
        
        return self._run_async(_check())
