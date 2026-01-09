# src/gui_chat/key_rotator.py
"""
API Key Rotation System for Gemini API
Supports multiple API keys with automatic rotation, backoff on rate limits,
and health monitoring.
"""

import asyncio
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class KeyStatus(Enum):
    """Status of an API key."""
    ACTIVE = "active"
    RATE_LIMITED = "rate_limited"
    EXHAUSTED = "exhausted"
    INVALID = "invalid"


@dataclass
class APIKey:
    """Represents a Gemini API key with usage tracking."""
    key: str
    status: KeyStatus = KeyStatus.ACTIVE
    requests_made: int = 0
    last_used: float = 0.0
    cooldown_until: float = 0.0
    error_count: int = 0
    name: str = ""  # Optional friendly name
    
    def is_available(self) -> bool:
        """Check if the key is available for use."""
        if self.status in (KeyStatus.INVALID, KeyStatus.EXHAUSTED):
            return False
        if self.status == KeyStatus.RATE_LIMITED:
            return time.time() > self.cooldown_until
        return True
    
    def mark_rate_limited(self, cooldown_seconds: int = 60):
        """Mark key as rate limited with cooldown period."""
        self.status = KeyStatus.RATE_LIMITED
        self.cooldown_until = time.time() + cooldown_seconds
        self.error_count += 1
        logger.warning(f"Key {self.name or self.key[:8]}... rate limited for {cooldown_seconds}s")
    
    def mark_used(self):
        """Mark key as used (update usage stats)."""
        self.requests_made += 1
        self.last_used = time.time()
        if self.status == KeyStatus.RATE_LIMITED and time.time() > self.cooldown_until:
            self.status = KeyStatus.ACTIVE
    
    def reset_status(self):
        """Reset key status to active."""
        if self.status != KeyStatus.INVALID:
            self.status = KeyStatus.ACTIVE
            self.error_count = 0


class KeyRotator:
    """
    Rotates through multiple Gemini API keys.
    Implements round-robin selection with automatic backoff for rate-limited keys.
    """
    
    def __init__(self, keys: Optional[List[str]] = None):
        self._keys: List[APIKey] = []
        self._current_index: int = 0
        self._lock = asyncio.Lock()
        
        if keys:
            self.add_keys(keys)
    
    def add_key(self, key: str, name: str = "") -> bool:
        """Add a single API key."""
        # Check if key already exists
        for existing_key in self._keys:
            if existing_key.key == key:
                logger.warning(f"Key {name or key[:8]}... already exists")
                return False
        
        api_key = APIKey(key=key, name=name or f"key_{len(self._keys) + 1}")
        self._keys.append(api_key)
        logger.info(f"Added API key: {api_key.name}")
        return True
    
    def add_keys(self, keys: List[str]) -> int:
        """Add multiple API keys. Returns number of keys added."""
        added = 0
        for i, key in enumerate(keys):
            if self.add_key(key, f"key_{len(self._keys) + 1}"):
                added += 1
        return added
    
    def load_keys_from_file(self, filepath: str) -> int:
        """Load API keys from a file (one per line). Returns number of keys loaded."""
        path = Path(filepath)
        if not path.exists():
            logger.error(f"API keys file not found: {filepath}")
            return 0
        
        keys_loaded = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                key = line.strip()
                if key and not key.startswith("#"):  # Skip empty lines and comments
                    if self.add_key(key):
                        keys_loaded += 1
        
        logger.info(f"Loaded {keys_loaded} API keys from {filepath}")
        return keys_loaded
    
    async def get_next_key(self) -> Optional[str]:
        """Get the next available API key using round-robin selection."""
        async with self._lock:
            if not self._keys:
                logger.error("No API keys available")
                return None
            
            # Try each key once
            attempts = len(self._keys)
            for _ in range(attempts):
                key = self._keys[self._current_index]
                self._current_index = (self._current_index + 1) % len(self._keys)
                
                if key.is_available():
                    key.mark_used()
                    logger.debug(f"Using key: {key.name}")
                    return key.key
            
            logger.warning("All API keys are exhausted or rate limited")
            return None
    
    async def report_success(self, api_key: str):
        """Report successful use of a key."""
        async with self._lock:
            for key in self._keys:
                if key.key == api_key:
                    if key.status == KeyStatus.RATE_LIMITED:
                        key.status = KeyStatus.ACTIVE
                    break
    
    async def report_rate_limit(self, api_key: str, cooldown_seconds: int = 60):
        """Report that a key hit a rate limit."""
        async with self._lock:
            for key in self._keys:
                if key.key == api_key:
                    key.mark_rate_limited(cooldown_seconds)
                    break
    
    async def report_error(self, api_key: str, is_auth_error: bool = False):
        """Report an error for a key."""
        async with self._lock:
            for key in self._keys:
                if key.key == api_key:
                    key.error_count += 1
                    if is_auth_error:
                        key.status = KeyStatus.INVALID
                        logger.error(f"Key {key.name} marked as invalid (auth error)")
                    elif key.error_count >= 5:
                        key.status = KeyStatus.EXHAUSTED
                        logger.warning(f"Key {key.name} marked as exhausted (too many errors)")
                    break
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all keys."""
        return {
            "total_keys": len(self._keys),
            "active_keys": sum(1 for k in self._keys if k.status == KeyStatus.ACTIVE),
            "rate_limited_keys": sum(1 for k in self._keys if k.status == KeyStatus.RATE_LIMITED),
            "invalid_keys": sum(1 for k in self._keys if k.status == KeyStatus.INVALID),
            "exhausted_keys": sum(1 for k in self._keys if k.status == KeyStatus.EXHAUSTED),
            "total_requests": sum(k.requests_made for k in self._keys),
            "keys": [
                {
                    "name": k.name,
                    "status": k.status.value,
                    "requests_made": k.requests_made,
                    "error_count": k.error_count
                }
                for k in self._keys
            ]
        }
    
    def reset_all_keys(self):
        """Reset all keys to active status."""
        for key in self._keys:
            key.reset_status()
        logger.info("All keys reset to active status")
    
    def has_available_keys(self) -> bool:
        """Check if any keys are available."""
        return any(k.is_available() for k in self._keys)
    
    @property
    def key_count(self) -> int:
        """Get total number of keys."""
        return len(self._keys)


# Cookie-based account rotation (for web-based authentication)
@dataclass
class GoogleAccount:
    """Represents a Google account with cookies for web authentication."""
    email: str
    cookie_1psid: str
    cookie_1psidts: str
    status: KeyStatus = KeyStatus.ACTIVE
    requests_made: int = 0
    last_used: float = 0.0
    cooldown_until: float = 0.0
    error_count: int = 0
    
    def is_available(self) -> bool:
        """Check if the account is available for use."""
        if self.status in (KeyStatus.INVALID, KeyStatus.EXHAUSTED):
            return False
        if self.status == KeyStatus.RATE_LIMITED:
            return time.time() > self.cooldown_until
        return True
    
    def mark_rate_limited(self, cooldown_seconds: int = 300):
        """Mark account as rate limited with cooldown period."""
        self.status = KeyStatus.RATE_LIMITED
        self.cooldown_until = time.time() + cooldown_seconds
        self.error_count += 1
        logger.warning(f"Account {self.email} rate limited for {cooldown_seconds}s")
    
    def mark_used(self):
        """Mark account as used."""
        self.requests_made += 1
        self.last_used = time.time()
        if self.status == KeyStatus.RATE_LIMITED and time.time() > self.cooldown_until:
            self.status = KeyStatus.ACTIVE


class AccountRotator:
    """
    Rotates through multiple Google accounts for web-based Gemini access.
    """
    
    def __init__(self):
        self._accounts: List[GoogleAccount] = []
        self._current_index: int = 0
        self._lock = asyncio.Lock()
    
    def add_account(self, email: str, cookie_1psid: str, cookie_1psidts: str) -> bool:
        """Add a Google account."""
        for acc in self._accounts:
            if acc.email == email:
                logger.warning(f"Account {email} already exists")
                return False
        
        account = GoogleAccount(
            email=email,
            cookie_1psid=cookie_1psid,
            cookie_1psidts=cookie_1psidts
        )
        self._accounts.append(account)
        logger.info(f"Added Google account: {email}")
        return True
    
    def load_accounts_from_file(self, filepath: str) -> int:
        """
        Load accounts from a JSON file.
        Format: [{"email": "...", "cookie_1psid": "...", "cookie_1psidts": "..."}]
        """
        path = Path(filepath)
        if not path.exists():
            logger.error(f"Accounts file not found: {filepath}")
            return 0
        
        accounts_loaded = 0
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for acc_data in data:
            if self.add_account(
                acc_data.get("email", f"account_{len(self._accounts)}"),
                acc_data.get("cookie_1psid", ""),
                acc_data.get("cookie_1psidts", "")
            ):
                accounts_loaded += 1
        
        logger.info(f"Loaded {accounts_loaded} accounts from {filepath}")
        return accounts_loaded
    
    async def get_next_account(self) -> Optional[GoogleAccount]:
        """Get the next available account using round-robin selection."""
        async with self._lock:
            if not self._accounts:
                logger.error("No accounts available")
                return None
            
            attempts = len(self._accounts)
            for _ in range(attempts):
                account = self._accounts[self._current_index]
                self._current_index = (self._current_index + 1) % len(self._accounts)
                
                if account.is_available():
                    account.mark_used()
                    logger.debug(f"Using account: {account.email}")
                    return account
            
            logger.warning("All accounts are exhausted or rate limited")
            return None
    
    async def report_rate_limit(self, email: str, cooldown_seconds: int = 300):
        """Report that an account hit a rate limit."""
        async with self._lock:
            for acc in self._accounts:
                if acc.email == email:
                    acc.mark_rate_limited(cooldown_seconds)
                    break
    
    async def report_auth_error(self, email: str):
        """Report authentication error for an account."""
        async with self._lock:
            for acc in self._accounts:
                if acc.email == email:
                    acc.status = KeyStatus.INVALID
                    logger.error(f"Account {email} marked as invalid (auth error)")
                    break
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all accounts."""
        return {
            "total_accounts": len(self._accounts),
            "active_accounts": sum(1 for a in self._accounts if a.status == KeyStatus.ACTIVE),
            "rate_limited_accounts": sum(1 for a in self._accounts if a.status == KeyStatus.RATE_LIMITED),
            "invalid_accounts": sum(1 for a in self._accounts if a.status == KeyStatus.INVALID),
            "total_requests": sum(a.requests_made for a in self._accounts),
            "accounts": [
                {
                    "email": a.email,
                    "status": a.status.value,
                    "requests_made": a.requests_made
                }
                for a in self._accounts
            ]
        }
    
    def has_available_accounts(self) -> bool:
        """Check if any accounts are available."""
        return any(a.is_available() for a in self._accounts)
    
    @property
    def account_count(self) -> int:
        """Get total number of accounts."""
        return len(self._accounts)
