
import os
from itertools import cycle
from typing import Optional
from ccba_rag.core.settings import settings
from ccba_rag.utils.logging import get_logger

logger = get_logger(__name__)

class KeyManager:
    """
    Manages API keys for services with quota limits (like LlamaCloud).
    Supports rotation and round-robin selection.
    """

    def __init__(self, key_env_var: str = "LLAMA_CLOUD_API_KEY", setting_attr: str = "llama_cloud_api_key"):
        self.key_env_var = key_env_var
        self.setting_attr = setting_attr
        self._keys: list[str] = []
        self._iterator = None
        self._load_keys()

    def _load_keys(self):
        """Load keys from environment or settings."""
        # Get raw string
        keys_str = os.getenv(self.key_env_var) or getattr(settings, self.setting_attr, "") or ""

        # Parse comma-separated keys
        self._keys = [k.strip() for k in keys_str.split(",") if k.strip()]

        if not self._keys:
            logger.warning(f"No API keys found for {self.key_env_var}")

        self._iterator = cycle(self._keys)
        logger.info(f"Loaded {len(self._keys)} keys for {self.key_env_var}")

    def get_keys(self) -> List[str]:
        """Return all available keys."""
        return self._keys

    def get_next_key(self) -> Optional[str]:
        """Get the next key in rotation."""
        if not self._keys:
            return None
        return next(self._iterator)

    @property
    def key_count(self) -> int:
        return len(self._keys)

# Singleton instances
llama_key_manager = KeyManager(key_env_var="LLAMA_CLOUD_API_KEY", setting_attr="llama_cloud_api_key")
