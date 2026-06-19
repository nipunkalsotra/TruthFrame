import asyncio
import time
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class TokenBucket:
    """A dynamic token bucket for managing TPM (Tokens Per Minute) and RPM (Requests Per Minute)."""
    def __init__(self, tpm: int, rpm: int):
        self.tpm_limit = tpm
        self.rpm_limit = rpm
        
        self.tpm_tokens = float(tpm)
        self.rpm_tokens = float(rpm)
        
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()

    async def consume(self, tokens: int = 0) -> bool:
        """Consume tokens and a request slot. Returns True if successful."""
        async with self.lock:
            self._replenish()
            
            if self.rpm_tokens >= 1 and self.tpm_tokens >= tokens:
                self.rpm_tokens -= 1
                self.tpm_tokens -= tokens
                return True
            return False

    def _replenish(self):
        """Replenish tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_update
        
        # Replenish TPM (Tokens Per Minute)
        self.tpm_tokens = min(
            self.tpm_limit,
            self.tpm_tokens + (elapsed * (self.tpm_limit / 60.0))
        )
        
        # Replenish RPM (Requests Per Minute)
        self.rpm_tokens = min(
            self.rpm_limit,
            self.rpm_tokens + (elapsed * (self.rpm_limit / 60.0))
        )
        
        self.last_update = now

    async def wait_for_slot(self, tokens: int = 0, timeout: float = 60.0):
        """Wait until tokens and a request slot are available."""
        start_time = time.monotonic()
        while True:
            if await self.consume(tokens):
                return True
            
            if time.monotonic() - start_time > timeout:
                return False
                
            # Wait a short interval before retrying
            await asyncio.sleep(0.5)

class GlobalRateLimiter:
    """Centralized rate limiter for multiple LLM/VLM services."""
    def __init__(self):
        # Default limits (can be overridden by environment variables)
        self.buckets: Dict[str, TokenBucket] = {
            "gemini": TokenBucket(tpm=1000000, rpm=15),  # Example Gemini Free Tier limits
            "groq": TokenBucket(tpm=30000, rpm=30)       # Example Groq limits
        }

    def set_limit(self, service: str, tpm: int, rpm: int):
        self.buckets[service] = TokenBucket(tpm, rpm)

    async def acquire(self, service: str, estimated_tokens: int = 1000):
        """Acquire a slot for the specified service."""
        bucket = self.buckets.get(service)
        if not bucket:
            return True # No limit defined for this service
            
        success = await bucket.wait_for_slot(estimated_tokens)
        if not success:
            logger.warning(f"Rate limit timeout for service: {service}")
        return success
