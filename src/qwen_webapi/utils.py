"""Configuration helpers and logging utilities for the Qwen API proxy."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

DEFAULT_MODEL_MAP: dict[str, str] = {
    "qwen": "qwen3-max",
    "qwen-think": "qwen3-235b-a22b",  # 2507
    "qwen-coder": "qwen3-coder-plus",
    "qwen-flash": "qwen-plus-2025-09-11",  # next-80b-a3b
    "qwen-vl": "qwen3-vl-plus",  # Qwen3-VL-235B-A22B
}

DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def _parse_log_level(value: str | None) -> int | None:
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return int(value)
    level = getattr(logging, value.upper(), None)
    return level if isinstance(level, int) else None


@dataclass(slots=True)
class AppConfig:
    """Represents runtime configuration for the application."""

    auth_token: str | None = None
    base_url: str = "https://chat.qwen.ai"
    port: int = int(os.getenv("PORT", "5000"))
    debug: bool = os.getenv("QWEN_DEBUG", "0") == "1"
    log_level: int | None = field(default=None, repr=False)
    log_format: str = field(default=DEFAULT_LOG_FORMAT, repr=False)
    model_map: dict[str, str] = field(default_factory=lambda: DEFAULT_MODEL_MAP.copy())

    @property
    def effective_log_level(self) -> int:
        return self.log_level if self.log_level is not None else logging.DEBUG if self.debug else logging.INFO

    def configure_logging(self, *, force: bool = False, extra_kwargs: dict[str, Any] | None = None) -> None:
        configure_logging(
            level=self.effective_log_level, log_format=self.log_format, force=force, extra_kwargs=extra_kwargs
        )


def configure_logging(
    *,
    level: int | None = None,
    log_format: str = DEFAULT_LOG_FORMAT,
    force: bool = False,
    extra_kwargs: dict[str, Any] | None = None,
) -> None:
    """Configure global logging using application settings."""

    effective_level = level if level is not None else logging.INFO
    kwargs: dict[str, Any] = {"level": effective_level, "format": log_format}
    if force:
        kwargs["force"] = True
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    logging.basicConfig(**kwargs)


def load_config(token="") -> AppConfig:
    """Load configuration from environment variables."""

    auth_token = token or os.getenv("QWEN_AUTH_TOKEN") or open("token.txt", "r").read().strip()
    base_url = os.getenv("QWEN_BASE_URL", "https://chat.qwen.ai")
    log_level = _parse_log_level(os.getenv("QWEN_LOG_LEVEL"))
    log_format = os.getenv("QWEN_LOG_FORMAT", DEFAULT_LOG_FORMAT)

    return AppConfig(auth_token=auth_token, base_url=base_url, log_level=log_level, log_format=log_format)


"""Custom exceptions used across the Qwen API proxy."""


class QwenAPIError(RuntimeError):
    """Raised when interaction with the upstream Qwen API fails."""


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


"""Telemetry and instrumentation helpers."""

logger = logging.getLogger(__name__)
T = TypeVar("T")


def instrument_call(name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that logs execution time and exceptions of a function."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs):
            logger.debug("%s: start", name)
            try:
                result = func(*args, **kwargs)
                logger.debug("%s: success", name)
                return result
            except Exception:  # pragma: no cover - best effort logging
                logger.exception("%s: failed", name)
                raise

        return wrapper

    return decorator
