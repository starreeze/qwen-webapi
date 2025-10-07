"""Python API wrapper for issuing chat completions."""

from __future__ import annotations

import logging
from typing import Any

import requests

from .core import RequestManager
from .utils import ConfigurationError, QwenAPIError, load_config

logger = logging.getLogger(__name__)
_MANAGER: RequestManager | None = None


def _ensure_manager(token="") -> RequestManager:
    global _MANAGER
    if _MANAGER is not None:
        return _MANAGER

    config = load_config(token)
    if not config.auth_token:
        raise ConfigurationError("QWEN_AUTH_TOKEN environment variable is not set")

    config.configure_logging()
    manager = RequestManager(
        auth_token=config.auth_token,
        base_url=config.base_url,
        model_map=config.model_map,
        session=requests.Session(),
    )
    manager.initialize()
    _MANAGER = manager
    return manager


class QwenApi:
    """Callable helper for issuing chat completions."""

    def __init__(
        self,
        model="qwen",
        enable_thinking: bool = False,
        thinking_budget: int | None = None,
        token="",
        **default_kwargs: Any,
    ) -> None:
        self._manager = _ensure_manager(token)
        self._model = model
        self._default_kwargs = default_kwargs | {"enable_thinking": enable_thinking}
        if thinking_budget is not None:
            self._default_kwargs["thinking_budget"] = thinking_budget

    def __call__(
        self,
        messages: list[dict] | str,
        *,
        image_paths: list[str] | None = None,
        enable_thinking: bool | None = None,
        thinking_budget: int | None = None,
        **kwargs: Any,
    ) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # Add image_paths to the first user message if provided
        if image_paths and len(messages) > 0:
            first_message = messages[0]
            if isinstance(first_message, dict) and first_message.get("role") == "user":
                first_message["image_paths"] = image_paths

        request_body = {"model": self._model, "messages": messages, **self._default_kwargs, **kwargs}
        if enable_thinking is not None:
            request_body["enable_thinking"] = enable_thinking
        if thinking_budget is not None:
            request_body["thinking_budget"] = thinking_budget

        is_streaming, payload = self._manager.chat_completions(request_body)
        if is_streaming:
            raise QwenAPIError("QwenApi does not support streaming responses")

        if not isinstance(payload, dict):
            raise QwenAPIError("Unexpected response payload")

        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise QwenAPIError(f"Malformed response payload: {payload}") from exc
