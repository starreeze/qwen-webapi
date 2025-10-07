"""Flask application providing OpenAI-compatible endpoints."""

from __future__ import annotations

import logging
from typing import Tuple

from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS

from .core import RequestManager
from .utils import AppConfig, ConfigurationError, QwenAPIError, load_config

logger = logging.getLogger(__name__)


def _create_manager(config: AppConfig) -> RequestManager:
    if not config.auth_token:
        raise ConfigurationError("QWEN_AUTH_TOKEN is required to start the server")

    manager = RequestManager(
        auth_token=config.auth_token, base_url=config.base_url, model_map=config.model_map
    )
    manager.initialize()
    return manager


def create_app(config: AppConfig | None = None) -> Flask:
    config = config or load_config()
    config.configure_logging(force=True)
    app = Flask(__name__)
    CORS(app)
    manager = _create_manager(config)

    @app.route("/v1/models", methods=["GET"])
    def list_models() -> Tuple[Response, int] | Response:
        try:
            openai_models = []
            for model_id, model_info in manager.models_info.items():
                info = model_info.get("info", {})
                openai_models.append(
                    {
                        "id": info.get("id", model_id),
                        "object": "model",
                        "created": info.get("created_at"),
                        "owned_by": model_info.get("owned_by"),
                    }
                )
            return jsonify({"object": "list", "data": openai_models})
        except Exception as exc:  # catch-all to return errors gracefully
            logger.exception("Failed to list models")
            return (
                jsonify(
                    {
                        "error": {
                            "message": f"Failed to fetch models: {exc}",
                            "type": "server_error",
                            "param": None,
                            "code": None,
                        }
                    }
                ),
                500,
            )

    @app.route("/v1/chat/completions", methods=["POST"])
    def chat_completions_endpoint() -> Tuple[Response, int] | Response:
        openai_request = request.get_json()
        if not openai_request:
            return (
                jsonify(
                    {
                        "error": {
                            "message": "Invalid JSON payload",
                            "type": "invalid_request_error",
                            "param": None,
                            "code": None,
                        }
                    }
                ),
                400,
            )

        try:
            is_streaming, payload = manager.chat_completions(openai_request)
        except QwenAPIError as exc:
            logger.exception("chat completion failed")
            return (
                jsonify(
                    {"error": {"message": str(exc), "type": "server_error", "param": None, "code": None}}
                ),
                500,
            )

        if is_streaming:
            assert hasattr(payload, "__iter__")
            return Response(stream_with_context(payload), content_type="text/event-stream")  # type: ignore
        return jsonify(payload)

    @app.route("/v1/chats/<chat_id>", methods=["DELETE"])
    def delete_chat(chat_id: str) -> Tuple[Response, int] | Response:
        try:
            success = manager.delete_chat(chat_id)
            if success:
                return jsonify({"message": f"Chat {chat_id} deleted", "success": True})
            return jsonify({"message": f"Failed to delete chat {chat_id}", "success": False}), 400
        except QwenAPIError as exc:
            logger.exception("Failed to delete chat %s", chat_id)
            return (
                jsonify(
                    {"error": {"message": str(exc), "type": "server_error", "param": None, "code": None}}
                ),
                500,
            )

    @app.route("/", methods=["GET"])
    def index() -> Response:
        return jsonify(
            {
                "message": "Qwen OpenAI-compatible proxy running",
                "docs": "https://platform.openai.com/docs/api-reference/chat",
            }
        )

    @app.route("/health", methods=["GET"])
    def health() -> Tuple[Response, int]:
        return jsonify({"status": "healthy"}), 200

    return app


def main() -> None:
    config = load_config()
    app = create_app(config)
    logger.info("Starting server on port %s", config.port)
    app.run(host="0.0.0.0", port=config.port, debug=config.debug)


if __name__ == "__main__":
    main()
