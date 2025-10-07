"""Core logic for interacting with the upstream Qwen API."""

from __future__ import annotations

import base64
import copy
import json
import logging
import mimetypes
import time
import uuid
from pathlib import Path
from typing import Iterator, Optional, Tuple

import requests

from .oss.sign import SignerV4
from .oss.types import Credentials, HttpRequest, SigningContext
from .utils import QwenAPIError, instrument_call

logger = logging.getLogger(__name__)


class RequestManager:
    """Handles requests to the upstream Qwen chat service."""

    def __init__(
        self,
        auth_token: str,
        base_url: str,
        model_map: dict[str, str],
        user_settings: Optional[dict] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        if not auth_token:
            raise ValueError("auth_token must be provided")

        self.auth_token = auth_token
        self.base_url = base_url.rstrip("/")
        self.model_map = model_map
        self.user_settings = user_settings or {}
        self.session = session or requests.Session()

        self.session.headers.update(
            {
                "accept-encoding": "gzip, deflate, br, zstd",
                "accept-language": "en-US,en;q=0.9",
                "content-type": "application/json",
                "source": "web",
                "DNT": "1",
                "Host": "chat.qwen.ai",
                "Origin": "https://chat.qwen.ai",
                "Referer": "https://chat.qwen.ai/",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:136.0) Gecko/20100101 Firefox/136.0",
            }
        )
        self._update_auth_header()

        self.models_info: dict[str, dict]
        self.user_info: dict

    # ------------------------------------------------------------------
    # Internal HTTP helpers

    def _update_auth_header(self) -> None:
        self.session.headers.update({"authorization": f"Bearer {self.auth_token}"})

    def _get(self, path: str) -> dict:
        response = self.session.get(f"{self.base_url}{path}")
        response.raise_for_status()
        return response.json()

    def _post(
        self, path: str, payload: dict, *, stream: bool = False, headers: Optional[dict] = None
    ) -> requests.Response:
        logger.debug("POST %s payload keys=%s", path, list(payload))
        response = self.session.post(f"{self.base_url}{path}", json=payload, stream=stream, headers=headers)
        response.raise_for_status()
        return response

    def _delete(self, path: str) -> dict:
        response = self.session.delete(f"{self.base_url}{path}")
        response.raise_for_status()
        return response.json()

    def _put(self, url: str, data: bytes, headers: Optional[dict] = None) -> requests.Response:
        """PUT request for uploading files to OSS."""

        response = self.session.put(url, data=data, headers=headers)
        response.raise_for_status()
        return response

    # ------------------------------------------------------------------
    # Image upload support

    @instrument_call("request_manager.upload_image")
    def upload_image(self, image_path: str, user_id: str) -> dict:
        """Upload an image and return file metadata for use in chat messages."""
        path = Path(image_path)
        if not path.exists():
            raise QwenAPIError(f"Image file not found: {image_path}")

        # Get file info
        filename = path.name
        filesize = path.stat().st_size
        mime_type = mimetypes.guess_type(filename)[0] or "image/png"

        # Validate image format
        if not mime_type.startswith("image/"):
            raise QwenAPIError(f"File is not an image: {filename}")

        file_ext = mime_type.split("/")[1]
        if file_ext not in ["jpeg", "jpg", "png"]:
            raise QwenAPIError(f"Unsupported image format: {file_ext}. Only jpg and png are supported.")

        # Step 1: Get STS token
        logger.debug("Requesting STS token for %s", filename)
        sts_payload = {"filename": filename, "filesize": filesize, "filetype": "image"}
        sts_response = self._post("/api/v2/files/getstsToken", sts_payload)
        sts_data = sts_response.json()

        if not sts_data.get("success"):
            raise QwenAPIError(f"Failed to get STS token: {sts_data}")

        file_data = sts_data["data"]
        upload_url = f"https://{file_data['bucketname']}.{file_data['endpoint']}/{file_data['file_path']}"
        file_url = file_data["file_url"]
        file_id = file_data["file_id"]

        # Step 2: Upload file to OSS with V4 signature
        logger.debug("Uploading image to OSS: %s", upload_url)
        with open(image_path, "rb") as f:
            file_content = f.read()

        # Create credentials from STS token
        credentials = Credentials(
            access_key_id=file_data["access_key_id"],
            access_key_secret=file_data["access_key_secret"],
            security_token=file_data["security_token"],
        )

        # Extract region from endpoint
        region = file_data["region"].replace("oss-", "") if "oss-" in file_data["region"] else "cn-hangzhou"

        # Create HTTP request
        headers = copy.deepcopy(self.session.headers)
        headers.pop("authorization")
        headers.update(
            {
                "content-type": mime_type,
                "content-length": str(filesize),
                "host": file_data["bucketname"] + "." + file_data["endpoint"],
            }
        )
        http_request = HttpRequest(method="PUT", url=upload_url, headers=headers, body=file_content)  # type: ignore

        # Create signing context
        signing_context = SigningContext(
            product="oss",
            region=region,
            bucket=file_data["bucketname"],
            key=file_data["file_path"],
            request=http_request,
            credentials=credentials,
        )

        # Sign the request
        signer = SignerV4()
        signer.sign(signing_context)

        # Use signed headers for upload
        upload_headers = dict(http_request.headers)
        self._put(upload_url, file_content, headers=upload_headers)

        # Step 3: Return file metadata in the format expected by chat API
        timestamp_ms = int(time.time() * 1000)
        return {
            "type": "image",
            "file": {
                "created_at": timestamp_ms,
                "data": {},
                "filename": filename,
                "hash": None,
                "id": file_id,
                "user_id": user_id,
                "meta": {"name": filename, "size": filesize, "content_type": mime_type},
                "update_at": timestamp_ms,
            },
            "id": file_id,
            "url": file_url,
            "name": filename,
            "collection_name": "",
            "progress": 0,
            "status": "uploaded",
            "greenNet": "success",
            "size": filesize,
            "error": "",
            "itemId": str(uuid.uuid4()),
            "file_type": mime_type,
            "showType": "image",
            "file_class": "vision",
            "uploadTaskId": str(uuid.uuid4()),
        }

    # ------------------------------------------------------------------
    # Initialization and metadata

    @instrument_call("request_manager.initialize")
    def initialize(self) -> None:
        try:
            self.user_info = self._get("/api/v1/auths/")
            models_response = self._get("/api/models")
            data = models_response.get("data")
            if not data:
                raise QwenAPIError("Missing model data in response")
            self.models_info = {model["id"]: model for model in data}

            settings_response = self._get("/api/v2/users/user/settings")
            self.user_settings = settings_response.get("data", {})
            logger.info("Fetched user settings and model list")
        except requests.exceptions.RequestException as exc:
            raise QwenAPIError(f"Unable to initialize Qwen client: {exc}") from exc

        # History synchronization is handled by higher-level applications.

    # ------------------------------------------------------------------
    # Core chat interaction

    def resolve_model_id(self, model_name: str) -> str:
        mapped_id = self.model_map.get(model_name)
        if mapped_id and mapped_id in self.models_info:
            return mapped_id
        if model_name in self.models_info:
            return model_name
        logger.warning("Unknown model '%s', falling back to default", model_name)
        return "qwen3-max"

    @instrument_call("request_manager.create_chat")
    def create_chat(self, model_id: str, title: str) -> str:
        payload = {
            "title": title,
            "models": [model_id],
            "chat_mode": "normal",
            "chat_type": "t2t",
            "timestamp": int(time.time() * 1000),
        }
        response = self._post("/api/v2/chats/new", payload)
        data = response.json()
        chat_id = data.get("data", {}).get("id")
        if not chat_id:
            raise QwenAPIError("Missing chat_id in create response")
        logger.info("Created upstream chat %s", chat_id)
        return chat_id

    @instrument_call("request_manager.delete_chat")
    def delete_chat(self, chat_id: str) -> bool:
        try:
            data = self._delete(f"/api/v2/chats/{chat_id}")
            success = data.get("success", False)
            if success:
                logger.info("Deleted upstream chat %s", chat_id)
            else:
                logger.warning("Upstream declined to delete chat %s", chat_id)
            return success
        except requests.exceptions.RequestException as exc:
            raise QwenAPIError(f"Failed to delete chat {chat_id}: {exc}") from exc

    @instrument_call("request_manager.chat_completions")
    def chat_completions(self, request_body: dict) -> Tuple[bool, Iterator[str] | dict]:
        messages = request_body["messages"]
        if not (isinstance(messages, list) and len(messages) == 1):
            raise QwenAPIError("Invalid messages format. Only single user message is supported.")

        # Parse message content and images
        user_message = messages[0]
        user_input, image_files = self._parse_message_content(user_message)

        model_name = request_body.get("model", "qwen3")
        stream = bool(request_body.get("stream", False))
        enable_thinking = request_body.get("enable_thinking", False)
        thinking_budget = request_body.get("thinking_budget")

        model_id = self.resolve_model_id(model_name)
        chat_id = self.create_chat(model_id, title="New Chat")

        feature_config = self._build_feature_config(model_id, enable_thinking, thinking_budget)
        payload = self._build_payload(chat_id, model_id, user_input, feature_config, image_files)

        if stream:
            return True, self._stream_chat(chat_id, model_name, payload)

        return False, self._non_stream_chat(chat_id, model_name, payload)

    def _parse_message_content(self, user_message: dict) -> Tuple[str, list[dict]]:
        """Parse user message and extract text content and images."""
        content = user_message.get("content", "")
        image_files = []

        # Get user_id for image uploads
        user_id = self.user_info["id"]

        # Handle OpenAI format with content as list
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type", "")

                    if item_type == "text" or item_type == "input_text":
                        text_parts.append(item.get("text", ""))

                    elif item_type == "image_url" or item_type == "input_image":
                        # Handle OpenAI format: {"type": "image_url", "image_url": {"url": "..."}}
                        if "image_url" in item:
                            image_url = item["image_url"]
                            if isinstance(image_url, dict):
                                url = image_url.get("url", "")
                            else:
                                url = image_url
                        else:
                            url = item.get("image_url", "")

                        if url:
                            # Handle base64 encoded images
                            if url.startswith("data:image/"):
                                image_file = self._handle_base64_image(url, user_id)
                                if image_file:
                                    image_files.append(image_file)
                            # Handle file paths
                            else:
                                image_file = self.upload_image(url, user_id)
                                image_files.append(image_file)

            user_input = " ".join(text_parts)

        # Handle simple string content
        elif isinstance(content, str):
            user_input = content
            # Check if there are image_paths in the message
            if "image_paths" in user_message:
                for image_path in user_message["image_paths"]:
                    image_file = self.upload_image(image_path, user_id)
                    image_files.append(image_file)
        else:
            user_input = str(content)

        return user_input, image_files

    def _handle_base64_image(self, data_url: str, user_id: str) -> Optional[dict]:
        """Handle base64 encoded image from data URL."""
        try:
            # Parse data URL: data:image/png;base64,<base64_data>
            if "," not in data_url:
                logger.warning("Invalid data URL format")
                return None

            header, base64_data = data_url.split(",", 1)

            # Extract mime type
            if ";" in header:
                mime_part = header.split(";")[0]
                mime_type = mime_part.replace("data:", "")
            else:
                mime_type = "image/png"

            # Decode base64
            image_data = base64.b64decode(base64_data)

            # Generate temporary filename
            ext = mime_type.split("/")[1]
            temp_filename = f"temp_image_{uuid.uuid4()}.{ext}"
            temp_path = Path(temp_filename)

            # Write to temporary file
            with open(temp_path, "wb") as f:
                f.write(image_data)

            try:
                # Upload the image
                image_file = self.upload_image(str(temp_path), user_id)
                return image_file
            finally:
                # Clean up temporary file
                if temp_path.exists():
                    temp_path.unlink()

        except Exception as exc:
            logger.error("Failed to handle base64 image: %s", exc)
            return None

    def _build_feature_config(
        self, model_id: str, enable_thinking: bool, thinking_budget: Optional[int]
    ) -> dict:
        feature_config: dict[str, object] = {"output_schema": "phase"}
        if enable_thinking:
            feature_config["thinking_enabled"] = True
            if thinking_budget is not None:
                feature_config["thinking_budget"] = thinking_budget
            else:
                default_budget = (
                    self.user_settings.get("model_config", {}).get(model_id, {}).get("thinking_budget")
                )
                if default_budget:
                    feature_config["thinking_budget"] = default_budget
        else:
            feature_config["thinking_enabled"] = False
        return feature_config

    @staticmethod
    def _build_payload(
        chat_id: str,
        model_id: str,
        user_input: str,
        feature_config: dict,
        image_files: Optional[list[dict]] = None,
    ) -> dict:
        timestamp_ms = int(time.time() * 1000)
        return {
            "stream": True,
            "incremental_output": True,
            "chat_id": chat_id,
            "chat_mode": "normal",
            "model": model_id,
            "parent_id": None,
            "messages": [
                {
                    "fid": str(uuid.uuid4()),
                    "parentId": None,
                    "childrenIds": [str(uuid.uuid4())],
                    "role": "user",
                    "content": user_input,
                    "user_action": "chat",
                    "files": image_files or [],
                    "timestamp": timestamp_ms,
                    "models": [model_id],
                    "chat_type": "t2t",
                    "feature_config": feature_config,
                    "extra": {"meta": {"subChatType": "t2t"}},
                    "sub_chat_type": "t2t",
                    "parent_id": None,
                }
            ],
            "timestamp": timestamp_ms,
        }

    # ------------------------------------------------------------------
    # Internal chat helpers

    def _stream_chat(self, chat_id: str, model_name: str, payload: dict) -> Iterator[str]:
        assistant_content = ""
        reasoning_text = ""
        finish_reason = "stop"

        try:
            with self._post(
                f"/api/v2/chat/completions?chat_id={chat_id}",
                payload,
                stream=True,
                headers={"x-accel-buffering": "no"},
            ) as response:
                for line in response.iter_lines(decode_unicode=True):
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        final_chunk = {
                            "id": f"chatcmpl-{chat_id[:10]}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                        }
                        yield f"data: {json.dumps(final_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.debug("Skipping malformed stream chunk: %s", data_str)
                        continue

                    choices = data.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    phase = delta.get("phase")
                    status = delta.get("status")
                    content = delta.get("content", "")

                    if phase == "think" and status != "finished":
                        reasoning_text += content
                    elif phase in {"answer", None} and content:
                        assistant_content += content
                        chunk = {
                            "id": f"chatcmpl-{chat_id[:10]}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                        }
                        if reasoning_text:
                            chunk["choices"][0]["delta"]["reasoning_content"] = reasoning_text
                            reasoning_text = ""
                        yield f"data: {json.dumps(chunk)}\n\n"

                    if status == "finished":
                        finish_reason = delta.get("finish_reason", "stop")
        except requests.exceptions.RequestException as exc:
            logger.error("Streaming request failed: %s", exc)
            error_chunk = {
                "id": "chatcmpl-error",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": f"Error during streaming: {exc}"},
                        "finish_reason": "error",
                    }
                ],
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
        finally:
            return

    def _non_stream_chat(self, chat_id: str, model_name: str, payload: dict) -> dict:
        response_text = ""
        reasoning_text = ""
        finish_reason = "stop"
        usage_data = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        try:
            with self._post(
                f"/api/v2/chat/completions?chat_id={chat_id}",
                payload,
                stream=True,
                headers={"x-accel-buffering": "no"},
            ) as response:
                for line in response.iter_lines(decode_unicode=True):
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.debug("Skipping malformed stream chunk: %s", data_str)
                        continue

                    choices = data.get("choices") or []
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    phase = delta.get("phase")
                    status = delta.get("status")

                    if phase == "think" and status != "finished":
                        reasoning_text += delta.get("content", "")
                    elif phase == "answer" and status != "finished":
                        response_text += delta.get("content", "")

                    if "usage" in data:
                        usage = data["usage"]
                        usage_data = {
                            "prompt_tokens": usage.get("input_tokens", 0),
                            "completion_tokens": usage.get("output_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                        }

                    if status == "finished":
                        finish_reason = delta.get("finish_reason", "stop")
        except requests.exceptions.RequestException as exc:
            raise QwenAPIError(f"Chat completion failed: {exc}") from exc

        response_payload = {
            "id": f"chatcmpl-{chat_id[:10]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage_data,
        }
        if reasoning_text:
            response_payload["choices"][0]["message"]["reasoning_content"] = reasoning_text

        return response_payload
