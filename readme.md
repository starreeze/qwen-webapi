# Qwen Web API Proxy

Qwen API Proxy exposes the upstream [chat.qwen.ai](https://chat.qwen.ai) service through an OpenAI-compatible API. It provides a local Flask server and a Python API for issuing chat completions programmatically.

Features:

- OpenAI-compatible `v1/chat/completions` endpoint with streaming support
- Python API `QwenApi` for synchronous completions

## Installation

Install dependencies:

```bash
pip install flask flask-cors requests
```

The service reads configuration from environment variables:

| Variable | Description | Default |
| --- | --- | --- |
| `QWEN_AUTH_TOKEN` | Authorization token copied from chat.qwen.ai | _required_ |
| `QWEN_BASE_URL` | Upstream API base URL | `https://chat.qwen.ai` |
| `PORT` | Port for the Flask server | `5000` |
| `QWEN_DEBUG` | Set to `1` to enable debug logging | `0` |

For authentication, you can either set the `QWEN_AUTH_TOKEN` environment variable or create a `token.txt` file in the current working directory with the token. To get the token, follow the steps:

1. login to [chat.qwen.ai](https://chat.qwen.ai);
2. press F12 to open the browser's developer tools;
3. navigate to the `Console` tab;
4. input `localStorage.getItem("token")` and press Enter;
5. copy the token (without quotes) and paste it into the `token.txt` file or set the `QWEN_AUTH_TOKEN` environment variable.

## Usage

### Run the server

```bash
python -m qwen.server
```

The app binds to `0.0.0.0:<PORT>` and exposes the following endpoints:

- `GET /health` – health check
- `GET /v1/models` – list models
- `POST /v1/chat/completions` – OpenAI-compatible chat completions
- `DELETE /v1/chats/<chat_id>` – delete a stored chat session upstream

### Python API

```python
from qwen import QwenApi

messages = [{"role": "user", "content": "Hello, Qwen!"}]
client = QwenApi("qwen-flash")
response = client(
    messages,
    enable_thinking=True,
    thinking_budget=2048,
)
print(response)
```

`QwenApi` raises `ConfigurationError` when the token is missing and `QwenAPIError` for upstream request issues.

#### Chat parameters

- `messages` (list): List of messages in the format of [{"role": "user", "content": "Hello, Qwen!"}]. Or a single string containing the user query.
- `model` (str): Model name. The name mapping is defined in `qwen/config.py`, including:
    ```python
    DEFAULT_MODEL_MAP: dict[str, str] = {
        "qwen": "qwen3-max",
        "qwen-think": "qwen3-235b-a22b",  # 2507
        "qwen-coder": "qwen3-coder-plus",
        "qwen-flash": "qwen-plus-2025-09-11",  # next-80b-a3b
        "qwen-vl": "qwen3-vl-plus",  # Qwen3-VL-235B-A22B
    }
    ```
- `enable_thinking` (bool): Set `True` to request autodocumented reasoning. Defaults to the upstream user preference when omitted. Not available on all models.
- `thinking_budget` (int): Optional token budget for reasoning content. Only used when `enable_thinking=True`.
- `stream` (bool): Force streaming behaviour. `QwenApi` always disables streaming, but the REST endpoint honours the flag.

## Notes

1. rate limit unknown - not recommended to use on main account
2. currently only supports single, pure-text message

## License

GPLv3 License
