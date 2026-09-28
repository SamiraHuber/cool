"""Shared VLM client factory and runtime selection registry for the frontend agent."""

from dataclasses import dataclass
import os
import socket
import struct
from threading import Lock
from urllib.parse import urlparse, urlunparse

from openai import OpenAI

VLM_API_URL = os.getenv("VLM_API_URL", "http://42b9e761e7e5:8000/v1")
VLM_MODEL = os.getenv("VLM_MODEL", "Qwen/Qwen3-VL-4B-Instruct")
VLM_FALLBACK_PORT = os.getenv("VLM_FALLBACK_PORT", "8002")

# The local on-prem vLLM server (docker service `vlm_server`), serving the 4B instruct
# model. This is the "local" dropdown option and is independent of VLM_API_URL, which may
# be pointed at a remote (e.g. DGX-tunnelled) server instead.
LOCAL_VLM_API_URL = os.getenv("LOCAL_VLM_API_URL", "http://vlm_server:8000/v1")
LOCAL_VLM_MODEL = os.getenv("LOCAL_VLM_MODEL", "Qwen/Qwen3-VL-4B-Instruct")

USE_KIMI = os.getenv("USE_KIMI", "").lower() in ("1", "true", "yes")
KIMI_API_URL = os.getenv("KIMI_API_URL", "https://api.moonshot.ai/v1")
KIMI_API_KEY = os.getenv("KIMI_API_KEY", "")
KIMI_MODEL = os.getenv("KIMI_MODEL", "kimi-k2.6")

USE_GEMINI = os.getenv("USE_GEMINI", "").lower() in ("1", "true", "yes")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")


@dataclass(frozen=True)
class VlmOption:
    id: str
    label: str
    source: str
    model: str
    base_url: str
    api_key: str
    enabled: bool = True
    reason: str = ""
    supports_required_tool_choice: bool = True
    candidate_urls: tuple[str, ...] = ()
    # Whether to force the Qwen3 "thinking" block on (True), off (False), or leave the
    # server default (None). Only meaningful for reasoning-style models served by vLLM;
    # translated into chat_template_kwargs.enable_thinking by get_vlm_extra_body().
    thinking: bool | None = None


_vlm_client: OpenAI | None = None
_vlm_client_option_id: str | None = None
_vlm_state_lock = Lock()


def _default_gateway_ip() -> str | None:
    try:
        with open("/proc/net/route", "r", encoding="utf-8") as route_file:
            for line in route_file.readlines()[1:]:
                fields = line.strip().split()
                if len(fields) < 3 or fields[1] != "00000000":
                    continue
                return socket.inet_ntoa(struct.pack("<L", int(fields[2], 16)))
    except Exception:
        return None
    return None


def _candidate_vlm_urls() -> list[str]:
    candidates: list[str] = []

    def add(url: str | None) -> None:
        if not url:
            return
        normalized = url.rstrip("/")
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    add(VLM_API_URL)

    parsed = urlparse(VLM_API_URL)
    path = parsed.path or "/v1"
    fallback_port = VLM_FALLBACK_PORT
    if parsed.scheme and fallback_port:
        add(urlunparse((parsed.scheme, f"127.0.0.1:{fallback_port}", path, "", "", "")))
        gateway_ip = _default_gateway_ip()
        if gateway_ip:
            add(urlunparse((parsed.scheme, f"{gateway_ip}:{fallback_port}", path, "", "", "")))

    return candidates


def _registered_options() -> list[VlmOption]:
    options = [
        # Default backend from VLM_API_URL / VLM_MODEL env (currently the DGX-tunnelled
        # Qwen3.5-9B). thinking=None leaves the server default (reasoning ON for Qwen3.5).
        VlmOption(
            id="default",
            label="Default (env)",
            source="local",
            model=VLM_MODEL,
            base_url=VLM_API_URL,
            api_key="not-needed",
            candidate_urls=tuple(_candidate_vlm_urls()),
            thinking=None,
        ),
        # DGX Qwen3.5-9B with reasoning explicitly ON.
        VlmOption(
            id="dgx_9b_think",
            label="DGX Qwen3.5-9B (thinking)",
            source="local",
            model=VLM_MODEL,
            base_url=VLM_API_URL,
            api_key="not-needed",
            candidate_urls=tuple(_candidate_vlm_urls()),
            thinking=True,
        ),
        # DGX Qwen3.5-9B with reasoning explicitly OFF (faster, terse, direct JSON).
        VlmOption(
            id="dgx_9b_nothink",
            label="DGX Qwen3.5-9B (no thinking)",
            source="local",
            model=VLM_MODEL,
            base_url=VLM_API_URL,
            api_key="not-needed",
            candidate_urls=tuple(_candidate_vlm_urls()),
            thinking=False,
        ),
        # The local on-prem 4B instruct model (docker vlm_server). No reasoning block.
        VlmOption(
            id="local",
            label="Local Qwen3-VL-4B",
            source="local",
            model=LOCAL_VLM_MODEL,
            base_url=LOCAL_VLM_API_URL,
            api_key="not-needed",
            candidate_urls=(LOCAL_VLM_API_URL,),
            thinking=None,
        ),
        VlmOption(
            id="kimi",
            label="Kimi",
            source="kimi",
            model=KIMI_MODEL,
            base_url=KIMI_API_URL,
            api_key=KIMI_API_KEY,
            enabled=bool(KIMI_API_KEY),
            reason="" if KIMI_API_KEY else "KIMI_API_KEY is not set.",
            supports_required_tool_choice=False,
        ),
        VlmOption(
            id="gemini",
            label="Gemini",
            source="gemini",
            model=GEMINI_MODEL,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=GEMINI_API_KEY,
            enabled=bool(GEMINI_API_KEY),
            reason="" if GEMINI_API_KEY else "GEMINI_API_KEY is not set.",
        ),
    ]
    return options


def _registered_options_by_id() -> dict[str, VlmOption]:
    return {option.id: option for option in _registered_options()}


def _initial_option_id() -> str:
    if USE_GEMINI and GEMINI_API_KEY:
        return "gemini"
    if USE_KIMI and KIMI_API_KEY:
        return "kimi"
    return "default"


_active_vlm_option_id = _initial_option_id()


def _option_to_dict(option: VlmOption, *, active: bool) -> dict:
    return {
        "id": option.id,
        "label": option.label,
        "source": option.source,
        "model": option.model,
        "enabled": option.enabled,
        "reason": option.reason,
        "active": active,
    }


def _reset_client_cache() -> None:
    global _vlm_client, _vlm_client_option_id
    _vlm_client = None
    _vlm_client_option_id = None


def _resolve_active_option_locked(options: dict[str, VlmOption]) -> VlmOption:
    global _active_vlm_option_id
    active = options.get(_active_vlm_option_id)
    if active and active.enabled:
        return active

    for option in options.values():
        if option.enabled:
            _active_vlm_option_id = option.id
            _reset_client_cache()
            return option

    fallback = options["local"]
    _active_vlm_option_id = fallback.id
    return fallback


def list_vlm_options() -> list[dict]:
    with _vlm_state_lock:
        options = _registered_options()
        active_option = _resolve_active_option_locked({option.id: option for option in options})
        active_id = active_option.id
        return [
            _option_to_dict(option, active=option.id == active_id)
            for option in options
        ]


def get_active_vlm_option() -> VlmOption:
    with _vlm_state_lock:
        return _resolve_active_option_locked(_registered_options_by_id())


def get_vlm_selection_state() -> dict:
    with _vlm_state_lock:
        options = _registered_options()
        active_option = _resolve_active_option_locked({option.id: option for option in options})
        active_id = active_option.id
        return {
            "active_option": _option_to_dict(active_option, active=True),
            "options": [
                _option_to_dict(option, active=option.id == active_id)
                for option in options
            ],
        }


def set_active_vlm_option(option_id: str) -> dict:
    global _active_vlm_option_id

    normalized_id = str(option_id or "").strip().lower()
    options = _registered_options_by_id()
    option = options.get(normalized_id)
    if option is None:
        raise ValueError(f"Unknown VLM option: {option_id}")
    if not option.enabled:
        raise ValueError(option.reason or f"VLM option '{option.label}' is not available.")

    with _vlm_state_lock:
        if _active_vlm_option_id != option.id:
            _active_vlm_option_id = option.id
            _reset_client_cache()

    return get_vlm_selection_state()


def get_vlm_model() -> str:
    """Return the active model name."""
    return get_active_vlm_option().model


def get_vlm_extra_body() -> dict | None:
    """Return the extra_body to merge into the chat request for the active option.

    For reasoning-style models (Qwen3.x on vLLM), the active option's `thinking` field is
    translated into chat_template_kwargs.enable_thinking so the caller can force the
    reasoning block on or off. Returns None when no override is needed (server default).
    """
    thinking = get_active_vlm_option().thinking
    if thinking is None:
        return None
    return {"chat_template_kwargs": {"enable_thinking": bool(thinking)}}


def get_vlm_source() -> str:
    """Return the active backend source."""
    return get_active_vlm_option().source


def is_kimi_backend() -> bool:
    """Return True when the active backend is Kimi."""
    return get_active_vlm_option().source == "kimi"


def is_gemini_backend() -> bool:
    """Return True when the active backend is Gemini."""
    return get_active_vlm_option().source == "gemini"


def active_backend_supports_required_tool_choice() -> bool:
    """Return whether the active backend supports tool_choice='required'."""
    return get_active_vlm_option().supports_required_tool_choice


def describe_active_vlm() -> dict:
    """Return a stable description of the currently selected VLM backend."""
    option = get_active_vlm_option()
    return {
        "id": option.id,
        "label": option.label,
        "source": option.source,
        "model": option.model,
        "enabled": option.enabled,
        "reason": option.reason,
    }


def get_vlm_client() -> OpenAI:
    """Return a cached OpenAI client pointing to the active backend."""
    global _vlm_client, _vlm_client_option_id

    option = get_active_vlm_option()
    with _vlm_state_lock:
        if _vlm_client is not None and _vlm_client_option_id == option.id:
            return _vlm_client

    if option.source == "local":
        last_exc: Exception | None = None
        for candidate in option.candidate_urls:
            client = OpenAI(base_url=candidate, api_key="not-needed", timeout=60.0)
            try:
                client.models.list()
                with _vlm_state_lock:
                    _vlm_client = client
                    _vlm_client_option_id = option.id
                return client
            except Exception as exc:
                last_exc = exc

        if last_exc is not None:
            raise RuntimeError(
                f"Could not connect to any local VLM backend. Tried: {', '.join(option.candidate_urls)}. "
                f"Last error: {last_exc}"
            )
        raise RuntimeError("No local VLM backend is configured.")

    if not option.api_key:
        env_name = "GEMINI_API_KEY" if option.source == "gemini" else "KIMI_API_KEY"
        raise RuntimeError(
            f"{option.label} is selected but {env_name} is not set. "
            f"Please set the {env_name} environment variable."
        )

    client = OpenAI(base_url=option.base_url, api_key=option.api_key, timeout=60.0)
    with _vlm_state_lock:
        _vlm_client = client
        _vlm_client_option_id = option.id
    return client
