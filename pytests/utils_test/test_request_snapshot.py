from importlib import util
from pathlib import Path

import json

from src.config.model_configs import APIProvider, ModelInfo
from src.llm_models.model_client.base_client import ResponseRequest
from src.llm_models.payload_content.message import MessageBuilder, RoleType
from src.llm_models.payload_content.resp_format import RespFormat, RespFormatType
from src.llm_models.payload_content.tool_option import ToolCall, ToolOption
from src.llm_models.request_snapshot import (
    attach_request_snapshot,
    deserialize_messages_snapshot,
    format_request_snapshot_log_info,
    save_failed_request_snapshot,
    serialize_messages_snapshot,
    serialize_response_request_snapshot,
)
from src.llm_models import request_snapshot
from src.maisaka import chat_loop_service as chat_loop_service_module
from src.maisaka.chat_loop_service import MaisakaChatLoopService


def _build_api_provider() -> APIProvider:
    return APIProvider(
        api_key="secret-token",
        base_url="https://example.com/v1",
        name="test-provider",
    )


def _build_model_info() -> ModelInfo:
    return ModelInfo(
        api_provider="test-provider",
        model_identifier="demo-model",
        name="demo-model",
    )


def _build_response_request() -> ResponseRequest:
    tool_call = ToolCall(
        args={"query": "MaiBot"},
        call_id="call_1",
        func_name="search_web",
        extra_content={"google": {"thought_signature": "c2lnbmF0dXJl"}},
    )
    message_list = [
        MessageBuilder().set_role(RoleType.User).add_text_content("你好").add_image_content("png", "ZmFrZQ==").build(),
        MessageBuilder().set_role(RoleType.Assistant).set_tool_calls([tool_call]).build(),
        MessageBuilder()
        .set_role(RoleType.Tool)
        .set_tool_call_id("call_1")
        .set_tool_name("search_web")
        .add_text_content('{"ok": true}')
        .build(),
    ]
    return ResponseRequest(
        extra_params={"trace_id": "trace-123"},
        max_tokens=256,
        message_list=message_list,
        model_info=_build_model_info(),
        response_format=RespFormat(RespFormatType.JSON_OBJ),
        temperature=0.2,
        tool_options=[ToolOption(name="search_web", description="搜索网页")],
    )


def test_message_snapshot_roundtrip_preserves_tool_messages() -> None:
    request = _build_response_request()

    snapshot_messages = serialize_messages_snapshot(request.message_list)
    restored_messages = deserialize_messages_snapshot(snapshot_messages)

    assert len(restored_messages) == 3
    assert restored_messages[0].role == RoleType.User
    assert restored_messages[0].get_text_content() == "你好"
    assert restored_messages[0].parts[1].image_format == "png"
    assert restored_messages[1].role == RoleType.Assistant
    assert restored_messages[1].tool_calls is not None
    assert restored_messages[1].tool_calls[0].func_name == "search_web"
    assert restored_messages[1].tool_calls[0].args == {"query": "MaiBot"}
    assert restored_messages[1].tool_calls[0].extra_content == {"google": {"thought_signature": "c2lnbmF0dXJl"}}
    assert restored_messages[2].role == RoleType.Tool
    assert restored_messages[2].tool_call_id == "call_1"
    assert restored_messages[2].tool_name == "search_web"


def test_failed_request_snapshot_contains_replay_entry(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(request_snapshot, "LLM_REQUEST_LOG_DIR", tmp_path)

    request = _build_response_request()
    provider = _build_api_provider()
    snapshot_path = save_failed_request_snapshot(
        api_provider=provider,
        client_type="openai",
        error=RuntimeError("boom"),
        internal_request=serialize_response_request_snapshot(request),
        model_info=request.model_info,
        operation="chat.completions.create",
        provider_request={
            "request_kwargs": {
                "extra_body": {"safe": "ok", "nested": {"token": "provider-token"}},
                "extra_headers": {"Authorization": "Bearer provider-token"},
                "extra_query": {"api_key": "provider-token"},
                "model": request.model_info.model_identifier,
            }
        },
    )

    assert snapshot_path is not None
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))

    assert payload["internal_request"]["request_kind"] == "response"
    assert payload["api_provider"]["name"] == "test-provider"
    assert payload["replay"]["file_uri"] == snapshot_path.as_uri()
    assert str(snapshot_path) in payload["replay"]["command"]
    assert "secret-token" not in snapshot_path.read_text(encoding="utf-8")
    request_kwargs = payload["provider_request"]["request_kwargs"]
    assert "extra_headers" not in request_kwargs
    assert "extra_query" not in request_kwargs
    assert request_kwargs["extra_body"]["safe"] == "ok"
    assert request_kwargs["extra_body"]["nested"]["token"] == "<redacted>"
    assert "provider-token" not in snapshot_path.read_text(encoding="utf-8")


def test_format_request_snapshot_log_info_includes_help_text_and_replay_command(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(request_snapshot, "LLM_REQUEST_LOG_DIR", tmp_path)

    request = _build_response_request()
    snapshot_path = save_failed_request_snapshot(
        api_provider=_build_api_provider(),
        client_type="openai",
        error=ValueError("invalid"),
        internal_request=serialize_response_request_snapshot(request),
        model_info=request.model_info,
        operation="chat.completions.create",
        provider_request={"request_kwargs": {"messages": []}},
    )

    assert snapshot_path is not None
    exc = RuntimeError("wrapped")
    attach_request_snapshot(exc, snapshot_path)

    log_info = format_request_snapshot_log_info(exc)
    assert "调用完整信息（如果需要求助，请发送该文本）:" in log_info
    assert str(snapshot_path) in log_info
    assert "请求快照链接" not in log_info
    assert snapshot_path.as_uri() not in log_info
    assert "使用以下命令重新请求: uv run python scripts/replay_llm_request.py" in log_info


def test_debug_planner_cache_keeps_replay_snapshot(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(chat_loop_service_module, "DEBUG_PLANNER_CACHE_DIR", tmp_path)
    monkeypatch.setattr(chat_loop_service_module.global_config.debug, "record_planner_request", True)

    service = MaisakaChatLoopService(session_id="session/1")
    request_snapshot = {
        "api_provider": {"name": "Demo", "client_type": "openai"},
        "client_type": "openai",
        "messages": [{"role": "user", "content": "must not overwrite diagnostics"}],
        "internal_request": {
            "extra_params": {"top_p": 0.95},
            "max_tokens": 256,
            "message_list": [
                {
                    "role": "system",
                    "parts": [{"type": "text", "text": "pick a tool"}],
                    "tool_calls": [],
                }
            ],
            "model_info": {
                "api_provider": "Demo",
                "extra_params": {},
                "force_stream_mode": False,
                "max_tokens": None,
                "model_identifier": "demo-model-id",
                "name": "demo-model",
                "temperature": None,
                "visual": False,
            },
            "request_kind": "response",
            "response_format": None,
            "temperature": 0.6,
            "tool_options": [
                {
                    "type": "function",
                    "function": {
                        "name": "finish",
                        "description": "finish planning",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        },
        "model_info": {
            "api_provider": "Demo",
            "extra_params": {},
            "force_stream_mode": False,
            "max_tokens": None,
            "model_identifier": "demo-model-id",
            "name": "demo-model",
            "temperature": None,
            "visual": False,
        },
        "operation": "chat.completions.create",
        "request_type": "must_not_overwrite",
        "response_format": None,
        "response_body": {"must_not": "overwrite"},
        "snapshot_version": 1,
        "tool_definitions": [{"name": "must_not_overwrite"}],
    }

    service._save_debug_planner_request_body(
        request_kind="planner",
        model_name="demo-model",
        messages=[],
        tool_definitions=[],
        response_format=None,
        selection_reason="test",
        selected_history_count=0,
        response_body={"raw": "response"},
        final_response_body={"final": "response"},
        provider_request={"request_kwargs": {"max_tokens": "<redacted>"}},
        request_snapshot=request_snapshot,
    )

    snapshot_files = list(tmp_path.glob("*.json"))
    assert len(snapshot_files) == 1
    payload = json.loads(snapshot_files[0].read_text(encoding="utf-8"))
    assert payload["request_kind"] == "planner"
    assert payload["request_type"] == "maisaka.planner"
    assert payload["internal_request"]["request_kind"] == "response"
    assert payload["internal_request"]["max_tokens"] == 256
    assert payload["provider_request"]["request_kwargs"]["max_tokens"] == "<redacted>"
    assert payload["messages"] == []
    assert payload["tool_definitions"] == []
    assert payload["response_body"] == {"raw": "response"}
    assert payload["final_response_body"] == {"final": "response"}
    assert payload["replay"]["command"].endswith(f'"{snapshot_files[0].as_posix()}"')
    assert "file_uri" not in payload["replay"]

    replay_script = Path(__file__).resolve().parents[2] / "scripts" / "replay_llm_request.py"
    spec = util.spec_from_file_location("replay_llm_request", replay_script)
    assert spec is not None and spec.loader is not None
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    request = module._build_response_request(payload["internal_request"])
    assert request.max_tokens == 256
    assert request.model_info.name == "demo-model"
    assert len(request.message_list) == 1
    assert len(request.tool_options or []) == 1
