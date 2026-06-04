# ruff: noqa: E402

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

from src.config.config import config_manager
from src.llm_models.model_client.base_client import AudioTranscriptionRequest, ResponseRequest, client_registry
from src.llm_models.model_client.base_client import EmbeddingRequest
from src.llm_models.request_snapshot import (
    deserialize_messages_snapshot,
    deserialize_model_info_snapshot,
    deserialize_response_format_snapshot,
    deserialize_tool_options_snapshot,
    serialize_api_provider_snapshot,
    serialize_messages_snapshot,
    serialize_model_info_snapshot,
    serialize_tool_options_snapshot,
)
from src.llm_models.payload_content.tool_option import normalize_tool_options
from src.plugin_runtime.hook_payloads import deserialize_prompt_messages
from src.services.service_task_resolver import get_available_models


def _load_snapshot(snapshot_path: Path) -> dict[str, Any]:
    """加载请求快照。"""
    return json.loads(snapshot_path.read_text(encoding="utf-8"))


def _resolve_api_provider(provider_name: str):
    """根据名称解析当前配置中的 API Provider。"""
    model_config = config_manager.get_model_config()
    for api_provider in model_config.api_providers:
        if api_provider.name == provider_name:
            return api_provider
    raise ValueError(f"当前配置中不存在名为 {provider_name!r} 的 API Provider")


def _resolve_model_info(model_name: str):
    """根据模型名称或模型标识符解析当前配置中的模型。"""
    normalized_model_name = str(model_name or "").strip()
    model_config = config_manager.get_model_config()
    for model_info in model_config.models:
        if model_info.name == normalized_model_name:
            return model_info
    for model_info in model_config.models:
        if model_info.model_identifier == normalized_model_name:
            return model_info
    raise ValueError(f"当前配置中不存在名为或标识符为 {normalized_model_name!r} 的模型")


def _resolve_task_max_tokens(request_kind: str) -> int | None:
    """从当前任务配置中解析旧 planner cache 缺失的 max_tokens。"""
    task_name = str(request_kind or "planner").strip() or "planner"
    task_config = get_available_models().get(task_name)
    if task_config is None:
        return None
    return task_config.max_tokens


def _build_snapshot_from_planner_cache(snapshot: dict[str, Any]) -> dict[str, Any]:
    """将旧版 planner debug cache 转换成 replay 脚本原生快照。"""
    raw_provider_request = snapshot.get("provider_request")
    provider_request: dict[str, Any] = raw_provider_request if isinstance(raw_provider_request, dict) else {}
    raw_request_kwargs = provider_request.get("request_kwargs")
    request_kwargs: dict[str, Any] = raw_request_kwargs if isinstance(raw_request_kwargs, dict) else {}
    model_info = _resolve_model_info(str(snapshot.get("model") or request_kwargs.get("model") or ""))
    api_provider = _resolve_api_provider(model_info.api_provider)
    request_kind = str(snapshot.get("request_kind") or "planner")

    raw_max_tokens = request_kwargs.get("max_tokens")
    max_tokens = raw_max_tokens if isinstance(raw_max_tokens, int) else _resolve_task_max_tokens(request_kind)
    response_format = snapshot.get("response_format") if isinstance(snapshot.get("response_format"), dict) else None

    messages = deserialize_prompt_messages(snapshot.get("messages") or [])
    tool_options = normalize_tool_options(snapshot.get("tool_definitions") or [])
    internal_request = {
        "extra_params": dict(request_kwargs.get("extra_body") or {}),
        "max_tokens": max_tokens,
        "message_list": serialize_messages_snapshot(messages),
        "model_info": serialize_model_info_snapshot(model_info),
        "request_kind": "response",
        "response_format": response_format,
        "temperature": request_kwargs.get("temperature"),
        "tool_options": serialize_tool_options_snapshot(tool_options),
    }
    return {
        **snapshot,
        "api_provider": serialize_api_provider_snapshot(api_provider),
        "client_type": api_provider.client_type,
        "internal_request": internal_request,
        "model_info": serialize_model_info_snapshot(model_info),
        "operation": provider_request.get("operation") or "chat.completions.create",
        "snapshot_version": snapshot.get("snapshot_version") or 1,
    }


def _normalize_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """兼容新 replay 快照与旧 planner debug cache。"""
    if isinstance(snapshot.get("internal_request"), dict) and isinstance(snapshot.get("api_provider"), dict):
        return snapshot
    if isinstance(snapshot.get("messages"), list) and isinstance(snapshot.get("tool_definitions"), list):
        return _build_snapshot_from_planner_cache(snapshot)
    return snapshot


def _build_response_request(snapshot: dict[str, Any]) -> ResponseRequest:
    """从快照构建响应请求对象。"""
    return ResponseRequest(
        extra_params=dict(snapshot.get("extra_params") or {}),
        max_tokens=snapshot.get("max_tokens"),
        message_list=deserialize_messages_snapshot(snapshot.get("message_list") or []),
        model_info=deserialize_model_info_snapshot(snapshot.get("model_info") or {}),
        response_format=deserialize_response_format_snapshot(snapshot.get("response_format")),
        temperature=snapshot.get("temperature"),
        tool_options=deserialize_tool_options_snapshot(snapshot.get("tool_options")),
    )


def _build_embedding_request(snapshot: dict[str, Any]) -> EmbeddingRequest:
    """从快照构建嵌入请求对象。"""
    return EmbeddingRequest(
        embedding_input=str(snapshot.get("embedding_input") or ""),
        extra_params=dict(snapshot.get("extra_params") or {}),
        model_info=deserialize_model_info_snapshot(snapshot.get("model_info") or {}),
    )


def _build_audio_request(snapshot: dict[str, Any]) -> AudioTranscriptionRequest:
    """从快照构建音频转写请求对象。"""
    return AudioTranscriptionRequest(
        audio_base64=str(snapshot.get("audio_base64") or ""),
        extra_params=dict(snapshot.get("extra_params") or {}),
        max_tokens=snapshot.get("max_tokens"),
        model_info=deserialize_model_info_snapshot(snapshot.get("model_info") or {}),
    )


async def _replay(snapshot_path: Path) -> int:
    """回放一条失败请求快照。"""
    config_manager.initialize()
    snapshot = _normalize_snapshot(_load_snapshot(snapshot_path))

    internal_request = snapshot.get("internal_request")
    if not isinstance(internal_request, dict):
        raise ValueError("快照缺少 internal_request 字段")

    provider_snapshot = snapshot.get("api_provider")
    if not isinstance(provider_snapshot, dict):
        raise ValueError("快照缺少 api_provider 字段")

    provider_name = str(provider_snapshot.get("name") or "")
    if not provider_name:
        raise ValueError("快照中的 api_provider.name 不能为空")

    api_provider = _resolve_api_provider(provider_name)
    client = client_registry.get_client_class_instance(api_provider, force_new=True)

    request_kind = str(internal_request.get("request_kind") or "").strip()
    if request_kind == "response":
        response = await client.get_response(_build_response_request(internal_request))
    elif request_kind == "embedding":
        response = await client.get_embedding(_build_embedding_request(internal_request))
    elif request_kind == "audio_transcription":
        response = await client.get_audio_transcriptions(_build_audio_request(internal_request))
    else:
        raise ValueError(f"不支持的 request_kind: {request_kind!r}")

    output_payload = {
        "content": response.content,
        "embedding_length": len(response.embedding or []),
        "has_embedding": response.embedding is not None,
        "model_name": response.usage.model_name if response.usage is not None else None,
        "provider_name": response.usage.provider_name if response.usage is not None else None,
        "raw_data_type": type(response.raw_data).__name__ if response.raw_data is not None else None,
        "reasoning_content": response.reasoning_content,
        "tool_calls": [
            {
                "args": tool_call.args,
                "call_id": tool_call.call_id,
                "func_name": tool_call.func_name,
            }
            for tool_call in (response.tool_calls or [])
        ],
        "usage": {
            "completion_tokens": response.usage.completion_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        if response.usage is not None
        else None,
    }
    print(json.dumps(output_payload, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    """脚本入口。"""
    parser = argparse.ArgumentParser(description="回放失败的 LLM 请求快照。")
    parser.add_argument("snapshot_path", help="请求快照 JSON 文件路径")
    args = parser.parse_args()

    snapshot_path = Path(args.snapshot_path).expanduser().resolve()
    if not snapshot_path.exists():
        raise FileNotFoundError(f"快照文件不存在: {snapshot_path}")

    return asyncio.run(_replay(snapshot_path))


if __name__ == "__main__":
    raise SystemExit(main())
