from datetime import datetime
from types import SimpleNamespace
from typing import Any, Callable

import inspect

import pytest

from src.chat.replyer import maisaka_generator as replyer_module
from src.common.data_models.message_component_data_model import MessageSequence, TextComponent
from src.common.data_models.reply_generation_data_models import GenerationMetrics, LLMCompletionResult, ReplyGenerationResult
from src.core.tooling import ToolInvocation
from src.maisaka.builtin_tool import reply as reply_tool_module
from src.maisaka.builtin_tool.context import BuiltinToolRuntimeContext
from src.maisaka.chat_loop_service import register_maisaka_hook_specs
from src.maisaka.context.messages import SessionBackedMessage, ToolResultMessage
from src.plugin_runtime.host.hook_spec_registry import HookSpecRegistry


async def _call_message_factory(message_factory: Callable[..., Any], client: object) -> list[Any]:
    result = message_factory(client)
    if inspect.isawaitable(result):
        return await result
    return result


class _FakeLLMResult:
    response = "测试回复"
    reasoning = "先理解上下文，再给出自然回复。"
    model_name = "fake-model"
    tool_calls: list[Any] = []
    prompt_tokens = 12
    completion_tokens = 7
    total_tokens = 19


class _FakeLegacyLLMServiceClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args
        del kwargs

    async def generate_response_with_messages(
        self,
        *,
        message_factory: Callable[[object], list[Any]],
        options: Any = None,
    ) -> _FakeLLMResult:
        del options
        assert await _call_message_factory(message_factory, object())
        return _FakeLLMResult()


class _FakeReplyerHookManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def invoke_hook(self, hook_name: str, **kwargs: Any) -> SimpleNamespace:
        self.calls.append((hook_name, dict(kwargs)))
        if hook_name == "maisaka.replyer.before_request":
            modified_kwargs = dict(kwargs)
            reply_tool_args = dict(modified_kwargs.get("reply_tool_args") or {})
            reply_tool_args["hook_added"] = "yes"
            modified_kwargs["reply_tool_args"] = reply_tool_args
            return SimpleNamespace(kwargs=modified_kwargs, aborted=False)
        return SimpleNamespace(kwargs=dict(kwargs), aborted=False)


def _build_reply_target_history_message() -> SessionBackedMessage:
    return SessionBackedMessage(
        raw_message=MessageSequence([TextComponent("[测试用户] 问一下长记忆")]),
        visible_text="[测试用户] 问一下长记忆",
        timestamp=datetime.now(),
        message_id="msg-1",
        source_kind="user",
    )


def _build_reply_tool_ctx(chat_history: list[Any]) -> BuiltinToolRuntimeContext:
    target_message = SimpleNamespace(
        message_id="msg-1",
        message_info=SimpleNamespace(
            user_info=SimpleNamespace(
                user_cardname="测试用户",
                user_nickname="测试用户",
                user_id="user-1",
            )
        ),
    )
    runtime = SimpleNamespace(
        find_source_message_by_id=lambda message_id: target_message if message_id == "msg-1" else None,
        log_prefix="[test]",
        chat_stream=SimpleNamespace(platform=reply_tool_module.CLI_PLATFORM_NAME),
        session_id="session-1",
        _chat_history=chat_history,
        _clear_force_continue_until_reply=lambda: None,
        _is_focus_mode_active_for_current_chat=lambda: False,
        _record_reply_sent=lambda: None,
        run_sub_agent=None,
    )
    engine = SimpleNamespace(_get_runtime_manager=lambda: None)
    return BuiltinToolRuntimeContext(engine=engine, runtime=runtime)


def test_reply_tool_schema_requires_reference_info() -> None:
    tool_spec = reply_tool_module.get_tool_spec()
    parameters_schema = tool_spec.parameters_schema
    assert parameters_schema is not None
    properties = parameters_schema["properties"]

    assert properties["reference_info"]["type"] == "string"
    assert "无" in properties["reference_info"]["description"]
    assert parameters_schema["required"] == ["msg_id", "reference_info"]


def test_replyer_hook_specs_include_reference_info() -> None:
    registry = HookSpecRegistry()
    register_maisaka_hook_specs(registry)

    for hook_name in (
        "maisaka.replyer.before_request",
        "maisaka.replyer.before_model_request",
        "maisaka.replyer.after_response",
    ):
        hook_spec = registry.get_hook_spec(hook_name)
        assert hook_spec is not None
        parameters_schema = hook_spec.parameters_schema
        assert parameters_schema["properties"]["reference_info"]["type"] == "string"
        assert "reference_info" in parameters_schema["required"]


@pytest.mark.asyncio
async def test_replyer_hooks_receive_reference_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(replyer_module, "LLMServiceClient", _FakeLegacyLLMServiceClient)
    monkeypatch.setattr(replyer_module, "load_prompt", lambda *args, **kwargs: "reply prompt")

    fake_hook_manager = _FakeReplyerHookManager()
    generator = replyer_module.MaisakaReplyGenerator(
        chat_stream=None,
        request_type="test_reply_reference_info",
        enable_visual_message=False,
    )
    monkeypatch.setattr(generator, "_get_runtime_manager", lambda: fake_hook_manager)

    success, _ = await generator.generate_reply_with_context(
        stream_id="session-reply-reference-info",
        chat_history=[],
        reply_reason="测试原因",
        reference_info="测试参考",
        reply_tool_args={"route": "fast"},
    )

    assert success is True
    before_call = fake_hook_manager.calls[0]
    before_model_call = fake_hook_manager.calls[1]
    after_call = fake_hook_manager.calls[2]
    assert before_call[0] == "maisaka.replyer.before_request"
    assert before_call[1]["reference_info"] == "测试参考"
    assert before_call[1]["reply_tool_args"] == {"route": "fast"}
    assert before_model_call[0] == "maisaka.replyer.before_model_request"
    assert before_model_call[1]["reference_info"] == "测试参考"
    assert before_model_call[1]["reply_tool_args"] == {"route": "fast", "hook_added": "yes"}
    assert after_call[0] == "maisaka.replyer.after_response"
    assert after_call[1]["reference_info"] == "测试参考"
    assert after_call[1]["reply_tool_args"] == {"route": "fast", "hook_added": "yes"}


def test_replyer_uses_reply_guide_when_reply_reason_empty() -> None:
    generator = replyer_module.MaisakaReplyGenerator(
        chat_stream=None,
        request_type="test_reply_guide",
        enable_visual_message=False,
    )

    final_user_message = generator._build_final_user_message(
        reply_message=None,
        reply_reason="",
        reference_info="测试参考",
        reply_tool_args={"reply_guide": "只回应日期纠正，不要展开。"},
    )

    assert "【回复指引(仅供参考)】\n只回应日期纠正，不要展开。" in final_user_message
    assert "【参考信息】\n测试参考" in final_user_message


def test_replyer_ignores_reply_guide_when_reply_reason_present() -> None:
    generator = replyer_module.MaisakaReplyGenerator(
        chat_stream=None,
        request_type="test_reply_guide",
        enable_visual_message=False,
    )

    final_user_message = generator._build_final_user_message(
        reply_message=None,
        reply_reason="已有 planner 推理",
        reply_tool_args={"reply_guide": "这段不应进入主 replyer prompt"},
    )

    assert "【最新推理】\n已有 planner 推理" in final_user_message
    assert "回复指引" not in final_user_message
    assert "这段不应进入主 replyer prompt" not in final_user_message


@pytest.mark.asyncio
async def test_reply_tool_passes_reference_info_to_replyer(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    fake_reply_result = ReplyGenerationResult(
        success=True,
        completion=LLMCompletionResult(response_text="测试回复"),
        metrics=GenerationMetrics(overall_ms=11.5),
        monitor_detail={},
    )

    class _FakeReplyer:
        async def generate_reply_with_context(self, **kwargs: Any) -> tuple[bool, ReplyGenerationResult]:
            captured.update(kwargs)
            return True, fake_reply_result

    monkeypatch.setattr(reply_tool_module.replyer_manager, "get_replyer", lambda **kwargs: _FakeReplyer())
    monkeypatch.setattr(reply_tool_module, "render_cli_message", lambda text: text)

    tool_ctx = _build_reply_tool_ctx([])
    invocation = ToolInvocation(
        tool_name="reply",
        arguments={"msg_id": "msg-1", "reference_info": "已有参考"},
        reasoning="测试推理",
    )

    result = await reply_tool_module.handle_tool(tool_ctx, invocation)

    assert result.success is True
    assert captured["reply_reason"] == "测试推理"
    assert captured["reference_info"] == "已有参考"
    assert "reference_info" not in captured["reply_tool_args"]


@pytest.mark.asyncio
async def test_reply_tool_merges_memory_reference_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    fake_reply_result = ReplyGenerationResult(
        success=True,
        completion=LLMCompletionResult(response_text="测试回复"),
        metrics=GenerationMetrics(overall_ms=11.5),
        monitor_detail={},
    )

    class _FakeReplyer:
        async def generate_reply_with_context(self, **kwargs: Any) -> tuple[bool, ReplyGenerationResult]:
            captured.update(kwargs)
            return True, fake_reply_result

    monkeypatch.setattr(reply_tool_module.replyer_manager, "get_replyer", lambda **kwargs: _FakeReplyer())
    monkeypatch.setattr(reply_tool_module, "render_cli_message", lambda text: text)

    tool_ctx = _build_reply_tool_ctx(
        [
            ToolResultMessage(
                content="旧检索完成",
                timestamp=datetime.now(),
                tool_call_id="tool-call-old",
                tool_name="query_memory",
                metadata={"replyer_memory_reference": "旧记忆不应合并"},
            ),
            _build_reply_target_history_message(),
            ToolResultMessage(
                content="失败检索",
                timestamp=datetime.now(),
                tool_call_id="tool-call-failed",
                tool_name="query_memory",
                success=False,
                metadata={"replyer_memory_reference": "失败记忆不应合并"},
            ),
            ToolResultMessage(
                content="其他工具",
                timestamp=datetime.now(),
                tool_call_id="tool-call-other",
                tool_name="other_tool",
                metadata={"replyer_memory_reference": "其他工具记忆不应合并"},
            ),
            ToolResultMessage(
                content="检索完成",
                timestamp=datetime.now(),
                tool_call_id="tool-call-1",
                tool_name="query_memory",
                metadata={"replyer_memory_reference": "【长期记忆检索结果-内部参考】\n1. 测试记忆"},
            ),
        ]
    )
    invocation = ToolInvocation(
        tool_name="reply",
        arguments={"msg_id": "msg-1", "reference_info": "已有参考"},
    )

    result = await reply_tool_module.handle_tool(tool_ctx, invocation)

    assert result.success is True
    assert "已有参考" in captured["reference_info"]
    assert "长期记忆检索结果-内部参考" in captured["reference_info"]
    assert "旧记忆不应合并" not in captured["reference_info"]
    assert "失败记忆不应合并" not in captured["reference_info"]
    assert "其他工具记忆不应合并" not in captured["reference_info"]
    assert "reference_info" not in captured["reply_tool_args"]


def test_replyer_prompt_keeps_reason_and_reference_info() -> None:
    generator = replyer_module.MaisakaReplyGenerator(
        chat_stream=None,
        request_type="test_reference_info_prompt",
        enable_visual_message=False,
    )

    final_user_message = generator._build_final_user_message(
        reply_message=None,
        reply_reason="测试推理",
        reference_info="测试参考",
    )

    assert "【最新推理】\n测试推理" in final_user_message
    assert "【参考信息】\n测试参考" in final_user_message
