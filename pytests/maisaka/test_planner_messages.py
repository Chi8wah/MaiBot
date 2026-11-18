from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from sqlmodel import Session, create_engine
import pytest

from src.chat.message_receive.message import SessionMessage
from src.common.data_models.mai_message_data_model import MessageInfo, UserInfo
from src.common.data_models.message_component_data_model import (
    ForwardComponent,
    ForwardNodeComponent,
    MessageSequence,
    ReplyComponent,
    TextComponent,
)
from src.common.database.database_model import Messages
from src.llm_models.payload_content.context_item import get_item_text
from src.maisaka.chat_loop_service import MaisakaChatLoopService
from src.maisaka.context.message_id_alias import to_display_message_id
from src.maisaka.context.planner_messages import build_planner_prefix
from src.maisaka.reasoning_engine import MaisakaReasoningEngine
from src.maisaka.runtime import MaisakaHeartFlowChatting


def test_build_planner_prefix_marks_self_message() -> None:
    prefix = build_planner_prefix(
        timestamp=datetime(2026, 8, 20, 1, 9, 30),
        user_name="麦麦",
        message_id="self-message",
        is_self_message=True,
    )

    assert 'is_self_message="true"' in prefix


def test_build_planner_prefix_does_not_mark_user_message() -> None:
    prefix = build_planner_prefix(
        timestamp=datetime(2026, 8, 20, 1, 9, 30),
        user_name="用户",
        message_id="user-message",
    )

    assert 'is_self_message="true"' not in prefix


def _make_message(message_id: str, user_name: str, raw_message: MessageSequence) -> SessionMessage:
    message = SessionMessage(message_id=message_id, timestamp=datetime(2026, 1, 1, 12), platform="qq")
    message.message_info = MessageInfo(user_info=UserInfo(user_id=user_name, user_nickname=user_name))
    message.session_id = "group-session"
    message.raw_message = raw_message
    return message


async def _planner_message_text(message: SessionMessage) -> str:
    engine = MaisakaReasoningEngine(SimpleNamespace(_is_focus_mode_active_for_current_chat=lambda: False))
    history_message = await engine._build_history_message(message)
    assert history_message is not None
    service = MaisakaChatLoopService(chat_system_prompt="system")
    items, history_count = service._build_request_messages([history_message], enable_visual_message=False)
    assert history_count == 1
    return get_item_text(items[1])


@pytest.mark.asyncio
@pytest.mark.parametrize("include_forward", [False, True], ids=["text", "with-forward"])
async def test_planner_includes_quote_text_when_original_is_outside_window(
    monkeypatch: pytest.MonkeyPatch,
    include_forward: bool,
) -> None:
    lookup = AsyncMock()
    monkeypatch.setattr(SessionMessage, "process_reply_component", lookup)
    quote = ReplyComponent(
        "1001",
        target_message_content="访谈会在周五晚上直播",
        target_message_sender_nickname="主持人",
    )
    raw_message = MessageSequence([quote, TextComponent("在哪个平台播出？")])
    if include_forward:
        raw_message.components.append(
            ForwardNodeComponent(
                [
                    ForwardComponent(
                        user_nickname="发送者",
                        message_id="forward-node",
                        content=[TextComponent("转发里的正文")],
                    )
                ]
            )
        )
    message = _make_message("1002", "观众", raw_message)

    text = await _planner_message_text(message)

    assert 'quote="1001"' in text.splitlines()[0]
    assert '<quote msg_id="1001" user="主持人">\n访谈会在周五晚上直播\n</quote>\n在哪个平台播出？' in text
    if include_forward:
        assert "转发里的正文" in text
    lookup.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("original_session", ["group-session", "another-session"])
async def test_planner_resolves_legacy_quote_in_its_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    original_session: str,
) -> None:
    db = create_engine(f"sqlite:///{tmp_path / 'messages.db'}")
    Messages.__table__.create(db)
    monkeypatch.setattr("src.chat.message_receive.message.get_db_session", lambda **_: Session(db))
    original = _make_message("older-platform-message-id", "发送者", MessageSequence([TextComponent("早期原文 <测试>")]))
    original.session_id = original_session
    original.processed_plain_text = "早期原文 <测试>"
    with Session(db) as session:
        session.add(original.to_db_instance())
        session.commit()
    quote = ReplyComponent(original.message_id)
    message = _make_message("later-id", "用户", MessageSequence([quote, TextComponent("回答这条引用")]))

    text = await _planner_message_text(message)

    displayed_id = to_display_message_id(original.message_id)
    assert f'quote="{displayed_id}"' in text.splitlines()[0]
    assert text.endswith("回答这条引用")
    if original_session == message.session_id:
        assert f'<quote msg_id="{displayed_id}" user="发送者">' in text
        assert "早期原文 &lt;测试&gt;" in text
    else:
        assert "<quote " not in text
    assert quote.target_message_content is None


def test_sent_message_keeps_quote_text_in_planner_history() -> None:
    quote = ReplyComponent("user-msg", target_message_content="用户的问题", target_message_sender_nickname="用户")
    message = _make_message("sent-msg", "麦麦", MessageSequence([quote, TextComponent("机器人的回答")]))
    runtime = SimpleNamespace(
        _chat_history=[],
        _focus_group=None,
        _quote_counter=Mock(),
        _is_focus_mode_active_for_current_chat=lambda: False,
        _schedule_sent_image_recognition=Mock(),
        _emit_monitor_message_sent=Mock(),
    )

    assert MaisakaHeartFlowChatting.append_sent_message_to_chat_history(runtime, message)
    item = runtime._chat_history[0].to_context_item(enable_visual_message=False)
    assert item is not None
    text = get_item_text(item)
    assert 'is_self_message="true"' in text.splitlines()[0]
    assert '<quote msg_id="user-msg" user="用户">\n用户的问题\n</quote>\n机器人的回答' in text
