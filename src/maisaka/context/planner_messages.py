"""Maisaka 规划器消息构造工具。"""

from copy import copy
from datetime import datetime
from html import escape
from typing import Dict, Iterable, Optional, Sequence

import asyncio

from src.chat.message_receive.chat_manager import chat_manager
from src.chat.message_receive.message import SessionMessage
from src.common.data_models.message_component_data_model import (
    ImageComponent,
    MessageSequence,
    ReplyComponent,
    StandardMessageComponents,
)
from src.common.logger import get_logger
from src.common.message_repository import find_messages

from .message_id_alias import to_display_message_id
from .message_adapter import build_visible_text_from_sequence

logger = get_logger("maisaka_planner_messages")


def build_planner_prefix(
    *,
    timestamp: datetime,
    user_name: str,
    group_card: str = "",
    message_id: Optional[str] = None,
    chat_id: Optional[str] = None,
    quote_ids: Optional[Sequence[str]] = None,
    include_message_id: bool = True,
    include_chat_id: bool = False,
    is_self_message: bool = False,
) -> str:
    """构造 Maisaka 规划器使用的统一消息前缀。

    Args:
        timestamp: 消息时间。
        user_name: 展示给规划器的用户名。
        group_card: 群昵称。
        message_id: 消息 ID。
        chat_id: 聊天流 ID。
        quote_ids: 被引用消息 ID 列表。
        include_message_id: 是否输出 `msg_id` 段。
        include_chat_id: 是否输出 `chat_id` 段。
        is_self_message: 是否显式标注这条消息是 bot 自己发送的。

    Returns:
        str: 拼接完成的规划器前缀。
    """

    message_attrs: list[str] = []
    if include_message_id:
        message_attrs.append(f'msg_id="{escape(to_display_message_id(message_id), quote=True)}"')

    if include_chat_id:
        normalized_chat_id = str(chat_id or "").strip()
        if normalized_chat_id:
            message_attrs.append(f'session_id="{escape(normalized_chat_id, quote=True)}"')
            chat_name = chat_manager.get_session_name(normalized_chat_id)
            if chat_name:
                message_attrs.append(f'chat_name="{escape(chat_name, quote=True)}"')

    normalized_quote = _format_quote_ids(quote_ids)
    if normalized_quote:
        message_attrs.append(f'quote="{escape(normalized_quote, quote=True)}"')

    message_attrs.extend(
        [
            f'time="{escape(timestamp.strftime("%H:%M:%S"), quote=True)}"',
            f'user="{escape(user_name, quote=True)}"',
        ]
    )

    normalized_group_card = group_card.strip()
    if normalized_group_card:
        message_attrs.append(f'group_card="{escape(normalized_group_card, quote=True)}"')
    if is_self_message:
        message_attrs.append('is_self_message="true"')
    return f"<message {' '.join(message_attrs)}>\n"


def _format_quote_ids(quote_ids: Optional[Sequence[str]]) -> str:
    """将引用消息 ID 列表格式化为 XML 属性值。"""

    if not quote_ids:
        return ""

    normalized_ids: list[str] = []
    seen: set[str] = set()
    for raw_quote_id in quote_ids:
        quote_id = to_display_message_id(raw_quote_id)
        if not quote_id or quote_id in seen:
            continue
        seen.add(quote_id)
        normalized_ids.append(quote_id)
    return ",".join(normalized_ids)


def extract_quote_ids_from_message_sequence(message_sequence: MessageSequence) -> list[str]:
    """从消息片段中提取引用目标 ID，供 prompt 元信息使用。"""

    quote_ids: list[str] = []
    seen: set[str] = set()
    for component in message_sequence.components:
        if not isinstance(component, ReplyComponent):
            continue

        quote_id = component.target_message_id.strip()
        if not quote_id or quote_id in seen:
            continue
        seen.add(quote_id)
        quote_ids.append(quote_id)
    return quote_ids


def render_planner_quote_text(components: Iterable[StandardMessageComponents]) -> str:
    """将已解析的引用原文放在消息正文前，使用与 quote 属性一致的消息编号。"""
    blocks: Dict[str, str] = {}
    for component in components:
        if not isinstance(component, ReplyComponent):
            continue

        quote_id = escape(to_display_message_id(component.target_message_id), quote=True)
        content = (component.target_message_content or "").strip()
        if not quote_id or not content or quote_id in blocks:
            continue

        sender = (
            component.target_message_sender_cardname
            or component.target_message_sender_nickname
            or component.target_message_sender_id
        )
        sender_attr = f' user="{escape(sender, quote=True)}"' if sender else ""
        blocks[quote_id] = f'<quote msg_id="{quote_id}"{sender_attr}>\n{escape(content, quote=False)}\n</quote>\n'
    return "".join(blocks.values())


def _render_quoted_image_message(source: SessionMessage) -> str:
    """渲染引用消息里的图片和随图文字，使用已缓存的图片描述。"""
    components = [component for component in source.raw_message.components if not isinstance(component, ReplyComponent)]
    images = [component for component in components if isinstance(component, ImageComponent)]
    if not images:
        return ""

    from src.chat.image_system.image_manager import image_manager

    placeholders = {"", "[image]", "[图片，识别中.....]"}
    for image in images:
        if image.binary_hash and image.content.strip() in placeholders:
            description = image_manager.get_cached_image_description(image.binary_hash)
            if description:
                image.content = f"[图片：{description}]"

    if all(image.content.strip() in placeholders for image in images):
        return ""
    return build_visible_text_from_sequence(MessageSequence(components)).strip()


def _resolve_planner_quote(
    message: SessionMessage,
    target_id: str,
    quote: ReplyComponent,
) -> ReplyComponent:
    """读取同会话原消息，保留现成文本引用并补全图片或仅含编号的引用。"""
    matches = find_messages(session_id=message.session_id, platform=message.platform, message_id=target_id, limit=2)
    if len(matches) > 1:
        logger.warning(f"引用消息 ID 在当前聊天流中不唯一，session_id={message.session_id}，message_id={target_id}")
        return quote
    if not matches:
        return quote

    source = matches[0]
    image_quote = _render_quoted_image_message(source)
    if not image_quote and (quote.target_message_content or "").strip():
        return quote

    sender = source.message_info.user_info
    resolved = copy(quote)
    resolved.target_message_content = image_quote or source.processed_plain_text
    resolved.target_message_sender_id = sender.user_id
    resolved.target_message_sender_nickname = sender.user_nickname
    resolved.target_message_sender_cardname = sender.user_cardname
    return resolved


async def resolve_planner_quote_text(message: SessionMessage) -> str:
    """为 Planner 构造消息引用文本，并从同会话原消息补全图片描述。"""
    quotes: Dict[str, ReplyComponent] = {}
    for component in message.raw_message.components:
        if not isinstance(component, ReplyComponent):
            continue
        target_id = component.target_message_id.strip()
        if not target_id:
            continue
        existing = quotes.get(target_id)
        if existing is None or not (existing.target_message_content or "").strip():
            quotes[target_id] = component

    if quotes and message.session_id:
        for target_id, component in quotes.items():
            quotes[target_id] = await asyncio.to_thread(_resolve_planner_quote, message, target_id, component)

    return render_planner_quote_text(quotes.values())


def build_planner_user_prefix_from_session_message(
    message: SessionMessage,
    *,
    include_message_id: bool = True,
    include_chat_id: bool = False,
    is_self_message: bool = False,
) -> str:
    """根据真实会话消息构造规划器前缀。

    Args:
        message: 原始会话消息。
        include_message_id: 是否输出 `msg_id` 段。
        include_chat_id: 是否输出 `chat_id` 段。
        is_self_message: 是否显式标注这条消息是 bot 自己发送的。

    Returns:
        str: 规划器前缀字符串。
    """

    user_info = message.message_info.user_info
    user_name = user_info.user_nickname or user_info.user_id
    return build_planner_prefix(
        timestamp=message.timestamp,
        user_name=user_name,
        group_card=user_info.user_cardname or "",
        message_id=message.message_id,
        chat_id=message.session_id,
        quote_ids=extract_quote_ids_from_message_sequence(message.raw_message),
        include_message_id=include_message_id and not message.is_notify and bool(message.message_id),
        include_chat_id=include_chat_id,
        is_self_message=is_self_message,
    )
