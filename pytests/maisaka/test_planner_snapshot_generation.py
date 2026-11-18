from datetime import datetime
from types import SimpleNamespace
from typing import Tuple
from unittest.mock import AsyncMock, Mock

import pytest

from src.config.config import ModelConfig, config_manager, global_config
from src.config.model_configs import APIProvider, ModelInfo, ModelTaskConfig, TaskConfig
from src.llm_models import utils_model
from src.llm_models.model_client.base_client import APIResponse, ResponseRequest, UsageRecord
from src.llm_models.payload_content.context_item import build_portable_output_items
from src.maisaka import reply_supplement
from src.maisaka.context.compaction import PlannerRequestSnapshot, build_context_summary
from src.maisaka.context.messages import ReferenceMessage
from src.services import model_task_usage_service


@pytest.fixture
def snapshot_client(monkeypatch: pytest.MonkeyPatch) -> Tuple[PlannerRequestSnapshot, AsyncMock, Mock]:
    provider = APIProvider(name="test-provider", base_url="https://test.example/v1", auth_type="none")
    model = ModelInfo(model_type="llm", name="test-model", model_identifier="test-model", api_provider=provider.name)
    model_config = ModelConfig(
        models=[model],
        api_providers=[provider],
        model_task_config=ModelTaskConfig(planner=TaskConfig(model_list=[model.name])),
    )
    monkeypatch.setattr(config_manager, "model_config", model_config)
    monkeypatch.setattr(global_config.debug, "save_maisaka_reasoning", False)
    monkeypatch.setattr(global_config.debug, "enable_llm_cache_stats", False)
    monkeypatch.setattr(model_task_usage_service, "record_model_task_usage", Mock())

    get_response = AsyncMock()
    record_usage = Mock()
    monkeypatch.setattr(
        utils_model.client_registry,
        "get_client_class_instance_async",
        AsyncMock(return_value=SimpleNamespace(get_response=get_response)),
    )
    monkeypatch.setattr(utils_model.llm_usage_recorder, "record_usage_to_database", record_usage)
    snapshot = PlannerRequestSnapshot(request=ResponseRequest(model_info=model, context_items=[]), history=())
    return snapshot, get_response, record_usage


def _model_response(text: str) -> APIResponse:
    return APIResponse(
        output_items=build_portable_output_items(content=text, logical_turn_id="test-turn"),
        usage=UsageRecord(
            model_name="test-model",
            provider_name="test-provider",
            prompt_tokens=100,
            completion_tokens=12,
            total_tokens=112,
        ),
    )


@pytest.mark.asyncio
async def test_context_compaction_uses_snapshot_response(
    snapshot_client: Tuple[PlannerRequestSnapshot, AsyncMock, Mock],
) -> None:
    snapshot, get_response, record_usage = snapshot_client
    response = _model_response('{"topics": ["周末观赛安排"]}')
    get_response.return_value = response
    old_message = ReferenceMessage(content="讨论了周末观赛安排", timestamp=datetime(2026, 1, 1))

    summary = await build_context_summary(
        [old_message],
        planner_snapshot=snapshot,
        previous_summary=None,
        recent_messages=[],
        session_id="test-session",
    )

    assert summary.content == "当前话题:\n- 周末观赛安排"
    get_response.assert_awaited_once()
    assert record_usage.call_count == 1
    assert record_usage.call_args.kwargs["model_usage"] is response.usage
    assert record_usage.call_args.kwargs["request_type"] == "maisaka.context_compaction"


@pytest.mark.asyncio
async def test_reply_supplement_uses_snapshot_response(
    snapshot_client: Tuple[PlannerRequestSnapshot, AsyncMock, Mock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, get_response, record_usage = snapshot_client
    response = _model_response('{"msg_id": null, "parts": [{"type": "reply"}]}')
    get_response.return_value = response
    monkeypatch.setattr(reply_supplement, "build_reply_extensions_schema", lambda _: None)

    supplement = await reply_supplement.generate_reply_supplement(snapshot, "回复当前话题", "test-session", True)

    assert await supplement.attachments == [None]
    assert supplement.metadata["total_tokens"] == 112
    get_response.assert_awaited_once()
    assert record_usage.call_count == 1
    assert record_usage.call_args.kwargs["model_usage"] is response.usage
    assert record_usage.call_args.kwargs["request_type"] == "maisaka.reply_supplement"
