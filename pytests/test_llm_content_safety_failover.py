from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import pytest

from src.config.model_configs import APIProvider, ModelInfo, TaskConfig
from src.llm_models import utils_model
from src.llm_models.exceptions import ContentSafetyRejectionException
from src.llm_models.model_client.base_client import APIResponse, ResponseRequest
from src.llm_models.payload_content.context_item import build_portable_output_items
from src.llm_models.utils_model import LLMOrchestrator, RequestType


class _TextClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    async def get_response(self, request: ResponseRequest) -> APIResponse:
        self.calls += 1
        return APIResponse(
            output_items=build_portable_output_items(content=self.text, logical_turn_id=request.logical_turn_id),
            wire_protocol="chat_completions",
        )


@pytest.mark.asyncio
async def test_high_risk_rejection_switches_model_without_retry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    models = {
        name: ModelInfo(model_type="llm", name=name, model_identifier=name, api_provider=name)
        for name in ("rejected", "next")
    }
    providers = {
        name: APIProvider(name=name, base_url=f"https://{name}.example/v1", auth_type="none", max_retry=3)
        for name in models
    }
    clients = {
        "rejected": _TextClient(" \nTHE REQUEST WAS REJECTED BECAUSE IT WAS CONSIDERED HIGH RISK\n "),
        "next": _TextClient("正常回复"),
    }
    orchestrator = object.__new__(LLMOrchestrator)
    orchestrator.task_name = "replyer"
    orchestrator.request_type = "maisaka.replyer"
    orchestrator.session_id = ""
    orchestrator.model_for_task = TaskConfig(model_list=list(models), selection_strategy="sequential")
    orchestrator.model_usage = {name: (0, 0, 0) for name in models}
    orchestrator._explicit_model_usage = {}

    async def select_model(self: LLMOrchestrator, exclude_models=None, model_name=None):
        name = next(name for name in models if name not in (exclude_models or set()))
        self._adjust_model_usage(name, usage_penalty_delta=1)
        return models[name], providers[name], clients[name]

    snapshots = []
    statuses = []
    succeeded_models = []
    usage_outcomes = []

    @asynccontextmanager
    async def track_usage(*args: str) -> AsyncIterator[None]:
        del args
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            usage_outcomes.append(succeeded)

    monkeypatch.setattr(LLMOrchestrator, "_select_model", select_model)
    monkeypatch.setattr(utils_model, "track_model_task_usage", track_usage)
    monkeypatch.setattr(
        utils_model,
        "save_failed_request_snapshot",
        lambda **kwargs: snapshots.append(kwargs) or tmp_path / "failure.json",
    )
    monkeypatch.setattr(
        utils_model,
        "update_failed_request_attempt",
        lambda error, *, status: statuses.append(status),
    )
    monkeypatch.setattr(
        utils_model,
        "mark_request_succeeded",
        lambda request, response: succeeded_models.append(request.model_info.name),
    )

    result = await orchestrator._execute_request(RequestType.RESPONSE)

    assert result.model_info.name == "next"
    assert result.api_response.content == "正常回复"
    assert clients["rejected"].calls == 1
    assert clients["next"].calls == 1
    assert succeeded_models == ["next"]
    assert usage_outcomes == [False, True]
    assert statuses == ["switching_model"]
    assert len(snapshots) == 1
    assert isinstance(snapshots[0]["error"], ContentSafetyRejectionException)


@pytest.mark.asyncio
async def test_mentioning_high_risk_rejection_in_answer_is_not_filtered(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "上游曾返回 The request was rejected because it was considered high risk，但这只是引用。"

    @asynccontextmanager
    async def track_usage(*args: str) -> AsyncIterator[None]:
        del args
        yield

    monkeypatch.setattr(utils_model, "track_model_task_usage", track_usage)
    model = ModelInfo(model_type="llm", name="test", model_identifier="test", api_provider="test")
    provider = APIProvider(name="test", base_url="https://test.example/v1", auth_type="none")
    client = _TextClient(text)
    request = ResponseRequest(model_info=model, context_items=[])
    orchestrator = object.__new__(LLMOrchestrator)
    orchestrator.task_name = "replyer"
    orchestrator.request_type = "maisaka.replyer"

    result = await orchestrator._attempt_request_on_model(provider, client, request)

    assert result.api_response.content == text
    assert client.calls == 1
