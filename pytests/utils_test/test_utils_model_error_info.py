from src.llm_models.utils_model import LLMOrchestrator


def test_original_error_info_omits_cause_message_preview() -> None:
    error = RuntimeError("wrapped")
    cause = ValueError("Malformed response SECRET raw upstream preview")
    error.__cause__ = cause

    log_info = LLMOrchestrator._get_original_error_info(error)

    assert "底层异常类型: ValueError" in log_info
    assert "底层异常信息" not in log_info
    assert "Malformed response" not in log_info
    assert "SECRET" not in log_info
