"""Tests for centralized logging configuration, file persistence, run isolation, and masking."""

import asyncio
import io
import logging
from pathlib import Path
import tempfile

from observability.logging import (
    TraceActor,
    format_trace_event,
    get_logger,
    log_trace_event,
    sanitize_log_message,
    scoped_run_context,
    setup_logging,
    shutdown_logging,
)


def test_logging_configuration():
    """Verify logging setup formats messages and respects log level."""
    log_stream = io.StringIO()
    setup_logging(log_level="DEBUG", stream=log_stream)

    logger = get_logger("test_logger")
    logger.debug("Debug test message")
    logger.info("Info test message")

    output = log_stream.getvalue()
    assert "DEBUG" in output
    assert "Debug test message" in output
    assert "INFO" in output
    assert "Info test message" in output
    assert "test_logger:" in output


def test_logging_idempotency():
    """Verify multiple calls to setup_logging do not duplicate handlers."""
    root_logger = logging.getLogger()

    setup_logging(log_level="INFO")
    initial_handler_count = len(root_logger.handlers)

    setup_logging(log_level="INFO")
    assert len(root_logger.handlers) == initial_handler_count


def test_logging_level_filtering():
    """Verify messages below configured level are filtered out."""
    log_stream = io.StringIO()
    setup_logging(log_level="WARNING", stream=log_stream)

    logger = get_logger("filter_test")
    logger.info("Should not appear")
    logger.warning("Should appear")

    output = log_stream.getvalue()
    assert "Should not appear" not in output
    assert "Should appear" in output


def test_application_file_logging():
    """Verify logs are persisted to application.log."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        setup_logging(log_level="INFO", log_dir=temp_path)

        logger = get_logger("app_test_logger")
        logger.info("Application lifecycle test message")

        app_log_file = temp_path / "application.log"
        assert app_log_file.exists()

        content = app_log_file.read_text(encoding="utf-8")
        assert "Application lifecycle test message" in content
        assert "app_test_logger:" in content
        shutdown_logging()


def test_run_file_logging_and_isolation():
    """Verify workflow runs write to their own dedicated log files without cross-contamination."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        setup_logging(log_level="INFO", log_dir=temp_path)

        run_id_1 = "run-workflow-alpha"
        run_id_2 = "run-workflow-beta"

        logger = get_logger("workflow_logger")

        with scoped_run_context(run_id=run_id_1, workflow_name="RecommendationWorkflow"):
            logger.info("Alpha workflow started")
            log_trace_event(
                logger,
                agent_run_id=run_id_1,
                actor=TraceActor.INTENT_ROUTER,
                event_name="ROUTE_DECIDED",
                profile="recommend",
            )
            logger.info("Alpha workflow completed")

        with scoped_run_context(run_id=run_id_2, workflow_name="BRDWorkflow"):
            logger.info("Beta workflow started")
            log_trace_event(
                logger,
                agent_run_id=run_id_2,
                actor=TraceActor.LEAD_AGENT,
                event_name="ACTION_DECIDED",
                action_type="direct_work",
            )
            logger.info("Beta workflow completed")

        run1_file = temp_path / "runs" / f"{run_id_1}.log"
        run2_file = temp_path / "runs" / f"{run_id_2}.log"

        assert run1_file.exists()
        assert run2_file.exists()

        run1_content = run1_file.read_text(encoding="utf-8")
        run2_content = run2_file.read_text(encoding="utf-8")

        # Verify run 1 content
        assert "Alpha workflow started" in run1_content
        assert "ROUTE_DECIDED" in run1_content
        assert "profile=\"recommend\"" in run1_content
        assert "Beta workflow" not in run1_content

        # Verify run 2 content
        assert "Beta workflow started" in run2_content
        assert "ACTION_DECIDED" in run2_content
        assert "action_type=\"direct_work\"" in run2_content
        assert "Alpha workflow" not in run2_content

        shutdown_logging()


def test_concurrent_run_logging_isolation():
    """Verify concurrent async workflow executions do not mix their run-specific logs."""
    async def run_worker(run_id: str, message: str, delay: float, temp_path: Path):
        logger = get_logger(f"worker_{run_id}")
        async with scoped_run_context(run_id=run_id):
            logger.info("Start %s", message)
            await asyncio.sleep(delay)
            logger.info("Finish %s", message)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        setup_logging(log_level="INFO", log_dir=temp_path)

        async def main():
            await asyncio.gather(
                run_worker("concurrent-1", "Job 1", 0.05, temp_path),
                run_worker("concurrent-2", "Job 2", 0.02, temp_path),
            )

        asyncio.run(main())

        file1 = temp_path / "runs" / "concurrent-1.log"
        file2 = temp_path / "runs" / "concurrent-2.log"

        assert file1.exists()
        assert file2.exists()

        content1 = file1.read_text(encoding="utf-8")
        content2 = file2.read_text(encoding="utf-8")

        assert "Job 1" in content1
        assert "Job 2" not in content1

        assert "Job 2" in content2
        assert "Job 1" not in content2

        shutdown_logging()


def test_sensitive_credential_masking():
    """Verify API keys, passwords, and authorization tokens are masked."""
    test_message = (
        "Headers: {'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xyz'}, "
        "api_key: my_secret_api_key_12345, standalone=sk-proj1234567890abcdef, password='SecretPassword123!'"
    )

    sanitized = sanitize_log_message(test_message)

    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in sanitized
    assert "my_secret_api_key_12345" not in sanitized
    assert "sk-proj1234567890abcdef" not in sanitized
    assert "SecretPassword123!" not in sanitized
    assert "Bearer ***" in sanitized
    assert "api_key: ***" in sanitized
    assert "sk-***" in sanitized
    assert "password='***'" in sanitized


def test_llm_interaction_logging_captures_input_output_tokens():
    """Verify LLM inputs, outputs, tokens, and duration are captured and logged to run files."""
    import uuid
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.outputs import ChatGeneration, LLMResult
    from langchain_core.messages import AIMessage
    from observability.llm_telemetry import (
        get_current_telemetry_tracker,
        get_telemetry_callback_handler,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        setup_logging(log_dir=temp_path)

        run_id = f"llm-run-{uuid.uuid4()}"
        with scoped_run_context(run_id=run_id, workflow_name="RecommendationWorkflow"):
            handler = get_telemetry_callback_handler()
            call_id = uuid.uuid4()

            # 1. Trigger start
            messages = [
                [
                    SystemMessage(content="You are an Intent Router."),
                    HumanMessage(content="Recommend a payment gateway for low fee transactions"),
                ]
            ]
            handler.on_chat_model_start(
                serialized={"id": ["langchain", "chat_models", "openai", "ChatOpenAI"], "kwargs": {"model_name": "gpt-4o-mini"}},
                messages=messages,
                run_id=call_id,
            )

            # 2. Trigger end with tokens and structured output
            ai_msg = AIMessage(
                content='{"profile": "recommend", "high_stakes": false, "confidence": "high", "short_reason": "Clear recommendation query"}',
                response_metadata={"token_usage": {"prompt_tokens": 42, "completion_tokens": 18, "total_tokens": 60}},
            )
            gen = ChatGeneration(message=ai_msg)
            llm_result = LLMResult(generations=[[gen]], llm_output={"token_usage": {"prompt_tokens": 42, "completion_tokens": 18, "total_tokens": 60}})
            handler.on_llm_end(llm_result, run_id=call_id)

            tracker = get_current_telemetry_tracker()
            assert tracker is not None
            assert len(tracker.records) == 1
            record = tracker.records[0]
            assert record.input_prompt is not None
            assert "Recommend a payment gateway" in record.input_prompt
            assert record.output_text is not None
            assert "Clear recommendation query" in record.output_text
            assert record.total_tokens == 60
            assert record.input_tokens == 42
            assert record.output_tokens == 18
            assert record.rationale == "Clear recommendation query"
            assert record.model == "gpt-4o-mini"
            assert record.provider == "openai"

        run_file = temp_path / "runs" / f"{run_id}.log"
        assert run_file.exists()
        log_content = run_file.read_text(encoding="utf-8")

        assert "--- LLM INPUT ---" in log_content
        assert "Recommend a payment gateway" in log_content
        assert "--- LLM OUTPUT ---" in log_content
        assert "Clear recommendation query" in log_content
        assert "tokens=60" in log_content
        assert "in=42" in log_content
        assert "out=18" in log_content

        shutdown_logging()
