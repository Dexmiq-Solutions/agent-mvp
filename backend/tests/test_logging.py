"""Tests for logging configuration."""

import io
import logging

from app.core.logging import get_logger, setup_logging


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
    assert len(root_logger.handlers) == initial_handler_count == 1


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
