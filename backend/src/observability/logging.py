import logging
import sys
from typing import Any, TextIO

from core.config import get_settings

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_level: str | None = None,
    stream: TextIO | None = None,
) -> None:
    """Configure centralized application logging.

    Idempotent: Replaces existing handlers on the root logger to avoid duplicate log messages.
    """
    if log_level is None:
        log_level = get_settings().LOG_LEVEL

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to ensure idempotency and prevent duplicate log messages
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(stream or sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT))
    root_logger.addHandler(console_handler)

    # Set appropriate levels for noisy third-party loggers
    logging.getLogger("uvicorn").setLevel(numeric_level)
    logging.getLogger("uvicorn.error").setLevel(numeric_level)
    logging.getLogger("uvicorn.access").setLevel(
        logging.INFO if numeric_level <= logging.INFO else logging.WARNING
    )


def get_logger(name: str) -> logging.Logger:
    """Get a named logger instance."""
    return logging.getLogger(name)


class TraceActor:
    """Actor taxonomy for execution tracing."""

    LEAD_AGENT = "LEAD_AGENT"
    APPLICATION = "APPLICATION"
    SPECIALIZED_AGENT = "SPECIALIZED_AGENT"
    TOOL = "TOOL"


def format_trace_event(
    agent_run_id: str | None,
    actor: str,
    event_name: str,
    **metadata: Any,
) -> str:
    """Format a structured, correlated trace event line."""
    run_id_str = agent_run_id or "unspecified"
    meta_parts: list[str] = []
    for k, v in metadata.items():
        if v is not None:
            if isinstance(v, str):
                v_clean = v.replace("\n", " ").strip()
                if len(v_clean) > 200:
                    v_clean = v_clean[:197] + "..."
                meta_parts.append(f'{k}="{v_clean}"')
            else:
                meta_parts.append(f"{k}={v}")
    meta_str = (" " + " ".join(meta_parts)) if meta_parts else ""
    return f"[RUN: {run_id_str}] [ACTOR: {actor}] [EVENT: {event_name}]{meta_str}"


def log_trace_event(
    logger: logging.Logger,
    agent_run_id: str | None,
    actor: str,
    event_name: str,
    level: int = logging.INFO,
    **metadata: Any,
) -> None:
    """Log a structured, correlated trace event line."""
    msg = format_trace_event(agent_run_id, actor, event_name, **metadata)
    logger.log(level, msg)

