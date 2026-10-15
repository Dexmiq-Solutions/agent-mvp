import contextvars
import logging
import os
from pathlib import Path
import re
import sys
import threading
from typing import Any, Optional, TextIO

from core.config import get_settings

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------------------------------------
# Sensitive Credential Sanitization
# ---------------------------------------------------------------------------

SENSITIVE_PATTERNS = [
    (re.compile(r"(Bearer\s+)\S{8,}", re.IGNORECASE), r"\1***"),
    (re.compile(r"(api[_\-]?key[\"']?\s*[:=]\s*[\"']?)[A-Za-z0-9_\-]{8,}([\"']?)", re.IGNORECASE), r"\1***\2"),
    (re.compile(r"(password[\"']?\s*[:=]\s*[\"']?)[^\"'\s]{4,}([\"']?)", re.IGNORECASE), r"\1***\2"),
    (re.compile(r"(sk-[a-zA-Z0-9_\-]{8,})", re.IGNORECASE), r"sk-***"),
    (re.compile(r"(token[\"']?\s*[:=]\s*[\"']?)[A-Za-z0-9_\-\.]{8,}([\"']?)", re.IGNORECASE), r"\1***\2"),
]


def sanitize_log_message(msg: str) -> str:
    """Mask credentials, API keys, tokens, and passwords from log strings."""
    if not msg or not isinstance(msg, str):
        return msg
    sanitized = msg
    for pattern, repl in SENSITIVE_PATTERNS:
        sanitized = pattern.sub(repl, sanitized)
    return sanitized


class SanitizedFormatter(logging.Formatter):
    """Custom formatter ensuring sensitive credentials are never written to any log output."""

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        return sanitize_log_message(formatted)


# ---------------------------------------------------------------------------
# File Logging & Directory Paths
# ---------------------------------------------------------------------------

def get_log_dir() -> Path:
    """Get and automatically create the centralized log directory."""
    settings = get_settings()
    custom_dir = getattr(settings, "LOG_DIR", None) or os.getenv("LOG_DIR")
    if custom_dir:
        log_dir = Path(custom_dir).resolve()
    else:
        # Default to backend/logs directory
        log_dir = Path(__file__).resolve().parent.parent.parent / "logs"

    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "runs").mkdir(parents=True, exist_ok=True)
    return log_dir


# ---------------------------------------------------------------------------
# Execution Context Management (Isolated via contextvars)
# ---------------------------------------------------------------------------

_current_run_id: contextvars.ContextVar[Optional[str]] = (
    contextvars.ContextVar("current_run_id", default=None)
)

_current_workflow_name: contextvars.ContextVar[Optional[str]] = (
    contextvars.ContextVar("current_workflow_name", default=None)
)


def get_current_run_id() -> Optional[str]:
    """Retrieve the active run ID for the current async or thread context."""
    return _current_run_id.get()


def set_current_run_id(run_id: Optional[str]) -> contextvars.Token:
    """Set the active run ID for the current context."""
    return _current_run_id.set(run_id)


def reset_current_run_id(token: contextvars.Token) -> None:
    """Reset the active run ID for the current context."""
    _current_run_id.reset(token)


def get_current_workflow_name() -> Optional[str]:
    """Retrieve the active workflow name for the current context."""
    return _current_workflow_name.get()


# ---------------------------------------------------------------------------
# Run File Logging Handler
# ---------------------------------------------------------------------------

class RunFileLoggingHandler(logging.Handler):
    """Dispatches log records to isolated per-run log files based on active execution context.

    Ensures that concurrent and repeated workflow runs write exclusively to their own
    dedicated log files without cross-contamination.
    """

    def __init__(self, log_dir: Path, formatter: logging.Formatter) -> None:
        super().__init__()
        self.runs_dir = log_dir / "runs"
        self.formatter = formatter
        self._handlers: dict[str, logging.FileHandler] = {}
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            run_id = (
                getattr(record, "agent_run_id", None)
                or getattr(record, "run_id", None)
                or get_current_run_id()
            )
            if not run_id:
                return

            handler = self._get_or_create_handler(run_id)
            handler.emit(record)
        except Exception:
            self.handleError(record)

    def _get_or_create_handler(self, run_id: str) -> logging.FileHandler:
        with self._lock:
            if run_id not in self._handlers:
                self.runs_dir.mkdir(parents=True, exist_ok=True)
                file_path = self.runs_dir / f"{run_id}.log"
                fh = logging.FileHandler(file_path, encoding="utf-8")
                fh.setFormatter(self.formatter)
                fh.setLevel(logging.DEBUG)
                self._handlers[run_id] = fh
            return self._handlers[run_id]

    def close_run(self, run_id: str) -> None:
        """Flush, close, and remove the file handler for a completed run."""
        with self._lock:
            handler = self._handlers.pop(run_id, None)
            if handler:
                try:
                    handler.flush()
                    handler.close()
                except Exception:
                    pass


_global_run_log_handler: Optional[RunFileLoggingHandler] = None
_file_logging_enabled: bool = True


def close_run_log_file(run_id: str) -> None:
    """Close and flush the run-specific log file for a given run ID."""
    global _global_run_log_handler
    if _global_run_log_handler is not None:
        _global_run_log_handler.close_run(run_id)


class scoped_run_context:
    """Context manager (sync and async) to isolate a workflow execution run.

    Sets active run_id and workflow_name in ContextVars, attaches an LLMTelemetryTracker
    for token and duration tracking, and ensures log messages are written to the dedicated
    runs/<run_id>.log file without leaking into other concurrent executions.
    """

    def __init__(self, run_id: str, workflow_name: Optional[str] = None) -> None:
        self.run_id = run_id
        self.workflow_name = workflow_name
        self._token_run_id: Optional[contextvars.Token] = None
        self._token_wf: Optional[contextvars.Token] = None
        self._telemetry_token: Optional[contextvars.Token] = None

    def __enter__(self) -> "scoped_run_context":
        self._token_run_id = _current_run_id.set(self.run_id)
        if self.workflow_name:
            self._token_wf = _current_workflow_name.set(self.workflow_name)

        # Ensure run file logging handler is attached if file logging is enabled
        global _global_run_log_handler, _file_logging_enabled
        if _file_logging_enabled:
            root_logger = logging.getLogger()
            if _global_run_log_handler is None or _global_run_log_handler not in root_logger.handlers:
                target_dir = get_log_dir()
                (target_dir / "runs").mkdir(parents=True, exist_ok=True)
                formatter = SanitizedFormatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
                _global_run_log_handler = RunFileLoggingHandler(log_dir=target_dir, formatter=formatter)
                _global_run_log_handler.setLevel(logging.DEBUG)
                root_logger.addHandler(_global_run_log_handler)
                if root_logger.level > logging.INFO:
                    root_logger.setLevel(logging.INFO)

        # Ensure an active telemetry tracker is set for this run ID
        from observability.llm_telemetry import (
            LLMTelemetryTracker,
            get_current_telemetry_tracker,
            set_current_telemetry_tracker,
        )

        existing = get_current_telemetry_tracker()
        if existing is None or existing.run_id != self.run_id:
            tracker = LLMTelemetryTracker(run_id=self.run_id)
            self._telemetry_token = set_current_telemetry_tracker(tracker)

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._telemetry_token is not None:
            from observability.llm_telemetry import reset_current_telemetry_tracker

            reset_current_telemetry_tracker(self._telemetry_token)
            self._telemetry_token = None

        if self._token_wf is not None:
            _current_workflow_name.reset(self._token_wf)
            self._token_wf = None

        if self._token_run_id is not None:
            _current_run_id.reset(self._token_run_id)
            self._token_run_id = None

        close_run_log_file(self.run_id)

    async def __aenter__(self) -> "scoped_run_context":
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)


# ---------------------------------------------------------------------------
# Setup Centralized Application Logging
# ---------------------------------------------------------------------------

def shutdown_logging() -> None:
    """Close and flush all active handlers on the root logger and reset run log handler.

    Essential for clean process termination and Windows file lock release during test teardowns.
    """
    global _global_run_log_handler
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        try:
            handler.flush()
            handler.close()
        except Exception:
            pass
        root_logger.removeHandler(handler)

    if _global_run_log_handler is not None:
        with _global_run_log_handler._lock:
            for h in list(_global_run_log_handler._handlers.values()):
                try:
                    h.flush()
                    h.close()
                except Exception:
                    pass
            _global_run_log_handler._handlers.clear()
        _global_run_log_handler = None


def setup_logging(
    log_level: str | None = None,
    stream: TextIO | None = None,
    enable_file_logging: bool = True,
    log_dir: str | Path | None = None,
) -> None:
    """Configure centralized application logging.

    Idempotent: Replaces existing handlers on the root logger to avoid duplicate log messages.
    Supports console terminal output, persistent application-level logging, and isolated
    per-run workflow log dispatching.
    """
    global _global_run_log_handler, _file_logging_enabled
    _file_logging_enabled = enable_file_logging

    # Close and remove existing handlers to ensure idempotency and prevent duplicate log messages
    shutdown_logging()

    if log_level is None:
        log_level = get_settings().LOG_LEVEL

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    formatter = SanitizedFormatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    # 1. Console / Stream Handler (always enabled for terminal output)
    console_handler = logging.StreamHandler(stream or sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 2. File Logging (Application-level and Run-level)
    if enable_file_logging:
        target_dir = Path(log_dir) if log_dir else get_log_dir()
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "runs").mkdir(parents=True, exist_ok=True)

        # Application-level log file
        app_log_path = target_dir / "application.log"
        app_file_handler = logging.FileHandler(app_log_path, encoding="utf-8")
        app_file_handler.setLevel(numeric_level)
        app_file_handler.setFormatter(formatter)
        root_logger.addHandler(app_file_handler)

        # Run-level file dispatcher handler
        _global_run_log_handler = RunFileLoggingHandler(log_dir=target_dir, formatter=formatter)
        _global_run_log_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(_global_run_log_handler)

    # Set appropriate levels for noisy third-party loggers
    logging.getLogger("uvicorn").setLevel(numeric_level)
    logging.getLogger("uvicorn.error").setLevel(numeric_level)
    logging.getLogger("uvicorn.access").setLevel(
        logging.INFO if numeric_level <= logging.INFO else logging.WARNING
    )
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger instance."""
    return logging.getLogger(name)


class TraceActor:
    """Actor taxonomy for execution tracing."""

    LEAD_AGENT = "LEAD_AGENT"
    APPLICATION = "APPLICATION"
    SPECIALIZED_AGENT = "SPECIALIZED_AGENT"
    TOOL = "TOOL"
    RECOMMENDATION_WORKFLOW = "RECOMMENDATION_WORKFLOW"
    INTENT_ROUTER = "INTENT_ROUTER"


def format_trace_event(
    agent_run_id: str | None,
    actor: str,
    event_name: str,
    **metadata: Any,
) -> str:
    """Format a structured, correlated trace event line."""
    run_id_str = agent_run_id or get_current_run_id() or "unspecified"
    meta_parts: list[str] = []
    for k, v in metadata.items():
        if v is not None:
            if isinstance(v, str):
                v_clean = sanitize_log_message(v.replace("\n", " ").strip())
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
    effective_run_id = agent_run_id or get_current_run_id()
    msg = format_trace_event(effective_run_id, actor, event_name, **metadata)
    extra = {"agent_run_id": effective_run_id} if effective_run_id else {}
    logger.log(level, msg, extra=extra)
