# Copyright (c) 2026 AudioAlgorithms LLC. All rights reserved.
"""Forensic logger — Python port of AAL-ENG-FLOG-001 v4.0.

Single module, importable as ``from vocos_rt.forensic_log import log``.

Design (from prompt §9):
- Levels: error, warning, info, verbose. Default enabled: error+warning+info.
- Timestamp format: ``YYYY.MM.DD HH:MM:SS:MSEC`` (literal, not ISO).
- Entry format: ``<timestamp> [<level>] [<module>:<lineno>] <message>``.
- Sinks: file, console, UDP syslog. Multiple sinks may be active at once.
- Configured via env vars: ``VOCOS_RT_LOG_LEVEL``, ``VOCOS_RT_LOG_FILE``,
  ``VOCOS_RT_LOG_SYSLOG_HOST``, ``VOCOS_RT_LOG_SYSLOG_PORT``.
- Thread-safe.
- Hot path: ``verbose`` level only. Caller MUST guard arg-expensive calls
  with ``if log.verbose_enabled:`` to achieve true zero-cost when disabled.
  ``log.verbose(...)`` itself short-circuits at the function entry when the
  level is masked off — no string formatting, no frame inspection.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from enum import IntFlag
from typing import IO, Protocol


class Level(IntFlag):
    """Bit-mask levels. Combine with ``|`` to set the enabled mask."""

    NONE = 0
    ERROR = 1 << 0
    WARNING = 1 << 1
    INFO = 1 << 2
    VERBOSE = 1 << 3
    ALL = ERROR | WARNING | INFO | VERBOSE
    DEFAULT = ERROR | WARNING | INFO


_LEVEL_NAMES: dict[Level, str] = {
    Level.ERROR: "error",
    Level.WARNING: "warning",
    Level.INFO: "info",
    Level.VERBOSE: "verbose",
}


@dataclass(frozen=True, slots=True)
class LogRecord:
    """One log entry; passed to every active sink."""

    timestamp: datetime
    level: Level
    module: str
    lineno: int
    message: str

    def format_line(self) -> str:
        ts = self.timestamp
        # YYYY.MM.DD HH:MM:SS:MSEC -- literal, not ISO
        ms = ts.microsecond // 1000
        ts_str = f"{ts.year:04d}.{ts.month:02d}.{ts.day:02d} {ts.hour:02d}:{ts.minute:02d}:{ts.second:02d}:{ms:03d}"
        return f"{ts_str} [{_LEVEL_NAMES[self.level]}] [{self.module}:{self.lineno}] {self.message}"


class Sink(Protocol):
    """Sink interface. Implementations must be thread-safe in ``write``."""

    def write(self, record: LogRecord) -> None: ...
    def close(self) -> None: ...


class ConsoleSink:
    """Writes to stderr by default. Lock-protected against interleaving."""

    __slots__ = ("_stream", "_lock")

    def __init__(self, stream: IO[str] | None = None) -> None:
        self._stream = stream if stream is not None else sys.stderr
        self._lock = threading.Lock()

    def write(self, record: LogRecord) -> None:
        line = record.format_line() + "\n"
        with self._lock:
            self._stream.write(line)
            self._stream.flush()

    def close(self) -> None:
        # Do not close stderr / stdout
        pass


class FileSink:
    """Append-only file sink. Opens lazily; flush after every write."""

    __slots__ = ("_path", "_fh", "_lock")

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        # Ensure parent directory exists
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._fh: IO[str] | None = open(path, "a", encoding="utf-8")

    def write(self, record: LogRecord) -> None:
        line = record.format_line() + "\n"
        with self._lock:
            if self._fh is None:
                return
            self._fh.write(line)
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


class UdpSyslogSink:
    """RFC-5424-ish UDP datagram sink. One datagram per record."""

    __slots__ = ("_addr", "_sock", "_lock")

    def __init__(self, host: str, port: int) -> None:
        self._addr = (host, int(port))
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._lock = threading.Lock()

    def write(self, record: LogRecord) -> None:
        payload = record.format_line().encode("utf-8", errors="replace")
        with self._lock:
            try:
                self._sock.sendto(payload, self._addr)
            except OSError:
                # UDP best-effort: never raise from a sink
                pass

    def close(self) -> None:
        with self._lock:
            try:
                self._sock.close()
            except OSError:
                pass


_LEVEL_BY_NAME: dict[str, Level] = {
    "error": Level.ERROR,
    "warning": Level.WARNING,
    "info": Level.INFO,
    "verbose": Level.VERBOSE,
    "all": Level.ALL,
    "default": Level.DEFAULT,
}


def _parse_env_level(value: str | None) -> Level:
    """Accept comma-separated names (``error,warning,info``) or single name."""
    if not value:
        return Level.DEFAULT
    mask = Level.NONE
    for part in value.lower().split(","):
        part = part.strip()
        if not part:
            continue
        if part not in _LEVEL_BY_NAME:
            raise ValueError(f"Unknown forensic-log level: {part!r}")
        mask |= _LEVEL_BY_NAME[part]
    return mask if mask != Level.NONE else Level.DEFAULT


class _Logger:
    """Thread-safe multi-sink logger. Single module-level instance ``log``."""

    __slots__ = ("_mask", "_sinks", "_sinks_lock")

    def __init__(self, mask: Level) -> None:
        self._mask = mask
        self._sinks: list[Sink] = []
        self._sinks_lock = threading.Lock()

    # --- configuration ---

    def set_level(self, mask: Level) -> None:
        self._mask = mask

    def add_sink(self, sink: Sink) -> None:
        with self._sinks_lock:
            self._sinks.append(sink)

    def remove_sink(self, sink: Sink) -> None:
        with self._sinks_lock:
            try:
                self._sinks.remove(sink)
            except ValueError:
                pass
            sink.close()

    def close_all_sinks(self) -> None:
        with self._sinks_lock:
            for sink in self._sinks:
                sink.close()
            self._sinks.clear()

    # --- level-check fast path (hot path callers MUST use these) ---

    @property
    def verbose_enabled(self) -> bool:
        return bool(self._mask & Level.VERBOSE)

    @property
    def info_enabled(self) -> bool:
        return bool(self._mask & Level.INFO)

    @property
    def warning_enabled(self) -> bool:
        return bool(self._mask & Level.WARNING)

    @property
    def error_enabled(self) -> bool:
        return bool(self._mask & Level.ERROR)

    # --- emit ---

    def _emit(self, level: Level, msg: str, args: tuple[object, ...]) -> None:
        # Frame inspection only happens past the level mask; cheap when disabled.
        frame = sys._getframe(2)
        module = frame.f_globals.get("__name__", "?")
        lineno = frame.f_lineno
        text = msg % args if args else msg
        record = LogRecord(
            timestamp=datetime.now(),
            level=level,
            module=module,
            lineno=lineno,
            message=text,
        )
        # Snapshot the sink list to avoid holding the lock during sink.write
        with self._sinks_lock:
            sinks = list(self._sinks)
        for sink in sinks:
            sink.write(record)

    def error(self, msg: str, *args: object) -> None:
        if not (self._mask & Level.ERROR):
            return
        self._emit(Level.ERROR, msg, args)

    def warning(self, msg: str, *args: object) -> None:
        if not (self._mask & Level.WARNING):
            return
        self._emit(Level.WARNING, msg, args)

    def info(self, msg: str, *args: object) -> None:
        if not (self._mask & Level.INFO):
            return
        self._emit(Level.INFO, msg, args)

    def verbose(self, msg: str, *args: object) -> None:
        # Hot path. Caller should additionally guard with ``if log.verbose_enabled:``
        # for arg-expensive call sites to skip arg evaluation entirely.
        if not (self._mask & Level.VERBOSE):
            return
        self._emit(Level.VERBOSE, msg, args)


def _build_default_logger() -> _Logger:
    mask = _parse_env_level(os.environ.get("VOCOS_RT_LOG_LEVEL"))
    logger = _Logger(mask)

    # Default: console sink always active
    logger.add_sink(ConsoleSink())

    # Optional file sink
    log_file = os.environ.get("VOCOS_RT_LOG_FILE")
    if log_file:
        logger.add_sink(FileSink(log_file))

    # Optional UDP syslog sink
    syslog_host = os.environ.get("VOCOS_RT_LOG_SYSLOG_HOST")
    syslog_port = os.environ.get("VOCOS_RT_LOG_SYSLOG_PORT")
    if syslog_host and syslog_port:
        logger.add_sink(UdpSyslogSink(syslog_host, int(syslog_port)))

    return logger


log: _Logger = _build_default_logger()
"""Module-level logger singleton. Use ``log.info(...)``, ``log.verbose(...)``, etc."""


__all__ = [
    "ConsoleSink",
    "FileSink",
    "Level",
    "LogRecord",
    "Sink",
    "UdpSyslogSink",
    "log",
]
