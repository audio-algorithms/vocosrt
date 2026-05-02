# Copyright (c) 2026 AudioAlgorithms LLC. All rights reserved.
"""Tests for vocos_rt.forensic_log.

Validates the AAL-ENG-FLOG-001 v4.0 spec (per prompt sec.9):
- Timestamp format YYYY.MM.DD HH:MM:SS:MSEC (literal, not ISO)
- Entry format <ts> [<level>] [<module>:<lineno>] <message>
- 4 levels with bit-mask gating
- Multi-sink fan-out
- Thread safety
- Lazy formatting (verbose short-circuits without arg evaluation when caller guards)
"""

from __future__ import annotations

import io
import re
import threading
from datetime import datetime
from pathlib import Path

import pytest

from vocos_rt.forensic_log import (
    ConsoleSink,
    FileSink,
    Level,
    LogRecord,
    _Logger,
    _parse_env_level,
)

TS_PATTERN = re.compile(
    r"^\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2}:\d{3} "  # timestamp
    r"\[(error|warning|info|verbose)\] "              # level
    r"\[[\w.]+:\d+\] "                                 # module:lineno
    r".+$"                                              # message
)


# ----------------------------- LogRecord format -----------------------------


def test_format_line_timestamp_literal_not_iso() -> None:
    rec = LogRecord(
        timestamp=datetime(2026, 5, 1, 14, 30, 45, 123_456),
        level=Level.INFO,
        module="test_mod",
        lineno=42,
        message="hello world",
    )
    line = rec.format_line()
    assert line.startswith("2026.05.01 14:30:45:123 ")
    assert "[info] [test_mod:42] hello world" in line


def test_format_line_matches_full_pattern() -> None:
    rec = LogRecord(
        timestamp=datetime(2026, 1, 2, 3, 4, 5, 6_000),
        level=Level.ERROR,
        module="m",
        lineno=1,
        message="msg",
    )
    line = rec.format_line()
    assert TS_PATTERN.match(line), f"line did not match expected pattern: {line!r}"


def test_msec_padding_zero_left_pads_to_three_digits() -> None:
    rec = LogRecord(
        timestamp=datetime(2026, 1, 1, 0, 0, 0, 5_000),  # 5 ms
        level=Level.INFO,
        module="m",
        lineno=1,
        message="x",
    )
    assert ":005 " in rec.format_line()


# ----------------------------- Level mask gating -----------------------------


def test_default_mask_excludes_verbose() -> None:
    logger = _Logger(Level.DEFAULT)
    assert logger.error_enabled
    assert logger.warning_enabled
    assert logger.info_enabled
    assert not logger.verbose_enabled


def test_disabled_level_does_not_emit() -> None:
    sink = _CapturingSink()
    logger = _Logger(Level.ERROR)  # only ERROR enabled
    logger.add_sink(sink)
    logger.warning("ignored")
    logger.info("ignored")
    logger.verbose("ignored")
    assert sink.records == []
    logger.error("kept")
    assert len(sink.records) == 1
    assert sink.records[0].message == "kept"


def test_all_levels_route_to_all_active_sinks() -> None:
    sink_a = _CapturingSink()
    sink_b = _CapturingSink()
    logger = _Logger(Level.ALL)
    logger.add_sink(sink_a)
    logger.add_sink(sink_b)
    logger.error("e")
    logger.warning("w")
    logger.info("i")
    logger.verbose("v")
    for sink in (sink_a, sink_b):
        levels = [r.level for r in sink.records]
        assert levels == [Level.ERROR, Level.WARNING, Level.INFO, Level.VERBOSE]


# ----------------------------- Lazy formatting -----------------------------


def test_lazy_format_with_args() -> None:
    sink = _CapturingSink()
    logger = _Logger(Level.INFO)
    logger.add_sink(sink)
    logger.info("frame %d hash %s", 42, "abc")
    assert sink.records[0].message == "frame 42 hash abc"


def test_verbose_short_circuits_when_disabled() -> None:
    """When verbose is masked, the only cost is one bool check; no _emit, no fmt."""
    sink = _CapturingSink()
    logger = _Logger(Level.DEFAULT)  # no VERBOSE
    logger.add_sink(sink)
    counter = {"calls": 0}

    def expensive() -> str:
        counter["calls"] += 1
        return "x"

    # Idiomatic guarded usage from the hot path
    if logger.verbose_enabled:
        logger.verbose("frame %s", expensive())
    assert counter["calls"] == 0
    assert sink.records == []


# ----------------------------- Sinks -----------------------------


def test_console_sink_writes_to_provided_stream() -> None:
    buf = io.StringIO()
    logger = _Logger(Level.INFO)
    logger.add_sink(ConsoleSink(stream=buf))
    logger.info("hi")
    out = buf.getvalue()
    assert out.endswith("hi\n")
    assert TS_PATTERN.match(out.strip())


def test_file_sink_appends_and_persists(tmp_path: Path) -> None:
    log_path = tmp_path / "sub" / "test.log"  # parent should auto-create
    sink = FileSink(str(log_path))
    logger = _Logger(Level.INFO)
    logger.add_sink(sink)
    logger.info("first")
    logger.info("second")
    sink.close()
    contents = log_path.read_text(encoding="utf-8").splitlines()
    assert len(contents) == 2
    assert contents[0].endswith("first")
    assert contents[1].endswith("second")


def test_file_sink_appends_across_open_close(tmp_path: Path) -> None:
    log_path = tmp_path / "test.log"
    s1 = FileSink(str(log_path))
    logger = _Logger(Level.INFO)
    logger.add_sink(s1)
    logger.info("one")
    s1.close()
    logger.remove_sink(s1)

    s2 = FileSink(str(log_path))
    logger.add_sink(s2)
    logger.info("two")
    s2.close()

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0].endswith("one")
    assert lines[1].endswith("two")


def test_remove_sink_closes_it(tmp_path: Path) -> None:
    log_path = tmp_path / "rem.log"
    sink = FileSink(str(log_path))
    logger = _Logger(Level.INFO)
    logger.add_sink(sink)
    logger.remove_sink(sink)
    # After remove_sink, internal fh should be None so subsequent writes are no-ops.
    sink.write(LogRecord(datetime.now(), Level.INFO, "m", 1, "post-close"))
    assert log_path.read_text(encoding="utf-8") == ""


# ----------------------------- Env-var parsing -----------------------------


@pytest.mark.parametrize(
    "env,expected",
    [
        (None, Level.DEFAULT),
        ("", Level.DEFAULT),
        ("error", Level.ERROR),
        ("error,warning", Level.ERROR | Level.WARNING),
        ("all", Level.ALL),
        ("default", Level.DEFAULT),
        ("ERROR, INFO", Level.ERROR | Level.INFO),  # case + spaces
    ],
)
def test_parse_env_level(env: str | None, expected: Level) -> None:
    assert _parse_env_level(env) == expected


def test_parse_env_level_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        _parse_env_level("debug")  # not a level we accept


# ----------------------------- Thread safety -----------------------------


def test_concurrent_writes_no_loss_no_interleave(tmp_path: Path) -> None:
    log_path = tmp_path / "mt.log"
    sink = FileSink(str(log_path))
    logger = _Logger(Level.INFO)
    logger.add_sink(sink)

    n_threads = 8
    n_per_thread = 200

    def worker(tid: int) -> None:
        for i in range(n_per_thread):
            logger.info("t%d-i%d", tid, i)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    sink.close()

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == n_threads * n_per_thread
    # Every line must match the expected per-line pattern (no interleaved partial writes)
    for line in lines:
        assert TS_PATTERN.match(line), f"corrupted line: {line!r}"


# ----------------------------- Helpers -----------------------------


class _CapturingSink:
    """Test-only sink that retains every record for assertion."""

    def __init__(self) -> None:
        self.records: list[LogRecord] = []
        self._lock = threading.Lock()

    def write(self, record: LogRecord) -> None:
        with self._lock:
            self.records.append(record)

    def close(self) -> None:
        pass
