from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WATCHDOG = ROOT / "scripts/watch_storage_headroom.sh"


def test_storage_watchdog_is_scoped_and_fail_closed() -> None:
    source = WATCHDOG.read_text(encoding="utf-8")
    assert "WATCH_PGIDS must list explicit process-group leaders" in source
    assert 'current="$(ps -o pgid= -p "$leader"' in source
    assert source.count('[[ "$current" == "$leader" ]]') >= 3
    assert source.count("2>/dev/null | tr -d ' ' || true") == 3
    assert 'kill -TERM -- "-$leader"' in source
    assert 'kill -KILL -- "-$leader"' in source
    assert "FAILURES_REQUIRED:-2" in source
    assert "MIN_WORKSPACE_FREE_GIB:-500" in source
    assert "MIN_ROOT_FREE_GIB:-100" in source
    assert "MIN_TMP_FREE_GIB:-50" in source
    assert "df -ih" in source
    assert "HF_TOKEN" not in source
