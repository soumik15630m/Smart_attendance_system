"""Unit tests for src/services/script_runner.py.

Uses real (but trivial) subprocesses -- short `python -c "..."` snippets --
instead of the actual onboarding scripts, so these run fast and don't
depend on camera hardware, GPUs, or a database.
"""

import sys
import time

import pytest

from src.services.script_runner import LocalScriptRunner, ScriptSpec


def _runner(tmp_path, script_id="demo", script_body="print('hi')", long_running=False):
    script_path = tmp_path / "demo.py"
    script_path.write_text(script_body)
    spec = ScriptSpec(
        script_id=script_id,
        title="Demo",
        description="Demo script",
        script_path="demo.py",
        category="Testing",
        long_running=long_running,
    )
    return LocalScriptRunner(tmp_path, [spec])


def _wait_until(predicate, timeout=5.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# --- start_script -----------------------------------------------------


def test_start_script_runs_to_completion(tmp_path):
    runner = _runner(tmp_path, script_body="print('done')")
    state = runner.start_script("demo")
    assert state["status"] == "running"

    assert _wait_until(lambda: runner.get_status("demo") == "completed")
    final = runner.get_script("demo")
    assert final["exit_code"] == 0
    assert final["finished_at"] is not None
    logs = runner.get_logs("demo")
    assert any("done" in line for line in logs)
    assert any("exited with code 0" in line for line in logs)


def test_start_script_marks_failed_on_nonzero_exit(tmp_path):
    runner = _runner(tmp_path, script_body="import sys; sys.exit(3)")
    runner.start_script("demo")
    assert _wait_until(lambda: runner.get_status("demo") == "failed")
    assert runner.get_script("demo")["exit_code"] == 3


def test_start_script_rejects_second_start_while_running(tmp_path):
    runner = _runner(tmp_path, script_body="import time; time.sleep(2)")
    runner.start_script("demo")
    assert runner.get_status("demo") == "running"

    with pytest.raises(RuntimeError, match="already running"):
        runner.start_script("demo")

    # Clean up the still-running process so the test doesn't leak it.
    runner.stop_script("demo")
    _wait_until(lambda: runner.get_status("demo") in {"stopped", "completed", "failed"})


def test_start_script_unknown_id_raises_key_error(tmp_path):
    runner = _runner(tmp_path)
    with pytest.raises(KeyError):
        runner.start_script("does-not-exist")


def test_start_script_missing_file_raises_runtime_error(tmp_path):
    spec = ScriptSpec(
        script_id="ghost",
        title="Ghost",
        description="File doesn't exist on disk",
        script_path="ghost.py",
        category="Testing",
    )
    runner = LocalScriptRunner(tmp_path, [spec])
    with pytest.raises(RuntimeError, match="Script not found"):
        runner.start_script("ghost")


# --- stop_script --------------------------------------------------------


def test_stop_script_rejects_when_not_running(tmp_path):
    runner = _runner(tmp_path)
    with pytest.raises(RuntimeError, match="not currently running"):
        runner.stop_script("demo")


def test_stop_script_unknown_id_raises_key_error(tmp_path):
    runner = _runner(tmp_path)
    with pytest.raises(KeyError):
        runner.stop_script("does-not-exist")


def test_stop_script_marks_stopped_and_terminates_process(tmp_path):
    runner = _runner(tmp_path, script_body="import time; time.sleep(30)")
    runner.start_script("demo")
    assert runner.get_status("demo") == "running"

    state = runner.stop_script("demo")
    # stop_script blocks (up to 5s) waiting for the process to exit, so by
    # the time it returns the background thread has often already flipped
    # the status to its terminal value.
    assert state["status"] in {"stopping", "stopped"}

    assert _wait_until(lambda: runner.get_status("demo") == "stopped")
    final = runner.get_script("demo")
    assert final["exit_code"] is not None
    assert final["exit_code"] != 0


# --- _build_command validation -------------------------------------------


def test_build_command_register_face_requires_name(tmp_path):
    script_path = tmp_path / "register_face.py"
    script_path.write_text("print('noop')")
    spec = ScriptSpec(
        script_id="register_face",
        title="Register Face",
        description="",
        script_path="register_face.py",
        category="Onboarding",
    )
    runner = LocalScriptRunner(tmp_path, [spec])
    with pytest.raises(ValueError, match="Name is required"):
        runner._build_command(spec, {})


def test_build_command_register_face_includes_name_and_employee_id(tmp_path):
    script_path = tmp_path / "register_face.py"
    script_path.write_text("print('noop')")
    spec = ScriptSpec(
        script_id="register_face",
        title="Register Face",
        description="",
        script_path="register_face.py",
        category="Onboarding",
    )
    runner = LocalScriptRunner(tmp_path, [spec])
    command = runner._build_command(spec, {"name": "Alice", "employee_id": "EMP-1"})
    assert command == [
        sys.executable,
        str(script_path),
        "--name",
        "Alice",
        "--employee-id",
        "EMP-1",
    ]


def test_build_command_non_register_scripts_ignore_payload(tmp_path):
    runner = _runner(tmp_path, script_id="seed_db")
    spec = runner._get_spec("seed_db")
    command = runner._build_command(spec, {"name": "irrelevant"})
    assert command == [sys.executable, str(tmp_path / "demo.py")]


# --- get_logs / list_scripts ---------------------------------------------


def test_get_logs_clamps_tail_to_valid_range(tmp_path):
    runner = _runner(tmp_path, script_body="print('done')")
    runner.start_script("demo")
    assert _wait_until(lambda: runner.get_status("demo") == "completed")

    # Requesting an absurdly small or large tail is clamped, not rejected.
    assert len(runner.get_logs("demo", tail=1)) <= 10
    assert len(runner.get_logs("demo", tail=100000)) <= 600


def test_get_logs_unknown_id_raises_key_error(tmp_path):
    runner = _runner(tmp_path)
    with pytest.raises(KeyError):
        runner.get_logs("does-not-exist")


def test_list_scripts_reflects_state_changes(tmp_path):
    runner = _runner(tmp_path, script_body="print('done')")
    before = runner.list_scripts()
    assert before[0]["status"] == "idle"

    runner.start_script("demo")
    assert _wait_until(lambda: runner.get_status("demo") == "completed")

    after = runner.list_scripts()
    assert after[0]["status"] == "completed"
    assert after[0]["exit_code"] == 0
