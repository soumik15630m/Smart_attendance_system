from __future__ import annotations

import os
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_LOG_LINES = 600


@dataclass(frozen=True)
class ScriptSpec:
    script_id: str
    title: str
    description: str
    script_path: str
    category: str
    long_running: bool = False


@dataclass
class ScriptState:
    status: str = "idle"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    exit_code: int | None = None
    stop_requested: bool = False
    process: subprocess.Popen[str] | None = None
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_LOG_LINES))


class LocalScriptRunner:
    def __init__(self, project_root: Path, specs: list[ScriptSpec]):
        self.project_root = project_root
        self._specs = {spec.script_id: spec for spec in specs}
        self._ordered_ids = [spec.script_id for spec in specs]
        self._states = {spec.script_id: ScriptState() for spec in specs}
        self._lock = threading.RLock()

    def list_scripts(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                self._serialize_script(self._specs[script_id], self._states[script_id])
                for script_id in self._ordered_ids
            ]

    def get_script(self, script_id: str) -> dict[str, Any]:
        with self._lock:
            spec = self._get_spec(script_id)
            state = self._states[script_id]
            return self._serialize_script(spec, state)

    def get_logs(self, script_id: str, tail: int = 200) -> list[str]:
        clamped_tail = max(10, min(tail, MAX_LOG_LINES))
        with self._lock:
            self._get_spec(script_id)
            state = self._states[script_id]
            return list(state.logs)[-clamped_tail:]

    def get_status(self, script_id: str) -> str:
        with self._lock:
            self._get_spec(script_id)
            return self._states[script_id].status

    def start_script(
        self, script_id: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        payload = payload or {}

        with self._lock:
            spec = self._get_spec(script_id)
            state = self._states[script_id]

            if state.status in {"running", "stopping"}:
                raise RuntimeError(f"{script_id} is already running.")

            command = self._build_command(spec, payload)
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            creation_flags = (
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                if os.name == "nt"
                else 0
            )

            try:
                process = subprocess.Popen(
                    command,
                    cwd=str(self.project_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                    env=env,
                    creationflags=creation_flags,
                )
            except Exception as exc:
                raise RuntimeError(f"Unable to start {script_id}: {exc}") from exc

            state.status = "running"
            state.started_at = datetime.now(timezone.utc)
            state.finished_at = None
            state.exit_code = None
            state.stop_requested = False
            state.process = process
            state.logs.clear()
            state.logs.append(f"$ {' '.join(command)}")
            state.logs.append("Process started.")

            thread = threading.Thread(
                target=self._stream_output,
                args=(script_id, process),
                daemon=True,
            )
            thread.start()

            return self._serialize_script(spec, state)

    def stop_script(self, script_id: str) -> dict[str, Any]:
        with self._lock:
            spec = self._get_spec(script_id)
            state = self._states[script_id]
            process = state.process

            if state.status not in {"running", "stopping"} or process is None:
                raise RuntimeError(f"{script_id} is not currently running.")

            state.stop_requested = True
            state.status = "stopping"
            state.logs.append("Stopping process...")

        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        finally:
            with self._lock:
                state.logs.append("Stop signal sent.")
                return self._serialize_script(spec, state)

    def _stream_output(self, script_id: str, process: subprocess.Popen[str]) -> None:
        try:
            if process.stdout is not None:
                for line in process.stdout:
                    message = line.rstrip()
                    if message:
                        self._append_log(script_id, message)
        finally:
            if process.stdout is not None:
                process.stdout.close()

            return_code = process.wait()
            with self._lock:
                state = self._states[script_id]
                state.exit_code = return_code
                state.finished_at = datetime.now(timezone.utc)

                if state.stop_requested:
                    state.status = "stopped"
                elif return_code == 0:
                    state.status = "completed"
                else:
                    state.status = "failed"

                state.process = None
                state.stop_requested = False
                state.logs.append(f"Process exited with code {return_code}.")

    def _append_log(self, script_id: str, line: str) -> None:
        with self._lock:
            state = self._states[script_id]
            state.logs.append(line)

    def _build_command(self, spec: ScriptSpec, payload: dict[str, Any]) -> list[str]:
        script_path = self.project_root / spec.script_path
        if not script_path.exists():
            raise RuntimeError(f"Script not found: {spec.script_path}")

        command = [sys.executable, str(script_path)]

        if spec.script_id == "register_face":
            name = str(payload.get("name", "")).strip()
            employee_id = str(payload.get("employee_id", "")).strip()

            if not name:
                raise ValueError("Name is required to run face registration.")

            command.extend(["--name", name])
            command.extend(["--employee-id", employee_id])

        return command

    def _serialize_script(self, spec: ScriptSpec, state: ScriptState) -> dict[str, Any]:
        last_log = state.logs[-1] if state.logs else ""
        return {
            "id": spec.script_id,
            "title": spec.title,
            "description": spec.description,
            "category": spec.category,
            "long_running": spec.long_running,
            "status": state.status,
            "started_at": self._to_iso(state.started_at),
            "finished_at": self._to_iso(state.finished_at),
            "exit_code": state.exit_code,
            "last_log": last_log,
            "log_size": len(state.logs),
        }

    def _get_spec(self, script_id: str) -> ScriptSpec:
        spec = self._specs.get(script_id)
        if spec is None:
            raise KeyError(script_id)
        return spec

    @staticmethod
    def _to_iso(value: datetime | None) -> str | None:
        if value is None:
            return None
        return value.isoformat()


def default_script_specs() -> list[ScriptSpec]:
    return [
        ScriptSpec(
            script_id="test_gpu",
            title="GPU Diagnostics",
            description="Validate CUDA/cuDNN and ONNX Runtime provider setup.",
            script_path="scripts/test_gpu.py",
            category="Onboarding",
        ),
        ScriptSpec(
            script_id="seed_db",
            title="Seed Starter Data",
            description="Insert a sample person embedding for first-run verification.",
            script_path="scripts/seed_db.py",
            category="Onboarding",
        ),
        ScriptSpec(
            script_id="register_face",
            title="Register Face",
            description="Launch webcam capture and enroll a person into the system.",
            script_path="scripts/register_face.py",
            category="Onboarding",
        ),
        ScriptSpec(
            script_id="camera_client",
            title="Run Camera Client",
            description="Start the edge camera pipeline and recognition loop.",
            script_path="scripts/camera_client.py",
            category="Operations",
            long_running=True,
        ),
        ScriptSpec(
            script_id="show_attendance",
            title="Attendance Snapshot",
            description="Read recent attendance records directly from the database.",
            script_path="scripts/show_attendance.py",
            category="Operations",
        ),
    ]
