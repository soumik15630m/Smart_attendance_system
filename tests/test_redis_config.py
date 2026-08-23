"""Unit tests for src/redis_config.py's pure, dependency-free logic.

These target the branching that decides which cache backend to use and how
to auto-start a local Redis, without touching a real Redis server or the
network. Everything here is monkeypatched at the module level, since the
relevant constants (REDIS_HOST, REDIS_URL, ...) are read from the
environment once at import time.
"""

import pytest

from src import redis_config as rc

# --- _redis_host_port -------------------------------------------------


def test_redis_host_port_uses_redis_url_when_set(monkeypatch):
    monkeypatch.setattr(rc, "REDIS_URL", "redis://cache.internal:6380/2")
    monkeypatch.setattr(rc, "REDIS_HOST", "should-be-ignored")
    monkeypatch.setattr(rc, "REDIS_PORT", 9999)
    assert rc._redis_host_port() == ("cache.internal", 6380)


def test_redis_host_port_falls_back_to_host_port_vars(monkeypatch):
    monkeypatch.setattr(rc, "REDIS_URL", "")
    monkeypatch.setattr(rc, "REDIS_HOST", "10.0.0.5")
    monkeypatch.setattr(rc, "REDIS_PORT", 6400)
    assert rc._redis_host_port() == ("10.0.0.5", 6400)


def test_redis_host_port_defaults_when_url_missing_host_and_port(monkeypatch):
    monkeypatch.setattr(rc, "REDIS_URL", "redis://")
    assert rc._redis_host_port() == ("localhost", 6379)


# --- _is_local_redis_target ---------------------------------------------


@pytest.mark.parametrize(
    "host,expected",
    [
        ("localhost", True),
        ("127.0.0.1", True),
        ("::1", True),
        ("[::1]", True),
        ("LOCALHOST", True),
        ("cache.internal", False),
        ("10.0.0.5", False),
    ],
)
def test_is_local_redis_target(monkeypatch, host, expected):
    monkeypatch.setattr(rc, "REDIS_URL", "")
    monkeypatch.setattr(rc, "REDIS_HOST", host)
    monkeypatch.setattr(rc, "REDIS_PORT", 6379)
    assert rc._is_local_redis_target() is expected


# --- _parse_custom_start_command ----------------------------------------


def test_parse_custom_start_command_returns_none_when_unset(monkeypatch):
    monkeypatch.setattr(rc, "LOCAL_REDIS_START_CMD", "")
    assert rc._parse_custom_start_command() is None


def test_parse_custom_start_command_splits_shell_style(monkeypatch):
    monkeypatch.setattr(
        rc, "LOCAL_REDIS_START_CMD", 'redis-server --port 6379 --requirepass "a b"'
    )
    assert rc._parse_custom_start_command() == [
        "redis-server",
        "--port",
        "6379",
        "--requirepass",
        "a b",
    ]


# --- _is_running_in_container --------------------------------------------


def test_is_running_in_container_true_via_dockerenv(monkeypatch):
    monkeypatch.setattr(rc.os.path, "exists", lambda path: path == "/.dockerenv")
    assert rc._is_running_in_container() is True


def test_is_running_in_container_true_via_cgroup(monkeypatch, tmp_path):
    cgroup_file = tmp_path / "cgroup"
    cgroup_file.write_text("12:pids:/docker/abcdef1234\n")
    real_open = open

    monkeypatch.setattr(rc.os.path, "exists", lambda path: path == "/proc/1/cgroup")
    monkeypatch.setattr(
        "builtins.open", lambda path, *a, **kw: real_open(cgroup_file, *a, **kw)
    )
    assert rc._is_running_in_container() is True


def test_is_running_in_container_false_when_no_markers(monkeypatch, tmp_path):
    cgroup_file = tmp_path / "cgroup"
    cgroup_file.write_text("12:pids:/user.slice\n")
    real_open = open

    monkeypatch.setattr(rc.os.path, "exists", lambda path: path == "/proc/1/cgroup")
    monkeypatch.setattr(
        "builtins.open", lambda path, *a, **kw: real_open(cgroup_file, *a, **kw)
    )
    assert rc._is_running_in_container() is False


def test_is_running_in_container_false_when_no_dockerenv_or_cgroup(monkeypatch):
    monkeypatch.setattr(rc.os.path, "exists", lambda path: False)
    assert rc._is_running_in_container() is False


def test_is_running_in_container_handles_unreadable_cgroup(monkeypatch):
    def fake_exists(path):
        return path == "/proc/1/cgroup"

    def fake_open(*_args, **_kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(rc.os.path, "exists", fake_exists)
    monkeypatch.setattr("builtins.open", fake_open)
    assert rc._is_running_in_container() is False


# --- _docker_start_commands -----------------------------------------------


def test_docker_start_commands_empty_without_compose_file(monkeypatch):
    monkeypatch.setattr(rc.os.path, "exists", lambda path: False)
    assert rc._docker_start_commands() == []


def test_docker_start_commands_empty_when_already_in_container(monkeypatch):
    monkeypatch.setattr(rc.os.path, "exists", lambda path: True)
    monkeypatch.setattr(rc, "_is_running_in_container", lambda: True)
    assert rc._docker_start_commands() == []


def test_docker_start_commands_prefers_docker_cli_then_compose(monkeypatch):
    monkeypatch.setattr(rc.os.path, "exists", lambda path: True)
    monkeypatch.setattr(rc, "_is_running_in_container", lambda: False)
    monkeypatch.setattr(
        rc.shutil,
        "which",
        lambda exe: f"/usr/bin/{exe}" if exe in {"docker", "docker-compose"} else None,
    )
    commands = rc._docker_start_commands()
    assert commands == [
        (["docker", "compose", "up", "-d", "redis"], rc.PROJECT_ROOT),
        (["docker-compose", "up", "-d", "redis"], rc.PROJECT_ROOT),
    ]


def test_docker_start_commands_empty_when_neither_cli_available(monkeypatch):
    monkeypatch.setattr(rc.os.path, "exists", lambda path: True)
    monkeypatch.setattr(rc, "_is_running_in_container", lambda: False)
    monkeypatch.setattr(rc.shutil, "which", lambda exe: None)
    assert rc._docker_start_commands() == []


# --- _native_start_commands ------------------------------------------------


def test_native_start_commands_windows(monkeypatch):
    monkeypatch.setattr(rc, "REDIS_URL", "")
    monkeypatch.setattr(rc, "REDIS_HOST", "localhost")
    monkeypatch.setattr(rc, "REDIS_PORT", 6379)
    monkeypatch.setattr(rc.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        rc.shutil,
        "which",
        lambda exe: f"C:\\{exe}" if exe in {"redis-server.exe", "wsl"} else None,
    )
    commands = rc._native_start_commands()
    executables = [cmd[0][0] for cmd in commands]
    assert "redis-server.exe" in executables
    assert "wsl" in executables
    # redis-server (no .exe) wasn't found by `which`, so it's absent.
    assert "redis-server" not in executables


def test_native_start_commands_macos_prefers_brew_and_binary(monkeypatch):
    monkeypatch.setattr(rc, "REDIS_URL", "")
    monkeypatch.setattr(rc, "REDIS_HOST", "localhost")
    monkeypatch.setattr(rc, "REDIS_PORT", 6379)
    monkeypatch.setattr(rc.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        rc.shutil,
        "which",
        lambda exe: f"/opt/{exe}" if exe in {"brew", "redis-server"} else None,
    )
    commands = rc._native_start_commands()
    assert commands[0][0] == ["brew", "services", "start", "redis"]
    assert commands[1][0][0] == "redis-server"


def test_native_start_commands_linux_systemctl_and_service(monkeypatch):
    monkeypatch.setattr(rc, "REDIS_URL", "")
    monkeypatch.setattr(rc, "REDIS_HOST", "localhost")
    monkeypatch.setattr(rc, "REDIS_PORT", 6379)
    monkeypatch.setattr(rc.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        rc.shutil,
        "which",
        lambda exe: (
            f"/usr/bin/{exe}"
            if exe in {"systemctl", "service", "redis-server"}
            else None
        ),
    )
    commands = rc._native_start_commands()
    flat = [cmd[0] for cmd in commands]
    assert ["systemctl", "--user", "start", "redis"] in flat
    assert ["systemctl", "--user", "start", "redis-server"] in flat
    assert ["service", "redis-server", "start"] in flat
    assert ["service", "redis", "start"] in flat
    assert any(cmd[0] == "redis-server" for cmd in flat)


def test_native_start_commands_linux_no_tools_available(monkeypatch):
    monkeypatch.setattr(rc, "REDIS_URL", "")
    monkeypatch.setattr(rc, "REDIS_HOST", "localhost")
    monkeypatch.setattr(rc, "REDIS_PORT", 6379)
    monkeypatch.setattr(rc.platform, "system", lambda: "Linux")
    monkeypatch.setattr(rc.shutil, "which", lambda exe: None)
    assert rc._native_start_commands() == []


def test_native_start_commands_unknown_platform_tries_redis_server(monkeypatch):
    monkeypatch.setattr(rc, "REDIS_URL", "")
    monkeypatch.setattr(rc, "REDIS_HOST", "localhost")
    monkeypatch.setattr(rc, "REDIS_PORT", 6379)
    monkeypatch.setattr(rc.platform, "system", lambda: "FreeBSD")
    monkeypatch.setattr(
        rc.shutil,
        "which",
        lambda exe: "/usr/bin/redis-server" if exe == "redis-server" else None,
    )
    commands = rc._native_start_commands()
    assert commands and commands[0][0][0] == "redis-server"


def test_native_start_commands_uses_custom_port(monkeypatch):
    monkeypatch.setattr(rc, "REDIS_URL", "")
    monkeypatch.setattr(rc, "REDIS_HOST", "localhost")
    monkeypatch.setattr(rc, "REDIS_PORT", 6400)
    monkeypatch.setattr(rc.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        rc.shutil,
        "which",
        lambda exe: "/usr/bin/redis-server" if exe == "redis-server" else None,
    )
    commands = rc._native_start_commands()
    assert "6400" in commands[0][0]


# --- _default_start_commands (ordering + dedup) ---------------------------


def test_default_start_commands_prefers_docker_when_configured(monkeypatch):
    monkeypatch.setattr(rc, "PREFER_DOCKER_REDIS", True)
    monkeypatch.setattr(
        rc,
        "_docker_start_commands",
        lambda: [(["docker", "compose", "up", "-d", "redis"], "/proj")],
    )
    monkeypatch.setattr(
        rc,
        "_native_start_commands",
        lambda: [(["redis-server", "--port", "6379"], None)],
    )
    commands = rc._default_start_commands()
    assert commands[0][0] == ["docker", "compose", "up", "-d", "redis"]
    assert commands[1][0] == ["redis-server", "--port", "6379"]


def test_default_start_commands_prefers_native_when_docker_not_preferred(monkeypatch):
    monkeypatch.setattr(rc, "PREFER_DOCKER_REDIS", False)
    monkeypatch.setattr(
        rc,
        "_docker_start_commands",
        lambda: [(["docker", "compose", "up", "-d", "redis"], "/proj")],
    )
    monkeypatch.setattr(
        rc,
        "_native_start_commands",
        lambda: [(["redis-server", "--port", "6379"], None)],
    )
    commands = rc._default_start_commands()
    assert commands[0][0] == ["redis-server", "--port", "6379"]
    assert commands[1][0] == ["docker", "compose", "up", "-d", "redis"]


def test_default_start_commands_dedups_identical_entries(monkeypatch):
    monkeypatch.setattr(rc, "PREFER_DOCKER_REDIS", True)
    same_command = (["redis-server", "--port", "6379"], None)
    monkeypatch.setattr(rc, "_docker_start_commands", lambda: [same_command])
    monkeypatch.setattr(rc, "_native_start_commands", lambda: [same_command])
    commands = rc._default_start_commands()
    assert commands == [same_command]


def test_default_start_commands_empty_when_nothing_found(monkeypatch):
    monkeypatch.setattr(rc, "_docker_start_commands", list)
    monkeypatch.setattr(rc, "_native_start_commands", list)
    assert rc._default_start_commands() == []
