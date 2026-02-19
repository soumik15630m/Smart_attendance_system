import asyncio
import inspect
import os
import platform
import shlex
import shutil
import subprocess
from typing import Protocol
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from redis.asyncio import Redis

load_dotenv()

CACHE_BACKEND = os.getenv("CACHE_BACKEND", "auto").strip().lower()
REDIS_URL = os.getenv("REDIS_URL", "").strip()
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
UPSTASH_REDIS_REST_URL = os.getenv("UPSTASH_REDIS_REST_URL", "").strip()
UPSTASH_REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN", "").strip()
AUTO_START_LOCAL_REDIS = os.getenv(
    "AUTO_START_LOCAL_REDIS", "true"
).strip().lower() in {"1", "true", "yes", "on"}
LOCAL_REDIS_START_CMD = os.getenv("LOCAL_REDIS_START_CMD", "").strip()
LOCAL_REDIS_START_TIMEOUT_SECONDS = float(
    os.getenv("LOCAL_REDIS_START_TIMEOUT_SECONDS", "12")
)
PREFER_DOCKER_REDIS = os.getenv("PREFER_DOCKER_REDIS", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CacheClient(Protocol):
    async def get(self, key: str) -> str | None: ...

    async def setex(self, key: str, ttl_seconds: int, value: str) -> None: ...

    async def close(self) -> None: ...


class RedisTcpCache:
    def __init__(self, client: Redis):
        self.client = client

    async def get(self, key: str) -> str | None:
        return await self.client.get(key)

    async def setex(self, key: str, ttl_seconds: int, value: str) -> None:
        await self.client.setex(key, ttl_seconds, value)

    async def close(self) -> None:
        close_result = self.client.close()
        if inspect.isawaitable(close_result):
            await close_result


class UpstashRestCache:
    def __init__(self, rest_url: str, token: str):
        self.client = httpx.AsyncClient(
            base_url=rest_url.rstrip("/"),
            headers={"Authorization": f"Bearer {token}"},
            timeout=5.0,
        )

    async def _run(self, *command: str) -> object | None:
        response = await self.client.post("/", json=list(command))
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            if payload.get("error"):
                raise RuntimeError(str(payload["error"]))
            return payload.get("result")
        return None

    async def ping(self) -> None:
        result = await self._run("PING")
        if str(result).upper() != "PONG":
            raise RuntimeError("Upstash REST ping failed")

    async def get(self, key: str) -> str | None:
        result = await self._run("GET", key)
        if result is None:
            return None
        return str(result)

    async def setex(self, key: str, ttl_seconds: int, value: str) -> None:
        await self._run("SETEX", key, str(ttl_seconds), value)

    async def close(self) -> None:
        await self.client.aclose()


_cache_client: CacheClient | None = None
_cache_lock = asyncio.Lock()
_local_redis_start_attempted = False
_local_redis_start_lock = asyncio.Lock()


def _redis_host_port() -> tuple[str, int]:
    if REDIS_URL:
        parsed = urlparse(REDIS_URL)
        host = parsed.hostname or "localhost"
        port = parsed.port or 6379
        return host, port
    return REDIS_HOST, REDIS_PORT


def _is_local_redis_target() -> bool:
    host, _ = _redis_host_port()
    return host.strip("[]").lower() in {"localhost", "127.0.0.1", "::1"}


def _parse_custom_start_command() -> list[str] | None:
    if not LOCAL_REDIS_START_CMD:
        return None
    return shlex.split(LOCAL_REDIS_START_CMD, posix=os.name != "nt")


def _is_running_in_container() -> bool:
    if os.path.exists("/.dockerenv"):
        return True

    cgroup_path = "/proc/1/cgroup"
    if os.path.exists(cgroup_path):
        try:
            with open(cgroup_path, "r", encoding="utf-8") as cgroup_file:
                cgroup_text = cgroup_file.read().lower()
            return any(
                marker in cgroup_text
                for marker in ("docker", "containerd", "kubepods", "podman")
            )
        except OSError:
            return False
    return False


def _docker_start_commands() -> list[tuple[list[str], str | None]]:
    compose_path = os.path.join(PROJECT_ROOT, "docker-compose.yml")
    if not os.path.exists(compose_path):
        return []

    # Already in a container: skip docker CLI.
    if _is_running_in_container():
        return []

    commands: list[tuple[list[str], str | None]] = []
    if shutil.which("docker"):
        commands.append((["docker", "compose", "up", "-d", "redis"], PROJECT_ROOT))
    if shutil.which("docker-compose"):
        commands.append((["docker-compose", "up", "-d", "redis"], PROJECT_ROOT))
    return commands


def _native_start_commands() -> list[tuple[list[str], str | None]]:
    _, port = _redis_host_port()
    system_name = platform.system().lower()
    commands: list[tuple[list[str], str | None]] = []

    def add_redis_server(executable: str) -> None:
        if shutil.which(executable):
            commands.append(
                (
                    [
                        executable,
                        "--port",
                        str(port),
                        "--save",
                        "",
                        "--appendonly",
                        "no",
                    ],
                    None,
                )
            )

    if system_name == "windows":
        add_redis_server("redis-server.exe")
        add_redis_server("redis-server")
        if shutil.which("wsl"):
            commands.append(
                (
                    [
                        "wsl",
                        "redis-server",
                        "--port",
                        str(port),
                        "--save",
                        "",
                        "--appendonly",
                        "no",
                    ],
                    None,
                )
            )
    elif system_name == "darwin":
        if shutil.which("brew"):
            commands.append((["brew", "services", "start", "redis"], None))
        add_redis_server("redis-server")
    elif system_name == "linux":
        if shutil.which("systemctl"):
            commands.append((["systemctl", "--user", "start", "redis"], None))
            commands.append((["systemctl", "--user", "start", "redis-server"], None))
        if shutil.which("service"):
            commands.append((["service", "redis-server", "start"], None))
            commands.append((["service", "redis", "start"], None))
        add_redis_server("redis-server")
    else:
        add_redis_server("redis-server")

    return commands


def _default_start_commands() -> list[tuple[list[str], str | None]]:
    commands: list[tuple[list[str], str | None]] = []
    docker_commands = _docker_start_commands()
    native_commands = _native_start_commands()

    if PREFER_DOCKER_REDIS:
        commands.extend(docker_commands)
        commands.extend(native_commands)
    else:
        commands.extend(native_commands)
        commands.extend(docker_commands)

    # Keep order, drop dupes.
    seen: set[tuple[tuple[str, ...], str | None]] = set()
    deduped: list[tuple[list[str], str | None]] = []
    for command, cwd in commands:
        key = (tuple(command), cwd)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((command, cwd))
    return deduped


def _spawn_detached(command: list[str], cwd: str | None = None) -> None:
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
            subprocess, "DETACHED_PROCESS", 0
        )
        subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    else:
        subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )


async def _build_redis_cache() -> RedisTcpCache:
    redis = (
        Redis.from_url(REDIS_URL, decode_responses=True)
        if REDIS_URL
        else Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True,
        )
    )
    try:
        ping_result = redis.ping()
        if inspect.isawaitable(ping_result):
            await ping_result
        return RedisTcpCache(redis)
    except Exception:
        close_result = redis.close()
        if inspect.isawaitable(close_result):
            await close_result
        raise


async def _wait_for_redis_ready(timeout_seconds: float) -> bool:
    timeout = max(timeout_seconds, 1.0)
    deadline = asyncio.get_running_loop().time() + timeout

    while asyncio.get_running_loop().time() < deadline:
        try:
            cache = await _build_redis_cache()
            await cache.close()
            return True
        except Exception:
            await asyncio.sleep(0.5)

    return False


async def _start_local_redis_if_needed(original_error: Exception) -> None:
    if not AUTO_START_LOCAL_REDIS:
        raise RuntimeError(
            "Redis is unavailable and automatic local startup is disabled "
            "(AUTO_START_LOCAL_REDIS=false)."
        ) from original_error

    if not _is_local_redis_target():
        host, port = _redis_host_port()
        raise RuntimeError(
            "Redis is unavailable and target is not local "
            f"({host}:{port}), so automatic local startup is not possible."
        ) from original_error

    global _local_redis_start_attempted
    async with _local_redis_start_lock:
        if _local_redis_start_attempted:
            return
        _local_redis_start_attempted = True

    commands: list[tuple[list[str], str | None]] = []
    custom_command = _parse_custom_start_command()
    if custom_command:
        commands.append((custom_command, PROJECT_ROOT))
    commands.extend(_default_start_commands())

    print(
        "Redis auto-start detection: "
        f"os={platform.system()}, "
        f"inside_container={_is_running_in_container()}, "
        f"prefer_docker={PREFER_DOCKER_REDIS}"
    )

    if not commands:
        raise RuntimeError(
            "No local Redis startup command found for the detected environment. "
            "Install redis-server, Docker/Docker Compose, or set "
            "LOCAL_REDIS_START_CMD in .env."
        ) from original_error

    failures: list[str] = []
    for command, cwd in commands:
        try:
            _spawn_detached(command, cwd)
            if await _wait_for_redis_ready(LOCAL_REDIS_START_TIMEOUT_SECONDS):
                print(f"Local Redis started automatically: {' '.join(command)}")
                return
            failures.append(
                f"{' '.join(command)} (process launched but Redis not ready)"
            )
        except Exception as error:
            failures.append(f"{' '.join(command)} ({error})")

    joined = "; ".join(failures)
    raise RuntimeError(f"Failed to auto-start local Redis. Attempts: {joined}") from (
        original_error
    )


async def _build_cache_client() -> CacheClient:
    backend = CACHE_BACKEND
    if backend not in {"auto", "redis", "upstash_rest"}:
        backend = "auto"

    if backend in {"auto", "upstash_rest"}:
        if UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN:
            upstash_cache = UpstashRestCache(
                UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN
            )
            try:
                await upstash_cache.ping()
                print("Cache backend: Upstash REST")
                return upstash_cache
            except Exception as error:
                await upstash_cache.close()
                print(f"Upstash REST unavailable: {error}")
        elif backend == "upstash_rest":
            print(
                "Upstash REST selected but credentials are missing. "
                "Falling back to automatic cache detection."
            )

    if backend in {"auto", "redis", "upstash_rest"}:
        try:
            cache = await _build_redis_cache()
            print("Cache backend: Redis TCP")
            return cache
        except Exception as error:
            print(f"Redis unavailable: {error}")
            await _start_local_redis_if_needed(error)
            cache = await _build_redis_cache()
            print("Cache backend: Redis TCP (auto-started local)")
            return cache

    raise RuntimeError(f"Unsupported CACHE_BACKEND: {backend}")


async def init_cache() -> None:
    await get_cache_client()


async def shutdown_cache() -> None:
    global _cache_client
    async with _cache_lock:
        if _cache_client is not None:
            await _cache_client.close()
            _cache_client = None


async def get_cache_client() -> CacheClient:
    global _cache_client
    if _cache_client is not None:
        return _cache_client

    async with _cache_lock:
        if _cache_client is None:
            _cache_client = await _build_cache_client()
        return _cache_client


async def get_redis():
    cache = await get_cache_client()
    yield cache
