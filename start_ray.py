#!/usr/bin/env python3
# Copyright (c) 2026 VaultIQ Inc. All rights reserved.
#
# Proprietary and Confidential.
#
# This file and the source code contained herein are the exclusive property of
# VaultIQ Inc. The intellectual and technical concepts contained herein are
# proprietary to VaultIQ Inc. and are protected by copyright law, trade secret
# law, and applicable confidentiality obligations.
#
# Access to this source code is restricted to authorized employees, contractors,
# or agents of VaultIQ Inc. who are bound by written confidentiality,
# nondisclosure, employment, contractor, license, or other agreements expressly
# covering such access.
#
# Unauthorized use, copying, reproduction, modification, publication,
# distribution, disclosure, display, or transmission of this file, in whole or
# in part, is strictly prohibited without the prior written permission of
# VaultIQ Inc.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure of this source code, which includes confidential,
# proprietary, and/or trade secret information of VaultIQ Inc.

"""
Usage:
    start_ray.py [--userid USERID] [--ollama-base-url URL [URL ...]] [--venv VENV_PATH] [--join HOSTNAME] [--autoscaling CONFIG_PATH] [--daemon]

Three modes of operation:
- Standalone (no --join, no --autoscaling): Start Ray as a standalone server that can act as a cluster head
- Standalone with autoscaling (--autoscaling CONFIG_PATH): Start Ray head with autoscaling configuration
- Worker (with --join): Join an existing Ray cluster running on the specified head node

Memory allocation formula:
    available = system_memory - CPU_RAM_RESERVED_FOR_OS_GB
                - (CPU_RAM_RESERVED_FOR_MARKER_PER_GPU_GB × num_gpus)
                - (CPU_RAM_RESERVED_FOR_OLLAMA_PER_GPU_GB × num_gpus)
    ray_memory = available × CPU_RAM_RAY_PERCENT_FROM_REMAINING

On shared hosts (e.g. aws), you can masquerade as other developers via --userid

Use --daemon flag when running under systemd to keep the process alive and monitor Ray health
"""

import argparse
import math
import os
import sys
import subprocess
import time
import signal
import socket
import platform
import getpass
import json
import urllib.parse
from pathlib import Path
from typing import Optional, Callable
import stat
import shutil
import re
import shlex

try:
    import resource
except ImportError:  # pragma: no cover - resource is unavailable on Windows.
    resource = None

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    # Ensure local imports work even when run outside the v3 directory.
    sys.path.insert(0, str(THIS_DIR))

from lifecycle.userid_override import apply_userid_override, resolve_userid_override  # noqa: E402

# Keep userid override ahead of any project_settings import so profile
# resolution matches the other startup scripts.
apply_userid_override(THIS_DIR, sys.argv[1:])
from env_flags import enabled_env_name  # noqa: E402

from memory_allocation import (  # noqa: E402
    CPU_RAM_RESERVED_FOR_OS_GB,
    CPU_RAM_RESERVED_FOR_MARKER_PER_GPU_GB,
    CPU_RAM_RESERVED_FOR_OLLAMA_PER_GPU_GB,
    CPU_RAM_RAY_PERCENT_FROM_REMAINING,
    calculate_ray_reserved_gb,
    calculate_object_store_bytes,
)

from backend.utils.locality import get_local_addresses  # noqa: E402
from backend.utils import gpu_placement  # noqa: E402

MACOS_RAY_NOFILE_LIMIT = 65536


def _format_rlimit_value(value: int) -> str:
    if resource is not None and value == resource.RLIM_INFINITY:
        return "unlimited"
    return str(value)


def _ensure_macos_nofile_limit(target: int = MACOS_RAY_NOFILE_LIMIT) -> None:
    """Raise the macOS file-descriptor soft limit before Ray starts."""
    if platform.system() != "Darwin":
        return
    if resource is None:
        print("[!] macOS file descriptor limit not checked: resource module unavailable")
        return

    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (OSError, ValueError) as exc:
        print(f"[!] macOS file descriptor limit not checked: {exc}")
        return

    if soft >= target:
        print(f"[*] macOS file descriptor limit already high enough: {soft}")
        return

    requested_soft = target if hard == resource.RLIM_INFINITY else min(target, hard)
    if requested_soft <= soft:
        print(
            f"[!] macOS file descriptor soft limit is {soft}; hard limit "
            f"{_format_rlimit_value(hard)} prevents raising it to {target}. "
            f"Run: sudo launchctl limit maxfiles {target} unlimited"
        )
        return

    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (requested_soft, hard))
        new_soft, new_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (OSError, ValueError) as exc:
        print(
            f"[!] Could not raise macOS file descriptor limit from {soft} "
            f"to {target}: {exc}. Run: sudo launchctl limit maxfiles "
            f"{target} unlimited"
        )
        return

    print(f"[*] macOS file descriptor limit: raised soft limit from {soft} to {new_soft}")
    if new_soft < target:
        print(
            f"[!] macOS file descriptor soft limit is {new_soft}; hard limit "
            f"{_format_rlimit_value(new_hard)} prevents raising it to {target}. "
            f"Run: sudo launchctl limit maxfiles {target} unlimited"
        )


def _parse_http_host_port(value: str, default_port: int) -> tuple[str, int]:
    """Parse host/port from an HTTP URL or host:port string."""
    raw = str(value or "").strip()
    if not raw:
        return "127.0.0.1", default_port

    normalized = raw if "://" in raw else f"http://{raw}"
    parsed = urllib.parse.urlparse(normalized)
    host = (parsed.hostname or "").strip() or "127.0.0.1"
    port = parsed.port or default_port
    return host, port


def _probe_local_host(host: str) -> str:
    stripped = str(host or "").strip().lower()
    if stripped in {"", "0.0.0.0", "::", "[::]"}:
        return "127.0.0.1"
    return stripped


def _resolve_ray_dashboard_host() -> str:
    """Resolve the Ray dashboard bind address.

    Fix for GHSA-gx77-xgc2-4888 (CVE-2025-34351): the Ray dashboard/Jobs API
    has no token auth, so it must bind to loopback unconditionally. There is
    no env override; remote dashboard access requires the separate token-auth
    rollout (RAY_AUTH_MODE), tracked separately.
    """
    return "127.0.0.1"


def _localhost_has_ipv4() -> bool:
    """True if "localhost" resolves to IPv4 the way Ray resolves it (AI_ADDRCONFIG)."""
    try:
        return bool(socket.getaddrinfo(
            "localhost", None, socket.AF_INET, socket.SOCK_STREAM, 0, socket.AI_ADDRCONFIG
        ))
    except socket.gaierror:
        return False


def _format_http_base_url(host: str, port: int) -> str:
    host = _probe_local_host(host)
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{int(port)}"


def _split_csv_urls(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _is_localish_host(host: str) -> bool:
    stripped = str(host or "").strip().lower()
    if stripped in {"", "0.0.0.0", "::", "[::]"}:
        return True
    return stripped in get_local_addresses()


def _is_localish_url(value: str, default_port: int) -> bool:
    host, _ = _parse_http_host_port(value, default_port)
    return _is_localish_host(host)


def _iter_listening_ports_for_matching_processes(
    matcher: Callable[[str, list[str]], bool],
) -> list[tuple[str, int]]:
    """Return listening host/port pairs for processes matching *matcher*."""
    try:
        import psutil  # type: ignore
    except ImportError:
        return []

    listeners: set[tuple[str, int]] = set()
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            info = proc.info or {}
            name = str(info.get("name") or "")
            cmdline = [str(part) for part in (info.get("cmdline") or [])]
            if not matcher(name, cmdline):
                continue

            net_connections = getattr(proc, "net_connections", None)
            if callable(net_connections):
                connections = net_connections(kind="inet")
            else:
                connections = proc.connections(kind="inet")  # type: ignore[attr-defined]
        except Exception:
            continue

        for conn in connections:
            try:
                status = getattr(conn, "status", "")
                if status != psutil.CONN_LISTEN:
                    continue
                laddr = getattr(conn, "laddr", None)
                if not laddr:
                    continue
                host = getattr(laddr, "ip", None) or laddr[0]
                port = getattr(laddr, "port", None) or laddr[1]
                if host and port:
                    listeners.add((str(host), int(port)))
            except Exception:
                continue

    return sorted(listeners, key=lambda item: (item[1], item[0]))


def _is_ollama_process(name: str, cmdline: list[str]) -> bool:
    lowered_name = str(name or "").strip().lower()
    joined = " ".join(cmdline).strip().lower()
    return lowered_name == "ollama" or ("ollama" in joined and "serve" in joined)


def _is_torch_runtime_process(name: str, cmdline: list[str]) -> bool:
    joined = " ".join(cmdline).strip().lower()
    return "backend.torch_runtime.service" in joined


def _validate_runtime_urls(flag_name: str, urls: list[str]) -> list[str]:
    """Reject malformed URLs from CLI overrides with a clear actionable error.

    Catches the common bash footgun of passing `[url1 url2]` thinking it is a
    list — the brackets become literal characters in the URL, which then breaks
    httpx with a confusing "missing protocol" error far downstream.
    """
    cleaned: list[str] = []
    for raw in urls:
        candidate = (raw or "").strip().rstrip("/")
        if not candidate:
            print(
                f"[ERROR] {flag_name} got an empty URL value. Pass space-separated "
                f"URLs without brackets, e.g.\n"
                f"           {flag_name} http://host1:11434 http://host2:11434"
            )
            sys.exit(2)
        if candidate[0] in "[]" or candidate[-1] in "[]":
            print(
                f"[ERROR] {flag_name} value {raw!r} contains stray bracket "
                f"characters. The flag uses nargs='+' (space-separated), not a "
                f"shell list. Use:\n"
                f"           {flag_name} http://host1:11434 http://host2:11434"
            )
            sys.exit(2)
        if not (candidate.startswith("http://") or candidate.startswith("https://")):
            print(
                f"[ERROR] {flag_name} value {raw!r} is missing the 'http://' or "
                f"'https://' scheme."
            )
            sys.exit(2)
        cleaned.append(candidate)
    return cleaned


def _resolve_runtime_settings():
    """Resolve the project settings object at call time.

    Kept as a module-level seam so tests can patch the exact object
    ``_maybe_bind_local_runtime_urls`` reads, instead of patching a
    separately imported ``project_settings`` copy whose module identity
    may differ from the one this module resolves.
    """
    from project_settings import settings

    return settings


def _publish_runtime_env(env_var: str, value: str) -> None:
    """Publish a discovered runtime endpoint to the process environment.

    Routed through a single module-level seam so a test can intercept the
    writes and assert on them without leaking into the real ``os.environ``.
    """
    os.environ[env_var] = value


class RayStartupManager:
    def __init__(
        self,
        userid: Optional[str],
        ollama_base_urls: Optional[list[str]],
        venv: Optional[str],
        join_head: Optional[str] = None,
        autoscaling_config: Optional[str] = None,
        daemon: bool = False,
        torch_base_urls: Optional[list[str]] = None,
    ):
        self.userid = self._resolve_userid(userid)
        self.venv = venv
        self.join_head = join_head  # If provided, we're in worker mode
        self.autoscaling_config = autoscaling_config  # Path to autoscaling config file
        self.daemon = daemon  # Run in daemon mode for systemd
        self.ollama_base_urls = ollama_base_urls  # None = local, set = remote override(s)
        self.torch_base_urls = torch_base_urls  # None = local, set = remote override(s)

        # Set developer userid for project_settings to pick up
        os.environ['VAULTIQ_DEVELOPER_USERID'] = self.userid
        if ollama_base_urls:
            normalized = _validate_runtime_urls("--ollama-base-url", ollama_base_urls)
            os.environ["OLLAMA_BASE_URL"] = ",".join(normalized)
            if len(normalized) == 1:
                print(f"[*] Using Ollama base URL override: {normalized[0]}")
            else:
                print(f"[*] Using Ollama base URL overrides: {', '.join(normalized)}")
        if torch_base_urls:
            normalized = _validate_runtime_urls("--torch-base-url", torch_base_urls)
            os.environ["TORCH_RUNTIME_BASE_URL"] = ",".join(normalized)
            if len(normalized) == 1:
                print(f"[*] Using torch runtime base URL override: {normalized[0]}")
            else:
                print(f"[*] Using torch runtime base URL overrides: {', '.join(normalized)}")

        # Now project_settings will load the correct configuration
        self._load_settings()
        self._setup_environment()

    def _resolve_userid(self, userid: Optional[str]) -> str:
        """Resolve developer userid from CLI, env, local settings, or login name."""
        resolved_userid = resolve_userid_override(
            THIS_DIR,
            explicit_userid=userid,
            env=os.environ,
        )
        if resolved_userid:
            return resolved_userid

        return self._get_userid()

    def _get_userid(self) -> str:
        """Get the userid from the OS login, requiring an override for ec2-user."""
        current_user = getpass.getuser()

        if current_user == 'ec2-user':
            print("[ERROR] Running as ec2-user requires --userid or VAULTIQ_DEVELOPER_USERID")
            print("Usage: start_ray.py --userid <userid>")
            sys.exit(1)

        return current_user

    def _load_settings(self):
        """Load settings from project_settings module"""
        # These should be available from project_settings after setting VAULTIQ_DEVELOPER_USERID
        from project_settings import settings
        from shared.ray_connection import cluster_mode, parse_gcs_address, shared_gcs_address
        self.ray_cluster_mode = cluster_mode(settings)
        explicit_join = self.join_head and ":" in self.join_head
        if explicit_join:
            parse_gcs_address(self.join_head)
        self.ray_address = (
            shared_gcs_address(settings)
            if self.ray_cluster_mode == "shared" and not explicit_join else ""
        )
        self.ray_gcs_port = settings.RAY_GCS_PORT
        self.ray_dashboard_port = settings.RAY_DASHBOARD_PORT
        self.ray_client_server_port = settings.RAY_CLIENT_SERVER_PORT
        self.ray_runtime_env_agent_port = settings.RAY_RUNTIME_ENV_AGENT_PORT
        self.ray_dashboard_agent_grpc_port = settings.RAY_DASHBOARD_AGENT_GRPC_PORT
        self.ray_dashboard_agent_listenport = settings.RAY_DASHBOARD_AGENT_LISTENPORT
        self.ray_namespace = settings.RAY_NAMESPACE
        self.ray_tmpdir = settings.RAY_TMPDIR
        #self.ray_head_node_name = settings.RAY_HEAD_NODE_NAME

        print(f"[*] Loaded configuration for user: {self.userid}")
        print(f"    GCS Port: {self.ray_gcs_port}")
        print(f"    Dashboard Port: {self.ray_dashboard_port}")
        print(f"    Namespace: {self.ray_namespace}")

    def _setup_environment(self):
        """Set up environment variables for Ray"""
        os.environ['RAY_DEDUP_LOGS'] = '0'
        os.environ['GRPC_KEEPALIVE_TIME_MS'] = '10000'
        os.environ['GRPC_KEEPALIVE_TIMEOUT_MS'] = '5000'
        os.environ['RAY_USAGE_STATS_ENABLED'] = '0'
        os.environ['RAY_memory_monitor_refresh_ms'] = '250'
        os.environ['APP_LAUNCH_TIME'] = str(int(time.time()))

        # Set up tessdata based on OS
        system = platform.system()
        if system == 'Linux':
            os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/4.00/tessdata'
        elif system == 'Darwin':
            os.environ['TESSDATA_PREFIX'] = '/usr/local/share/tessdata'

        # Set up CUDA - if required
        gpu_count = self.get_gpu_count()
        if system == "Darwin" or gpu_count == 0:
            return

        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        print(f"[*] Detected {gpu_count} GPU(s)")

    def _is_ollama_service_running(self, host: str, port: int) -> bool:
        """Check if the Ollama service is reachable."""
        import urllib.request
        try:
            probe_host = _probe_local_host(host)
            url = f"http://{probe_host}:{port}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _is_torch_service_running(self, host: str, port: int) -> bool:
        """Check if the torch runtime service is reachable."""
        import urllib.request
        try:
            probe_host = _probe_local_host(host)
            url = f"http://{probe_host}:{port}/api/health"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _discover_local_service_base_url(
        self,
        *,
        candidate_urls: list[str],
        default_port: int,
        healthcheck: Callable[[str, int], bool],
        process_matcher: Callable[[str, list[str]], bool],
    ) -> Optional[str]:
        seen: set[tuple[str, int]] = set()

        for candidate in candidate_urls:
            if not candidate:
                continue
            host, port = _parse_http_host_port(candidate, default_port)
            probe_key = (_probe_local_host(host), int(port))
            if probe_key in seen:
                continue
            seen.add(probe_key)
            if healthcheck(host, port):
                return _format_http_base_url(host, port)

        for host, port in _iter_listening_ports_for_matching_processes(process_matcher):
            probe_key = (_probe_local_host(host), int(port))
            if probe_key in seen:
                continue
            seen.add(probe_key)
            if healthcheck(host, port):
                return _format_http_base_url(host, port)

        return None

    def _set_local_runtime_metadata(self, env_var: str, discovered_base_url: str):
        host, port = _parse_http_host_port(discovered_base_url, 0)
        normalized = _format_http_base_url(host, port)
        _publish_runtime_env(env_var, normalized)
        if env_var == "OLLAMA_BASE_URL":
            _publish_runtime_env("VAULTIQ_LOCAL_OLLAMA_PORT", str(port))
        elif env_var == "TORCH_RUNTIME_BASE_URL":
            _publish_runtime_env("VAULTIQ_LOCAL_TORCH_PORT", str(port))
        return normalized, port

    def _maybe_bind_local_runtime_urls(self):
        """Bind this Ray node to already-running local runtime services.

        This is intentionally node-local only. It lets Ray workers and actors on
        the current node inherit the correct HTTP endpoint when the runtime was
        started manually on a non-default local port.
        """
        settings = _resolve_runtime_settings()

        if not self.ollama_base_urls:
            configured_ollama_urls: list[str] = []
            configured_ollama_base = str(getattr(settings, "OLLAMA_BASE_URL", "") or "").strip()
            configured_ollama_urls.extend(_split_csv_urls(configured_ollama_base))
            ollama_host_env = str(os.getenv("OLLAMA_HOST", "") or "").strip()
            if ollama_host_env:
                configured_ollama_urls.append(ollama_host_env)
            ollama_server = str(getattr(settings, "OLLAMA_SERVER", "127.0.0.1")).strip() or "127.0.0.1"
            ollama_port = int(getattr(settings, "OLLAMA_PORT", 11434))
            configured_ollama_urls.append(f"http://{ollama_server}:{ollama_port}")

            if configured_ollama_urls and not _is_localish_url(configured_ollama_urls[0], 11434):
                print(f"[*] Ollama: keeping configured remote service at {configured_ollama_urls[0].rstrip('/')}")
            else:
                discovered = self._discover_local_service_base_url(
                    candidate_urls=configured_ollama_urls,
                    default_port=11434,
                    healthcheck=self._is_ollama_service_running,
                    process_matcher=_is_ollama_process,
                )
                if discovered:
                    normalized, port = self._set_local_runtime_metadata("OLLAMA_BASE_URL", discovered)
                    print(f"[*] Ollama: discovered local service at {normalized}")
                    print(f"    Published VAULTIQ_LOCAL_OLLAMA_PORT={port}")
                else:
                    print("[*] Ollama: no healthy local service discovered; using existing configuration")

        if not self.torch_base_urls:
            configured_torch_urls: list[str] = []
            configured_torch_base = str(getattr(settings, "TORCH_RUNTIME_BASE_URL", "") or "").strip()
            configured_torch_urls.extend(_split_csv_urls(configured_torch_base))
            torch_host = str(getattr(settings, "TORCH_RUNTIME_HOST", "127.0.0.1")).strip() or "127.0.0.1"
            torch_port = int(getattr(settings, "TORCH_RUNTIME_PORT", 11436))
            configured_torch_urls.append(f"http://{torch_host}:{torch_port}")

            if configured_torch_urls and not _is_localish_url(configured_torch_urls[0], 11436):
                print(f"[*] Torch runtime: keeping configured remote service at {configured_torch_urls[0].rstrip('/')}")
            else:
                discovered = self._discover_local_service_base_url(
                    candidate_urls=configured_torch_urls,
                    default_port=11436,
                    healthcheck=self._is_torch_service_running,
                    process_matcher=_is_torch_runtime_process,
                )
                if discovered:
                    normalized, port = self._set_local_runtime_metadata("TORCH_RUNTIME_BASE_URL", discovered)
                    print(f"[*] Torch runtime: discovered local service at {normalized}")
                    print(f"    Published VAULTIQ_LOCAL_TORCH_PORT={port}")
                else:
                    print("[*] Torch runtime: no healthy local service discovered; using existing configuration")


    def get_cpu_count(self) -> int:
        """Get the number of CPU cores available on the system"""
        try:
            import psutil
            cpu_count = psutil.cpu_count(logical=True)
            if not cpu_count:
                raise ValueError("Unable to determine CPU count")
            return cpu_count
        except ImportError:
            print("[ERROR] psutil is not installed. Please install it with: pip install psutil")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Failed to determine CPU count: {e}")
            sys.exit(1)

    def get_memory_gb(self) -> int:
        """Get the total system memory in GB"""
        try:
            import psutil
            mem_bytes = psutil.virtual_memory().total
            return mem_bytes // (1024 * 1024 * 1024)
        except ImportError:
            print("[ERROR] psutil is not installed. Please install it with: pip install psutil")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Failed to determine memory size: {e}")
            sys.exit(1)

    def get_gpu_count(self) -> int:
        """Get the number of GPUs available on the system"""
        system = platform.system()

        # macOS always returns 1 GPU (Metal support)
        if system == 'Darwin':
            print("[*] macOS detected, assuming 1 GPU (Metal)")
            return 1

        visible_cuda_devices = self._cuda_visible_devices()

        # For Linux/Windows, check for NVIDIA GPUs
        primary_probe_succeeded = False
        physical_gpu_count = 0
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                primary_probe_succeeded = True
                gpu_lines = result.stdout.strip().split('\n')
                physical_gpu_count = len([line for line in gpu_lines if line.strip()])
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # nvidia-smi not found or timed out
            pass

        has_uuid_mask = bool(
            visible_cuda_devices
            and any(not device.isdigit() for device in visible_cuda_devices)
        )
        if has_uuid_mask:
            # A failed summary query cannot establish the physical GPU count
            # needed to validate UUIDs. The status-aware inventory probe keeps
            # a verified empty host distinct from an unavailable probe.
            inventory, probed = gpu_placement._gpu_inventory_with_status()
            if not probed:
                raise gpu_placement.GpuInventoryUnavailableError(
                    "CUDA_VISIBLE_DEVICES contains a GPU UUID, but GPU "
                    "hardware inventory could not be probed (both NVML "
                    "and the nvidia-smi fallback failed)."
                )
            if not primary_probe_succeeded:
                physical_gpu_count = len(inventory)
            visible_gpu_count = self._count_valid_visible_cuda_devices(
                visible_cuda_devices,
                physical_gpu_count,
                inventory,
            )
            print(
                f"[*] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} "
                f"exposes {visible_gpu_count}/{physical_gpu_count} NVIDIA GPU(s)"
            )
            return visible_gpu_count

        if primary_probe_succeeded:
            if visible_cuda_devices is not None:
                visible_gpu_count = self._count_valid_visible_cuda_devices(
                    visible_cuda_devices,
                    physical_gpu_count,
                )
                print(
                    f"[*] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} "
                    f"exposes {visible_gpu_count}/{physical_gpu_count} NVIDIA GPU(s)"
                )
                return visible_gpu_count
            if physical_gpu_count > 0:
                print(f"[*] Detected {physical_gpu_count} NVIDIA GPU(s)")
            return physical_gpu_count

        # No GPUs detected
        return 0

    @staticmethod
    def _cuda_visible_devices() -> Optional[list[str]]:
        value = os.environ.get("CUDA_VISIBLE_DEVICES")
        if value is None:
            return None
        value = value.strip()
        if not value or value == "-1":
            return []
        return [device.strip() for device in value.split(",") if device.strip()]

    @staticmethod
    def _count_valid_visible_cuda_devices(
        devices: list[str],
        physical_gpu_count: int,
        inventory: Optional[list] = None,
    ) -> int:
        if physical_gpu_count <= 0:
            return 0
        known_uuids = {device.uuid for device in (inventory or [])}
        count = 0
        for device in devices:
            if device.isdigit():
                if int(device) < physical_gpu_count:
                    count += 1
            elif device in known_uuids:
                count += 1
            else:
                print(
                    f"[!] Non-numeric CUDA_VISIBLE_DEVICES token {device!r} ignored; "
                    "only integer GPU indices or known GPU UUIDs are supported."
                )
        return count


    def _activate_venv(self):
        """Ensure this script runs under the VaultIQ venv interpreter."""
        if self.venv:
            venv_path = Path(self.venv)
        else:
            # Try to get from project_settings
            try:
                from project_settings import settings
                venv_path = Path(settings.VAULTIQ_VENV)
                print(f"[*] Using VaultIQ default venv: {venv_path}")
            except (ImportError, AttributeError):
                print("[ERROR] Not in a virtual environment and cannot find default")
                sys.exit(1)

        venv_python = venv_path / "bin" / "python"
        if not venv_python.exists():
            print(f"[ERROR] Cannot activate virtual environment at {venv_path}")
            sys.exit(1)

        current_python = Path(sys.executable).resolve()
        target_python = venv_python.resolve()

        # Running inside the right interpreter is mandatory for native modules
        # such as Ray's _raylet shared library.
        if current_python != target_python:
            print(f"[*] Re-launching with venv python: {target_python}")
            os.environ["VIRTUAL_ENV"] = str(venv_path)
            os.execv(str(target_python), [str(target_python), str(Path(__file__).resolve()), *sys.argv[1:]])

        print(f"[✓] Using virtual environment python: {current_python}")

    def check_ray_status(self) -> bool:
        """
        Check if Ray is installed and running
        Returns True if Ray is running, False otherwise
        """
        # Check if Ray is installed
        try:
            import ray
            print(f"[✓] Ray version {ray.__version__} found")
        except ImportError:
            print("[ERROR] Ray is not installed in this environment")
            sys.exit(1)

        # Check if Ray is running by checking the GCS port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', int(self.ray_gcs_port))) == 0

    def start_standalone(self):
        """Start Ray in standalone mode (can act as cluster head)"""
        if getattr(self, "ray_cluster_mode", "local") == "shared":
            from shared.ray_connection import parse_gcs_address

            _, configured_port = parse_gcs_address(self.ray_address)
            if configured_port != int(self.ray_gcs_port):
                raise ValueError(
                    f"Shared Ray RAY_ADDRESS port {configured_port} does not match "
                    f"RAY_GCS_PORT {self.ray_gcs_port}"
                )
        mode_desc = "STANDALONE mode with AUTOSCALING" if self.autoscaling_config else "STANDALONE mode"
        print(f"[🚀] Starting Ray in {mode_desc}")
        print("[*] This node can accept worker connections")

        # Get system resources
        cpu_count = self.get_cpu_count()
        memory_gb = self.get_memory_gb()
        # Validate placement before any legacy GPU probe. In particular, an
        # invalid Ray GPU claim must fail without touching NVML/nvidia-smi.
        static_plan = gpu_placement.resolve_configured_plan()
        gpu_count = (
            self.get_gpu_count()
            if static_plan is None
            else sum(len(runtime.uuids) for runtime in static_plan.runtimes.values())
        )

        # Under any declared static placement plan, advertise only the
        # disjoint ray_gpus set to Ray (which may legitimately be empty) and
        # keep the host RAM allowance tied to the GPUs locally-managed
        # runtimes actually own. With no declared plan (the default), both
        # counts stay equal to gpu_count exactly as before.
        ray_gpu_count = 0
        runtime_allowance_gpu_count = gpu_count
        # Ray itself is CPU-only. Keep physical/runtime GPU inventory and RAM
        # allowances independent from Ray's resource advertisement.
        if static_plan is not None:
            runtime_allowance_gpu_count = sum(
                len(runtime.uuids) for runtime in static_plan.runtimes.values()
            )
            print(
                "[*] Ray GPU pool: advertising 0 GPUs; Ray is CPU-only in Phase A"
            )

        # Allocate all cores up to 16, otherwise cap Ray at 16 cores
        ray_cores = cpu_count if cpu_count <= 16 else 16

        # Calculate Ray memory using shared formula
        ray_memory_gb = calculate_ray_reserved_gb(
            memory_gb, ray_gpu_count, runtime_allowance_gpu_count
        )
        ray_memory_bytes = ray_memory_gb * 1024 * 1024 * 1024
        system_name = platform.system()
        host_name = socket.gethostname()
        object_store_bytes = calculate_object_store_bytes(memory_gb, system_name, host_name)
        marker_reserve = CPU_RAM_RESERVED_FOR_MARKER_PER_GPU_GB * runtime_allowance_gpu_count
        ollama_reserve = CPU_RAM_RESERVED_FOR_OLLAMA_PER_GPU_GB * runtime_allowance_gpu_count
        total_reserve = CPU_RAM_RESERVED_FOR_OS_GB + marker_reserve + ollama_reserve
        available_gb = memory_gb - total_reserve
        if available_gb <= 0:
            print("[ERROR] Not enough memory!")
            print(f"        System: {memory_gb} GB")
            print(f"        OS reserve: {CPU_RAM_RESERVED_FOR_OS_GB} GB")
            print(
                f"        Marker reserve: {marker_reserve} GB "
                f"({CPU_RAM_RESERVED_FOR_MARKER_PER_GPU_GB} GB × {runtime_allowance_gpu_count} GPUs)"
            )
            print(
                f"        Ollama reserve: {ollama_reserve} GB "
                f"({CPU_RAM_RESERVED_FOR_OLLAMA_PER_GPU_GB} GB × {runtime_allowance_gpu_count} GPUs)"
            )
            print(f"        Total reserve: {total_reserve} GB")
            print(f"        Available: {available_gb} GB")
            sys.exit(1)
        print("[*] Memory allocation:")
        print(f"        System memory: {memory_gb} GB")
        print(f"        OS reserve: {CPU_RAM_RESERVED_FOR_OS_GB} GB")
        print(
            f"        Marker reserve: {marker_reserve} GB "
            f"({CPU_RAM_RESERVED_FOR_MARKER_PER_GPU_GB} GB × {runtime_allowance_gpu_count} GPUs)"
        )
        print(
            f"        Ollama reserve: {ollama_reserve} GB "
            f"({CPU_RAM_RESERVED_FOR_OLLAMA_PER_GPU_GB} GB × {runtime_allowance_gpu_count} GPUs)"
        )
        print(f"        Available after reserves: {available_gb} GB")
        print(f"        Ray allocation ({CPU_RAM_RAY_PERCENT_FROM_REMAINING}%): {ray_memory_gb} GB")

        print(f"[*] Allocating {ray_cores}/{cpu_count} cores, {ray_memory_gb}/{memory_gb} GB and {ray_gpu_count} GPUs to Ray")
        if object_store_bytes is not None:
            object_store_gb = object_store_bytes // (1024 * 1024 * 1024)
            if system_name == "Darwin":
                reason = "explicit conservative override for macOS"
            elif system_name == "Linux" and host_name.startswith("vaultiq-dev"):
                reason = "explicit conservative override for shared dev host"
            else:
                reason = f"explicit override for {system_name}"
            print(f"[*] Ray object store: {object_store_gb} GB ({reason})")
        else:
            print(f"[*] Ray object store: Ray default ({system_name})")

        # Get Tailscale IP for cluster connectivity
        try:
            tailscale_ip = subprocess.check_output(
                ['tailscale', 'ip', '-4'],
                text=True, stderr=subprocess.DEVNULL
            ).strip()
            node_ip = tailscale_ip
            print(f"[*] Using Tailscale IP for cluster connectivity: {tailscale_ip}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fall back to localhost if Tailscale not available
            node_ip = '127.0.0.1'
            print("[*] Tailscale not available, using localhost")

        # Create temp directory if needed
        os.makedirs(self.ray_tmpdir, exist_ok=True)

        # Make it group-owned, children inherit group (setgid), and prevent cross-deletes (sticky)
        skip_group_setup_env = enabled_env_name(
            "VAULTIQ_SKIP_SHARED_GROUP_SETUP",
        )
        if skip_group_setup_env:
            print(
                "[*] Skipping shared group ownership setup for "
                f"{self.ray_tmpdir} because {skip_group_setup_env}=1"
            )
        else:
            try:
                shutil.chown(self.ray_tmpdir, group="vaultiq")
                os.chmod(self.ray_tmpdir, stat.S_ISGID | stat.S_ISVTX | 0o770)  # 0o3770
            except PermissionError:
                print(f"Insufficient permissions to set vaultiq group ownership on {self.ray_tmpdir}")
            except LookupError:
                # when we run on a local laptop, the group may not exist. Ignore that error
                pass

        os.umask(0o002)

        ray_resources = {"head_node": 1}
        if not self.autoscaling_config:
            # In standalone/dev mode the single local head must also be able to
            # run indexing tasks. Autoscaled clusters keep indexing-worker off
            # the head so those tasks land only on worker node types.
            ray_resources["indexing-worker"] = 1
            print("[*] Standalone mode: advertising indexing-worker resource on the local head node")

        cmd = [
            'ray', 'start', '--head',
            '--port', str(self.ray_gcs_port),
            '--node-ip-address', node_ip,
            '--dashboard-port', str(self.ray_dashboard_port),
            '--dashboard-host', _resolve_ray_dashboard_host(),
            '--ray-client-server-port', str(self.ray_client_server_port),
            '--runtime-env-agent-port', str(self.ray_runtime_env_agent_port),
            '--include-dashboard', 'true',
            '--dashboard-agent-grpc-port', str(self.ray_dashboard_agent_grpc_port),
            '--dashboard-agent-listen-port', str(self.ray_dashboard_agent_listenport),
            '--temp-dir', self.ray_tmpdir,
            '--num-cpus', str(ray_cores),
            '--memory', str(ray_memory_bytes),
            '--num-gpus', str(ray_gpu_count),
            '--resources', json.dumps(ray_resources),  # head_node pins DB updates; standalone dev also doubles as indexing worker
        ]

        if object_store_bytes is not None:
            cmd.extend(['--object-store-memory', str(object_store_bytes)])

        # Add autoscaling config if provided
        if self.autoscaling_config:
            if not os.path.exists(self.autoscaling_config):
                print(f"[ERROR] Autoscaling config not found at {self.autoscaling_config}")
                sys.exit(1)
            cmd.extend(['--autoscaling-config', self.autoscaling_config])
            print(f"[*] Using autoscaling config: {self.autoscaling_config}")

        raylet_env = os.environ.copy()
        # Always hide GPUs from Ray child processes. This must be explicit,
        # including legacy/no-plan launches, so Ray cannot auto-discover them.
        raylet_env["CUDA_VISIBLE_DEVICES"] = ""
        if node_ip == '127.0.0.1' and not _localhost_has_ipv4():
            # With no non-loopback IPv4 address (e.g. a disconnected host), Ray
            # resolves localhost to ::1 and its IPv4-only metrics exporter fails
            # to bind, so the node never finishes starting.
            raylet_env["RAY_enable_metrics_collection"] = "0"
            print("[!] No IPv4 network address; starting Ray with metrics collection off")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=raylet_env,
            start_new_session=True,
        )
        if result.returncode != 0:
            print(f"[ERROR] Failed to start Ray: {result.stderr}")
            print("[!] Please check the error above and retry")
            sys.exit(1)

        print("[✓] Ray started successfully!")
        print(f"[↗] Dashboard: http://localhost:{self.ray_dashboard_port}")
        if node_ip != '127.0.0.1':
            print(f"[↗] Workers can join at: {node_ip}:{self.ray_gcs_port}")

    def _verify_existing_worker_zero_gpu(self) -> Optional[bool]:
        """Read this user's raylet command line for its advertised GPUs.

        ``True`` means every raylet advertises zero GPUs (Ray omits a zero GPU
        entry), ``False`` means one advertises GPUs, and ``None`` means the
        result is unknown (no raylet found, unreadable, or malformed). The read is
        time-bounded and never stops or modifies a process.
        """
        try:
            res = subprocess.run(
                ['pgrep', '-af', 'raylet', '-u', self.userid],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (subprocess.TimeoutExpired, OSError):
            return None
        # `pgrep -a` prints "<pid> <cmdline>". Keep only the real raylet
        # executable (argv[0] basename); the dashboard agent and workers carry
        # `--raylet-name=` arguments but are not the raylet.
        cmdlines = []
        for line in res.stdout.splitlines():
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            try:
                argv = shlex.split(parts[1])
            except ValueError:
                continue
            if argv and os.path.basename(argv[0]) == "raylet":
                cmdlines.append(parts[1])
        if res.returncode != 0 or not cmdlines:
            return None

        for line in cmdlines:
            quantities = [
                m.group(1) for m in re.finditer(r'--num-gpus(?:=|\s+)(\S+)', line)
            ]
            lists = list(re.finditer(r'--static_resource_list(?:=|\s+)(\S+)', line))
            for m in lists:
                tokens = m.group(1).split(",")
                if len(tokens) % 2:
                    return None
                quantities += [
                    qty for name, qty in zip(tokens[0::2], tokens[1::2])
                    if name == "GPU" or name.startswith("GPU:")
                ]
            # Ray omits zero-valued resources, so an absent GPU entry means
            # zero; a raylet with no resource arguments at all is unreadable.
            if not quantities and not lists:
                return None
            try:
                values = [float(q) for q in quantities]
            except ValueError:
                return None
            if any(not math.isfinite(v) or v < 0 for v in values):
                return None
            if any(v > 0 for v in values):
                return False
        return True

    def start_worker(self):
        """Start Ray in worker mode"""
        # Validate placement before checking for an existing worker.
        gpu_placement.resolve_configured_plan()
        # Check if worker is already running
        result = subprocess.run(['pgrep', '-f', 'ray::', '-u', self.userid], capture_output=True)
        if result.returncode == 0:
            worker_gpu_status = self._verify_existing_worker_zero_gpu()
            if worker_gpu_status is True:
                print("[✓] Ray worker is already running (verified CPU-only zero-GPU)")
                subprocess.run(['ray', 'status'])
                return
            if worker_gpu_status is False:
                print(
                    "[ERROR] Running Ray worker advertises GPUs, violating the "
                    "zero-GPU contract. Stop it manually (ray stop), then retry."
                )
            else:
                print(
                    "[ERROR] Could not verify that the running Ray worker "
                    "advertises zero GPUs. Inspect its raylet command line, "
                    "stop it manually if needed, then retry."
                )
            sys.exit(1)

        ray_address = self._worker_address()
        print(f"[🔗] Starting Ray in WORKER mode, connecting to: {ray_address}")

        cmd = ['ray', 'start', '--address', ray_address, '--temp-dir', self.ray_tmpdir, '--num-gpus', '0']
        worker_env = os.environ.copy()
        worker_env["CUDA_VISIBLE_DEVICES"] = ""
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=worker_env,
            start_new_session=True,
        )

        if result.returncode != 0:
            print(f"[ERROR] Failed to start Ray worker: {result.stderr}")
            sys.exit(1)

        print("[✓] Ray worker started successfully")
        from shared.ray_connection import parse_gcs_address
        head, _ = parse_gcs_address(ray_address)
        print(f"[↗] Dashboard available at: http://{head}:{self.ray_dashboard_port}")
        print("    (Dashboard is hosted on the head node)")

    def _worker_address(self) -> str:
        """Resolve an explicit join target without substituting a private shared port."""
        from shared.ray_connection import parse_gcs_address

        if ":" in self.join_head:
            parse_gcs_address(self.join_head)
            return self.join_head
        if getattr(self, "ray_cluster_mode", "local") == "shared":
            _, port = parse_gcs_address(self.ray_address)
        else:
            port = int(self.ray_gcs_port)
        return f"{self.join_head}:{port}"

    def _monitor_address(self) -> str:
        if self.join_head:
            return self._worker_address()
        if getattr(self, "ray_cluster_mode", "local") == "shared":
            return self.ray_address
        return f"127.0.0.1:{self.ray_gcs_port}"

    def _start_log_guard(self):
        """Keep Ray's unrotated GCS and C++ driver logs bounded."""
        guard_env = os.environ.copy()
        guard_env["VAULTIQ_RAY_LOG_GUARD_TMPDIR"] = str(self.ray_tmpdir)
        subprocess.Popen(
            [sys.executable, "-m", "lifecycle.ray_log_guard", "--userid", self.userid],
            cwd=str(THIS_DIR),
            env=guard_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    def monitor_loop(self):
        """Monitor Ray in a loop to keep service alive"""
        if not self.daemon:
            print("[*] Not in daemon mode, exiting normally")
            return

        print("[*] Running in daemon mode, starting monitoring loop...")

        # Set up signal handlers for graceful shutdown
        def signal_handler(sig, frame):
            print(f"\n[*] Received signal {sig}, shutting down Ray...")
            subprocess.run(['ray', 'stop', '--force'], capture_output=True)
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        while True:
            time.sleep(30)

            check_cmd = ['ray', 'status', '--address', self._monitor_address()]

            result = subprocess.run(check_cmd, capture_output=True)
            if result.returncode != 0:
                mode = "worker" if self.join_head else "standalone"
                print(f"[ERROR] Ray has crashed in {mode} mode, exiting for systemd restart...")
                sys.exit(1)

    def run(self):
        """Main entry point"""
        self._activate_venv()
        _ensure_macos_nofile_limit()
        # Discover and advertise already-running local runtimes to Ray workers.
        # TorchRuntime/AudioRuntime lifecycle now lives in the vaultiq CLI /
        # start_torch.py / start_audio.py; Ray only binds to what is already up.
        self._maybe_bind_local_runtime_urls()

        # Check Ray is installed (exits if not)
        self.check_ray_status()

        if self.join_head:
            # Worker mode - join existing cluster
            if self.autoscaling_config:
                print("[WARNING] --autoscaling flag is ignored in worker mode")
            self.start_worker()
        else:
            # Standalone mode - start as potential head
            self.start_standalone()

        self._start_log_guard()
        self.monitor_loop()


def main():
    parser = argparse.ArgumentParser(
        description='Ray startup script for VaultIQ',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start as standalone server (can act as cluster head)
  %(prog)s

  # Start as standalone server with autoscaling
  %(prog)s --autoscaling /home/vaultiq_user/ray_bootstrap_config.yaml

  # Start as standalone server with autoscaling in daemon mode (for systemd)
  %(prog)s --autoscaling /home/vaultiq_user/ray_bootstrap_config.yaml --daemon

  # Start as worker, joining cluster on dev2
  %(prog)s --join dev2

  # Specify custom virtual environment
  %(prog)s --venv /path/to/venv

  # On shared hosts, specify userid
  %(prog)s --userid myname --join dev1

  # Override Ollama base URL for Ray workers and OllamaProcessor actors
  %(prog)s --ollama-base-url http://100.91.31.15:11434

	Memory Allocation:
	  Edit the constants in memory_allocation.py to adjust memory allocation.
	  
	  Formula: ray_memory = (system - CPU_RAM_RESERVED_FOR_OS_GB
	                         - CPU_RAM_RESERVED_FOR_MARKER_PER_GPU_GB × num_gpus
	                         - CPU_RAM_RESERVED_FOR_OLLAMA_PER_GPU_GB × num_gpus) × CPU_RAM_RAY_PERCENT_FROM_REMAINING%%
	  
	  Current settings:
	    CPU_RAM_RESERVED_FOR_OS_GB = 3
	    CPU_RAM_RESERVED_FOR_MARKER_PER_GPU_GB = 2
	    CPU_RAM_RESERVED_FOR_OLLAMA_PER_GPU_GB = 3
	    CPU_RAM_RAY_PERCENT_FROM_REMAINING = 40
	  
	  Examples with current settings:
	    24GB, 1 GPU:  (24 - 3 - 2 - 3) × 0.40 = 6 GB for Ray
	    64GB, 1 GPU:  (64 - 3 - 2 - 3) × 0.40 = 22 GB for Ray
	    128GB, 2 GPU: (128 - 3 - 4 - 6) × 0.40 = 46 GB for Ray

	  Object store policy is handled separately:
	    macOS: always explicit 2 GB
	    Linux shared dev hosts (vaultiq-dev*): explicit 2 GB up to 16 GB RAM,
	      explicit 3 GB up to 32 GB RAM, otherwise Ray default
	    Other Linux hosts: Ray default
	        """
	    )

    parser.add_argument(
        '--userid',
        help='User ID (required for ec2-user)'
    )

    parser.add_argument(
        '--ollama-base-url',
        nargs='+',
        metavar='URL',
        help='Override Ollama base URL(s) for Ray workers and OllamaProcessor actors'
    )

    parser.add_argument(
        '--torch-base-url',
        nargs='+',
        metavar='URL',
        help='Override torch runtime base URL(s) for torch runtime services'
    )

    parser.add_argument(
        '--venv',
        help='Path to virtual environment (optional if already in venv)'
    )

    parser.add_argument(
        '--join',
        help='Head node hostname to join as worker (omit to start standalone)'
    )

    parser.add_argument(
        '--autoscaling',
        metavar='CONFIG_PATH',
        help='Path to Ray autoscaling config file (e.g., /home/vaultiq_user/ray_bootstrap_config.yaml)'
    )

    parser.add_argument(
        '--daemon',
        action='store_true',
        help='Run in daemon mode (keep process alive for systemd)'
    )

    args = parser.parse_args()

    try:
        manager = RayStartupManager(
            args.userid,
            args.ollama_base_url,
            args.venv,
            args.join,
            args.autoscaling,
            args.daemon,
            torch_base_urls=args.torch_base_url,
        )
        manager.run()
    except KeyboardInterrupt:
        print("\n[*] Shutting down...")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
