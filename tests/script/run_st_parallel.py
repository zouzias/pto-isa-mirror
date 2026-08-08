#!/usr/bin/env python3
# --------------------------------------------------------------------------------
# coding=utf-8
# A3 simple multi-NPU parallel smoke orchestrator.
# Standard library only. No NPU binaries are executed by this module directly.
# --------------------------------------------------------------------------------

import argparse
import dataclasses
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    import fcntl  # POSIX only
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class SmokeError(Exception):
    """Base error for the orchestrator."""


class DeviceDiscoveryError(SmokeError):
    """Raised when device allocation cannot be determined safely."""


class ManifestError(SmokeError):
    """Raised when the task manifest is malformed."""


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SmokeTask:
    index: int
    testcase: str
    gtest_filter: object  # Optional[str]
    debug_enable: bool


@dataclass(frozen=True)
class WorkerConfig:
    index: int
    physical_device: int
    build_dir: object  # Path
    log_path: object  # Path


@dataclass(frozen=True)
class TaskResult:
    task: SmokeTask
    worker_index: int
    physical_device: int
    returncode: int
    elapsed_ms: int


@dataclass(frozen=True)
class NpuStatus:
    device_id: int
    healthy: bool
    busy: bool


@dataclass
class DeviceLock:
    device_id: int
    fd: object  # file object
    path: str


# ---------------------------------------------------------------------------
# Task manifest parsing
# ---------------------------------------------------------------------------

def parse_task_manifest(manifest_path):
    """Parse the TSV manifest into an ordered list of SmokeTask.

    Contract per row: <testcase> TAB <gtest-filter-or-empty> TAB <debug-0-or-1>
    """
    tasks = []
    seen = set()
    with open(manifest_path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ManifestError(f"line {lineno}: expected 3 tab-separated fields, got {len(parts)}")
            testcase, gtest_filter, debug_str = parts
            if not testcase:
                raise ManifestError(f"line {lineno}: empty testcase")
            if debug_str not in ("0", "1"):
                raise ManifestError(f"line {lineno}: invalid debug value {debug_str!r}")
            debug_enable = debug_str == "1"
            triplet = (testcase, gtest_filter, debug_enable)
            if triplet in seen:
                raise ManifestError(f"line {lineno}: duplicate task triple {triplet!r}")
            seen.add(triplet)
            tasks.append(SmokeTask(
                index=len(tasks),
                testcase=testcase,
                gtest_filter=gtest_filter if gtest_filter else None,
                debug_enable=debug_enable,
            ))
    if not tasks:
        raise ManifestError("manifest is empty")
    return tasks


# ---------------------------------------------------------------------------
# Device list parsing
# ---------------------------------------------------------------------------

def parse_device_list(value):
    """Return unique non-negative integer IDs.

    Accepted: "0", "0,3", "3,5,7". Rejected: "0,", "0,0", "-1,3", "0,a".
    """
    if value is None:
        raise ValueError("value is None")
    tokens = value.split(",")
    if any(not t.strip() for t in tokens):
        raise ValueError("empty device entry")
    ids = []
    for tok in tokens:
        tok = tok.strip()
        if not re.fullmatch(r"\d+", tok):
            raise ValueError(f"non-numeric device id {tok!r}")
        ids.append(int(tok))
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate device ids")
    return ids


def parse_inherited_visible(value):
    """Parse ASCEND_RT_VISIBLE_DEVICES (comma or space separated). Empty -> []."""
    if value is None:
        return []
    ids = []
    for tok in re.split(r"[, ]+", value.strip()):
        if not tok:
            continue
        if not re.fullmatch(r"\d+", tok):
            raise ValueError(f"non-numeric visible device id {tok!r}")
        ids.append(int(tok))
    return ids


# ---------------------------------------------------------------------------
# npu-smi query
# ---------------------------------------------------------------------------

def query_npu_smi(timeout_seconds=10):
    """Run `npu-smi info`, parse device health and process ownership.

    Returns a list of NpuStatus. Raises DeviceDiscoveryError when the command
    is absent, times out, exits nonzero, or cannot be parsed confidently.
    """
    exe = shutil.which("npu-smi")
    if exe is None:
        raise DeviceDiscoveryError("npu-smi is unavailable")
    try:
        proc = subprocess.run(
            [exe, "info"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as e:
        raise DeviceDiscoveryError("npu-smi timed out") from e
    except OSError as e:
        raise DeviceDiscoveryError(f"npu-smi failed to start: {e}") from e
    if proc.returncode != 0:
        raise DeviceDiscoveryError(f"npu-smi exited nonzero: {proc.returncode}")
    return parse_npu_smi(proc.stdout, proc.stderr)


def parse_npu_smi(stdout, stderr=""):
    """Parse the observed 910B4-style `npu-smi info` output.

    Rules:
      - a device row looks like:  | 2     910B4               | OK ...
      - the same block then lists a chip line: | 0 | 0000:C2:00.0 | ...
      - a device is busy if a numeric process row associates a pid with it:
        | 0 | 12345 | ...  (in the Process table).
      - "No running processes found in NPU N" means that device is idle.
      - non-contiguous ids such as 0 and 3 are supported.
    """
    statuses = []
    device_rows = {}
    busy_ids = set()
    proc_section = False
    for line in (stdout or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "Process id" in stripped:
            proc_section = True
            continue
        # Device block header => end of process section once process section starts.
        m = re.match(r"^\|\s*(\d+)\s+(\S+)\s*\|\s*(OK|[A-Za-z]+)\s*\|", stripped)
        if m:
            dev_id = int(m.group(1))
            health = m.group(3).strip().lower()
            device_rows[dev_id] = (health == "ok")
            proc_section = False
            continue
        if "No running processes found in NPU" in stripped:
            nm = re.search(r"NPU\s+(\d+)", stripped)
            if nm:
                # Explicitly idle; remove any prior busy marking.
                busy_ids.discard(int(nm.group(1)))
            continue
        if proc_section:
            # Process row: | <npu_id> | <pid> | <name> | ...
            pm = re.match(r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|", stripped)
            if pm:
                busy_ids.add(int(pm.group(1)))
    if not device_rows:
        raise DeviceDiscoveryError("npu-smi output could not be parsed confidently")
    for dev_id in sorted(device_rows):
        statuses.append(NpuStatus(
            device_id=dev_id,
            healthy=device_rows[dev_id],
            busy=dev_id in busy_ids,
        ))
    return statuses


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def select_devices(requested, explicit=None, inherited=None, smi_statuses=None,
                   assume_available=False, best_effort=False):
    """Select up to `requested` device ids using the documented priority.

    Returns (devices, source) where source is one of:
      "PTO_ST_PARALLEL_DEVICES", "ASCEND_RT_VISIBLE_DEVICES", "npu-smi",
      "/dev", or None (insufficient).
    """
    # 1. Explicit operator assignment.
    if explicit is not None:
        ids = parse_device_list(explicit)
        if len(ids) < requested:
            if best_effort:
                return [], "PTO_ST_PARALLEL_DEVICES"
            raise DeviceDiscoveryError(
                f"PTO_ST_PARALLEL_DEVICES has {len(ids)} ids, need {requested}"
            )
        return ids[:requested], "PTO_ST_PARALLEL_DEVICES"

    # 2. Inherited visible list.
    if inherited:
        ids = list(dict.fromkeys(inherited))
        if len(ids) >= requested:
            return ids[:requested], "ASCEND_RT_VISIBLE_DEVICES"
        if not best_effort:
            raise DeviceDiscoveryError(
                f"ASCEND_RT_VISIBLE_DEVICES has {len(ids)} ids, need {requested}"
            )

    # 3. npu-smi healthy + idle.
    if smi_statuses is not None:
        candidates = [s.device_id for s in smi_statuses if s.healthy and not s.busy]
        if len(candidates) >= requested:
            return candidates[:requested], "npu-smi"

    # 4. /dev enumeration only with explicit opt-in.
    if assume_available:
        dev_ids = list_dev_davinci_ids()
        if len(dev_ids) >= requested:
            return dev_ids[:requested], "/dev"

    # 5. Best-effort serial fallback.
    if best_effort:
        return [], None
    raise DeviceDiscoveryError(
        "cannot determine NPU allocation safely; set PTO_ST_PARALLEL_DEVICES, "
        "provide ASCEND_RT_VISIBLE_DEVICES, or explicitly use --parallel-assume-available"
    )


def list_dev_davinci_ids():
    """Enumerate exact /dev/davinciN nodes, returning their numeric ids."""
    ids = []
    for ent in sorted(Path("/dev").iterdir()) if Path("/dev").is_dir() else []:
        m = re.fullmatch(r"davinci(\d+)", ent.name)
        if m:
            ids.append(int(m.group(1)))
    return ids


# ---------------------------------------------------------------------------
# Cooperative flock
# ---------------------------------------------------------------------------

def lock_candidate(device_id, lock_dir=None):
    """Acquire a nonblocking exclusive flock for a device, or return None.

    Returns a DeviceLock, or None if the device cannot be locked.
    """
    if fcntl is None:
        raise SmokeError("fcntl unavailable; cooperative locking not supported on this platform")
    lock_dir = Path(lock_dir or os.environ.get("PTO_ST_LOCK_DIR", "/tmp"))
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = lock_dir / f"pto-isa-smoke-npu-{device_id}.lock"
    try:
        fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    # Record context for diagnostics (ownership is flock, not file contents).
    try:
        os.write(fd, f"pid={os.getpid()}\n".encode())
    except OSError:
        pass
    return DeviceLock(device_id=device_id, fd=fd, path=str(path))


def release_lock(device_lock):
    """Release a device lock and close its fd, if any."""
    if device_lock is None:
        return
    try:
        fcntl.flock(device_lock.fd, fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        os.close(device_lock.fd)
    except Exception:
        pass