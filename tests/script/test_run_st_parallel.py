# --------------------------------------------------------------------------------
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

# Host-only unit tests for the A3 parallel smoke orchestrator.
# No NPU binaries or device access are required.

import os
import sys
import json
import queue
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_st  # noqa: E402
import run_st_parallel as rsp

_BASH = shutil.which("bash") or "bash"


def _make_base_argv(manifest, st_root, base_build, run_st_script):
    """Build the base argv for run_st_parallel.py in host-only tests."""
    return [
        "run_st_parallel.py",
        "--manifest", str(manifest),
        "--workers", "2",
        "--st-root", str(st_root),
        "--base-build-dir", str(base_build),
        "--run-st-script", str(run_st_script),
        "--run-mode", "npu",
        "--soc-version", "a3",
    ]


class BuildDirIsolationTest(unittest.TestCase):
    """Task 2: run_st.py --build-dir routes gen_data and binary cwd independently."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

        self.testcase_dir = self.root / "testcase" / "tfoo"
        self.testcase_dir.mkdir(parents=True)
        (self.testcase_dir / "gen_data.py").write_text("# generator\n")

        self.custom_build = self.root / "custom-build"
        self.custom_bin = self.custom_build / "bin"
        self.custom_bin.mkdir(parents=True)

        self.default_build = self.root / "build"
        self.default_bin = self.default_build / "bin"
        self.default_bin.mkdir(parents=True)

    def tearDown(self):
        self._tmp.cleanup()

    def test_gen_data_copies_to_custom_build_dir(self):
        golden = os.path.join("testcase", "tfoo", "gen_data.py")
        mock_rc = self._run_gen_data_in_cwd(
            str(self.root), golden, build_dir=str(self.custom_build)
        )
        cp_cmd = mock_rc.call_args_list[0].args[0]
        # First command must be `cp <golden> <custom-build>/gen_data.py`.
        self.assertEqual(cp_cmd[0], "cp")
        self.assertIn(str(self.custom_build), cp_cmd[-1])
        # The generator must be executed from the custom build root.
        gen_cmd = mock_rc.call_args_list[1].args[0]
        self.assertEqual(gen_cmd, [sys.executable, "gen_data.py"])

    def test_run_binary_uses_custom_bin_dir(self):
        binary = self.custom_bin / "tfoo"
        binary.write_text("fake binary")
        old = os.getcwd()
        os.chdir(str(self.root))
        try:
            with mock.patch.object(run_st, "run_command", return_value="") as mock_rc:
                run_st.run_binary(
                    "tfoo", "npu", args="TFooTest.case1", build_dir=str(self.custom_build)
                )
            cmd = mock_rc.call_args_list[0].args[0]
            # Binary must be invoked from the custom bin dir (cwd set by chdir).
            self.assertEqual(cmd[0], "./tfoo")
            self.assertEqual(cmd[1], "--gtest_filter=TFooTest.case1")
        finally:
            os.chdir(old)

    def test_run_binary_rejects_missing_bin_dir(self):
        missing = self.root / "nope"
        with self.assertRaises(FileNotFoundError):
            run_st.run_binary("tfoo", "npu", build_dir=str(missing))

    def test_run_gen_data_rejects_missing_build_root(self):
        missing = self.root / "missing-root"
        with self.assertRaises(FileNotFoundError):
            run_st.run_gen_data(
                os.path.join("testcase", "tfoo", "gen_data.py"), build_dir=str(missing)
            )

    def _run_gen_data_in_cwd(self, cwd, *args, **kwargs):
        """Invoke run_gen_data with a controlled cwd and a mocked run_command."""
        old = os.getcwd()
        os.chdir(cwd)
        try:
            with mock.patch.object(run_st, "run_command", return_value="") as mock_rc:
                run_st.run_gen_data(*args, **kwargs)
            return mock_rc
        finally:
            os.chdir(old)


class DeviceListTest(unittest.TestCase):
    def test_valid_lists(self):
        self.assertEqual(rsp.parse_device_list("0"), [0])
        self.assertEqual(rsp.parse_device_list("0,3"), [0, 3])
        self.assertEqual(rsp.parse_device_list("3,5,7"), [3, 5, 7])

    def test_invalid_lists(self):
        for bad in ("0,", "0,0", "-1,3", "0,a"):
            with self.assertRaises(ValueError, msg=f"should reject {bad!r}"):
                rsp.parse_device_list(bad)


class NpuSmiParseTest(unittest.TestCase):
    def test_healthy_no_processes(self):
        # Mirrors the observed 910B4 output with non-contiguous ids 0 and 3.
        out = (
            "| NPU   Name                | Health        | Power(W) ... |\n"
            "| 0     910B4               | OK            | 84.0        41 ... |\n"
            "| 0                         | 0000:C2:00.0  | 0           0    / 0  ... |\n"
            "| 3     910B4               | OK            | 87.5        42 ... |\n"
            "| 0                         | 0000:02:00.0  | 0           0    / 0  ... |\n"
            "| No running processes found in NPU 0 |\n"
            "| No running processes found in NPU 3 |\n"
        )
        statuses = rsp.parse_npu_smi(out)
        by_id = {s.device_id: s for s in statuses}
        self.assertEqual(set(by_id), {0, 3})
        self.assertTrue(by_id[0].healthy and not by_id[0].busy)
        self.assertTrue(by_id[3].healthy and not by_id[3].busy)

    def test_unhealthy_device(self):
        out = (
            "| NPU   Name                | Health |\n"
            "| 5     910B4               | Abnormal |\n"
        )
        statuses = rsp.parse_npu_smi(out)
        self.assertEqual(statuses[0].device_id, 5)
        self.assertFalse(statuses[0].healthy)

    def test_busy_with_process(self):
        out = (
            "| NPU   Name                | Health |\n"
            "| 0     910B4               | OK     |\n"
            "| 0     | 0000:C2:00.0 |\n"
            "| NPU   Chip     Process id |\n"
            "| 0 | 12345 | python | 100 |\n"
        )
        statuses = rsp.parse_npu_smi(out)
        self.assertTrue(statuses[0].busy)

    def test_idle_without_numbered_process(self):
        out = (
            "| NPU   Name                | Health |\n"
            "| 0     910B4               | OK     |\n"
            "| No running processes found in NPU 0 |\n"
        )
        statuses = rsp.parse_npu_smi(out)
        self.assertFalse(statuses[0].busy)

    def test_unparseable_raises(self):
        with self.assertRaises(rsp.DeviceDiscoveryError):
            rsp.parse_npu_smi("no device rows here")


class DeviceSelectionTest(unittest.TestCase):
    def test_explicit_precedence(self):
        devices, source = rsp.select_devices(
            2, rsp.DeviceSources(explicit="3,5", inherited=[0, 1], smi_statuses=[])
        )
        self.assertEqual(devices, [3, 5])
        self.assertEqual(source, "PTO_ST_PARALLEL_DEVICES")

    def test_inherited_used_when_no_explicit(self):
        devices, source = rsp.select_devices(
            2, rsp.DeviceSources(explicit=None, inherited=[0, 3], smi_statuses=[])
        )
        self.assertEqual(devices, [0, 3])
        self.assertEqual(source, "ASCEND_RT_VISIBLE_DEVICES")

    def test_npu_smi_used_when_no_lists(self):
        statuses = [rsp.NpuStatus(0, True, False), rsp.NpuStatus(3, True, False)]
        devices, source = rsp.select_devices(
            2, rsp.DeviceSources(explicit=None, inherited=[], smi_statuses=statuses)
        )
        self.assertEqual(devices, [0, 3])
        self.assertEqual(source, "npu-smi")

    def test_busy_and_unhealthy_excluded(self):
        statuses = [
            rsp.NpuStatus(0, True, True),   # busy
            rsp.NpuStatus(1, False, False),  # unhealthy
            rsp.NpuStatus(3, True, False),   # ok
        ]
        devices, source = rsp.select_devices(
            1, rsp.DeviceSources(explicit=None, inherited=[], smi_statuses=statuses)
        )
        self.assertEqual(devices, [3])

    def test_insufficient_strict_raises(self):
        statuses = [rsp.NpuStatus(0, True, False)]
        with self.assertRaises(rsp.DeviceDiscoveryError):
            rsp.select_devices(
                2, rsp.DeviceSources(explicit=None, inherited=[], smi_statuses=statuses)
            )

    def test_best_effort_falls_back(self):
        devices, source = rsp.select_devices(
            2, rsp.DeviceSources(explicit=None, inherited=[], smi_statuses=None),
            assume_available=False, best_effort=True,
        )
        self.assertEqual(devices, [])
        self.assertIsNone(source)

    def test_dev_requires_opt_in(self):
        with unittest.mock.patch.object(rsp, "list_dev_davinci_ids", return_value=[0, 3]):
            # without assume_available -> strict error
            with self.assertRaises(rsp.DeviceDiscoveryError):
                rsp.select_devices(
                    2, rsp.DeviceSources(explicit=None, inherited=[], smi_statuses=None)
                )
            # with assume_available -> /dev source
            devices, source = rsp.select_devices(
                2, rsp.DeviceSources(explicit=None, inherited=[], smi_statuses=None),
                assume_available=True,
            )
            self.assertEqual(devices, [0, 3])
            self.assertEqual(source, "/dev")


class ManifestTest(unittest.TestCase):
    def test_parse_valid(self):
        with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as f:
            f.write("mscatter\tMSCATTERTest.case1\t0\n")
            f.write("tfillpad\tTFILLPADTest.case2\t1\n")
            f.write("tmins\t\t0\n")
            path = f.name
        try:
            tasks = rsp.parse_task_manifest(path)
            self.assertEqual(len(tasks), 3)
            self.assertEqual(tasks[0].testcase, "mscatter")
            self.assertEqual(tasks[0].gtest_filter, "MSCATTERTest.case1")
            self.assertFalse(tasks[0].debug_enable)
            self.assertTrue(tasks[1].debug_enable)
            self.assertIsNone(tasks[2].gtest_filter)
        finally:
            os.unlink(path)

    def test_parse_rejects_bad_rows(self):
        with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as f:
            f.write("mscatter\tx\tbad\n")
            path = f.name
        try:
            with self.assertRaises(rsp.ManifestError):
                rsp.parse_task_manifest(path)
        finally:
            os.unlink(path)

    def test_parse_rejects_duplicate(self):
        with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as f:
            f.write("mscatter\tx\t0\nmscatter\tx\t0\n")
            path = f.name
        try:
            with self.assertRaises(rsp.ManifestError):
                rsp.parse_task_manifest(path)
        finally:
            os.unlink(path)


class WorkerIsolationTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.base = self.root / "base-build"
        self.base_bin = self.base / "bin"
        self.base_bin.mkdir(parents=True)
        (self.base_bin / "fake_test").write_text("fake binary")

    def tearDown(self):
        self._tmp.cleanup()

    def test_worker_builds_are_distinct_real_dirs(self):
        w1 = self.root / "worker-0" / "build"
        w2 = self.root / "worker-1" / "build"
        b1 = rsp.prepare_worker_build(self.base, w1)
        b2 = rsp.prepare_worker_build(self.base, w2)
        self.assertNotEqual(str(b1.resolve()), str(b2.resolve()))
        self.assertTrue(b1.is_dir())
        self.assertTrue(b2.is_dir())
        # Real directories, not symlinks to the shared bin.
        self.assertFalse(b1.is_symlink())
        self.assertFalse(b2.is_symlink())
        self.assertTrue((b1 / "fake_test").exists())
        self.assertTrue((b2 / "fake_test").exists())

    def test_remove_parallel_root_refuses_outside_marker(self):
        outside = self.root / "build"  # not under .smoke-parallel
        outside.mkdir(exist_ok=True)
        with self.assertRaises(rsp.SmokeError):
            rsp.remove_parallel_root(outside)

    def test_remove_parallel_root_removes_run_root(self):
        marker = self.root / ".smoke-parallel" / "12345"
        marker.mkdir(parents=True)
        rsp.remove_parallel_root(marker)
        self.assertFalse(marker.exists())


class ParallelSchedulingTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        rsp.reset_active_procs()
        self._tmp.cleanup()

    def test_two_workers_get_distinct_roots_and_devices(self):
        record = self.root / "record.jsonl"
        w1, w2, q, ctx, results = self._setup_worker_pair(record)
        with unittest.mock.patch.dict(os.environ, {"RECORD": str(record)}):
            th1 = threading.Thread(target=rsp.worker_loop, args=(w1, q, ctx, results))
            th2 = threading.Thread(target=rsp.worker_loop, args=(w2, q, ctx, results))
            th1.start()
            th2.start()
            th1.join()
            th2.join()

        rows = [json.loads(l) for l in record.read_text().splitlines()]
        build_dirs = {r["build_dir"] for r in rows}
        devices = {r["dev"] for r in rows}
        self.assertEqual(len(build_dirs), 2)
        self.assertEqual(len(devices), 2)
        self.assertEqual(devices, {"3", "5"})

    def _setup_worker_pair(self, record):
        """Prepare two workers and a task queue for the isolation test."""
        script = self.root / "fake_run_st.py"
        script.write_text(
            "import sys, os, json\n"
            "out = os.environ.get('RECORD','')\n"
            "with open(out,'a') as f:\n"
            "    f.write(json.dumps({'build_dir': sys.argv[sys.argv.index('--build-dir')+1],\n"
            "        'dev': os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}) + '\\n')\n"
            "sys.exit(0)\n"
        )
        manifest = self.root / "tasks.tsv"
        manifest.write_text("t1\tF1.case1\t0\nt2\tF2.case1\t0\n")

        base = self.root / "base"
        base_bin = base / "bin"
        base_bin.mkdir(parents=True)
        (base_bin / "fake_test").write_text("x")

        st_root = self.root / "st"
        st_root.mkdir()

        tasks = rsp.parse_task_manifest(str(manifest))
        w1_build = self.root / "w1" / "build"
        w2_build = self.root / "w2" / "build"
        rsp.prepare_worker_build(base, w1_build)
        rsp.prepare_worker_build(base, w2_build)
        w1 = rsp.WorkerConfig(0, 3, w1_build, self.root / "w1.log")
        w2 = rsp.WorkerConfig(1, 5, w2_build, self.root / "w2.log")

        q = queue.Queue()
        for t in tasks:
            q.put(t)
        q.put(None)
        q.put(None)
        print_lock = threading.Lock()
        results = []
        ctx = rsp.WorkerContext(
            run_st_script=str(script),
            auto_mode=False,
            print_lock=print_lock,
            now=time.perf_counter,
        )
        return w1, w2, q, ctx, results

    def test_failed_task_causes_nonzero_and_others_still_run(self):
        script = self.root / "fake_run_st.py"
        script.write_text(
            "import sys\n"
            "tc = sys.argv[sys.argv.index('-t')+1]\n"
            "sys.exit(1 if tc=='tbad' else 0)\n"
        )
        manifest = self.root / "tasks.tsv"
        manifest.write_text("tbad\tF.case1\t0\ntok\tF.case2\t0\n")
        tasks = rsp.parse_task_manifest(str(manifest))
        base = self.root / "base"
        (base / "bin").mkdir(parents=True)
        (base / "bin" / "fake_test").write_text("x")
        wb = self.root / "w" / "build"
        rsp.prepare_worker_build(base, wb)
        w = rsp.WorkerConfig(0, 0, wb, self.root / "w.log")
        q = queue.Queue()
        for t in tasks:
            q.put(t)
        q.put(None)
        results = []
        ctx = rsp.WorkerContext(
            run_st_script=str(script),
            auto_mode=False,
            print_lock=threading.Lock(),
            now=time.perf_counter,
        )
        rsp.worker_loop(w, q, ctx, results)
        self.assertEqual(len(results), 2)  # both tasks drained
        self.assertEqual(results[0].returncode, 1)
        self.assertEqual(results[1].returncode, 0)


class RegistryDumpTest(unittest.TestCase):
    """Task 5: dump-mode registry invariants (host-only, no NPU)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_registry_invariants(self):
        tasks = self._dump()
        # tfillpad: two tasks, one with debug enabled.
        tfp = [t for t in tasks if t[0] == "tfillpad"]
        self.assertEqual(len(tfp), 2)
        self.assertEqual({t[2] for t in tfp}, {"0", "1"})
        # mgather_gm2l1: single task, 4 unique filters.
        mg = [t for t in tasks if t[0] == "mgather_gm2l1"]
        self.assertEqual(len(mg), 1)
        filters = mg[0][1].split(":")
        self.assertEqual(len(filters), 4)
        self.assertEqual(len(set(filters)), 4)
        # tmins: empty filter.
        tmins = [t for t in tasks if t[0] == "tmins"]
        self.assertEqual(len(tmins), 1)
        self.assertEqual(tmins[0][1], "")
        # No empty-testcase rows.
        self.assertTrue(all(t[0] for t in tasks))

    def _dump(self):
        """Run run_st.sh in dump mode and return the parsed task list."""
        repo = Path(__file__).resolve().parents[2]  # repo root
        dump = self.root / "dump.tsv"
        env = dict(os.environ)
        env["PTO_ST_DUMP_TASKS"] = str(dump)
        # Replace python3 so the test runs on hosts where python3 resolves oddly.
        shim = self.root / "pyshim"
        shim.mkdir()
        pyexe = sys.executable.replace("\\", "/")
        (shim / "python3").write_text(
            "#!/bin/bash\nexec %s \"$@\"\n" % pyexe
        )
        os.chmod(shim / "python3", 0o755)
        env["PATH"] = str(shim) + os.pathsep + env.get("PATH", "")
        proc = subprocess.run(
            [_BASH, "tests/run_st.sh", "--a3", "--npu", "--simple"],
            cwd=str(repo), env=env, capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        tasks = []
        for line in dump.read_text().splitlines():
            parts = line.split("\t")
            tasks.append((parts[0], parts[1], parts[2]))
        return tasks


class ParallelOptionValidationTest(unittest.TestCase):
    """Task 6: build.sh / run_st.sh parallel option validation (host-only)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.repo = Path(__file__).resolve().parents[2]

    def tearDown(self):
        self._tmp.cleanup()

    def test_zero_parallel_rejected(self):
        rc, err = self._run_sh(parallel="0")
        self.assertNotEqual(rc, 0)
        self.assertIn("positive integer", err)

    def test_non_numeric_parallel_rejected(self):
        rc, err = self._run_sh(parallel="abc")
        self.assertNotEqual(rc, 0)

    def test_a5_parallel_rejected(self):
        # run_st.sh itself rejects --parallel>1 for A5.
        dump = self.root / "dump.tsv"
        env = dict(os.environ)
        env["PTO_ST_DUMP_TASKS"] = str(dump)
        shim = self.root / "pyshim2"
        shim.mkdir()
        (shim / "python3").write_text("#!/bin/bash\nexec %s \"$@\"\n" % sys.executable.replace("\\", "/"))
        os.chmod(shim / "python3", 0o755)
        env["PATH"] = str(shim) + os.pathsep + env.get("PATH", "")
        proc = subprocess.run(
            [_BASH, "tests/run_st.sh", "--a5", "--npu", "--simple", "--parallel=2"],
            cwd=str(self.repo), env=env, capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("supported only for A3", proc.stderr)

    def test_serial_dump_does_not_need_parallel(self):
        rc, err = self._run_sh()  # no parallel option -> serial default
        self.assertEqual(rc, 0)

    def _run_sh(self, *args, parallel=None, best_effort=None, assume=None):
        """Run run_st.sh in dump mode with given options; return returncode+stderr."""
        dump = self.root / "dump.tsv"
        env = dict(os.environ)
        env["PTO_ST_DUMP_TASKS"] = str(dump)
        shim = self.root / "pyshim"
        shim.mkdir()
        pyexe = sys.executable.replace("\\", "/")
        (shim / "python3").write_text("#!/bin/bash\nexec %s \"$@\"\n" % pyexe)
        os.chmod(shim / "python3", 0o755)
        env["PATH"] = str(shim) + os.pathsep + env.get("PATH", "")
        cmd = [_BASH, "tests/run_st.sh", "--a3", "--npu", "--simple"]
        if parallel is not None:
            cmd.append("--parallel=%s" % parallel)
        if best_effort:
            cmd.append("--parallel-best-effort")
        if assume:
            cmd.append("--parallel-assume-available")
        proc = subprocess.run(cmd, cwd=str(self.repo), env=env,
                              capture_output=True, text=True)
        return proc.returncode, proc.stderr


class BestEffortFallbackTest(unittest.TestCase):
    """P0: best-effort must actually execute tasks serially, not skip them."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.st_root = self.root / "st"
        self.st_root.mkdir()
        self.base_build = self.root / "base"
        self.base_build.mkdir()
        self.run_st_script = self.root / "run_st.py"

    def tearDown(self):
        self._tmp.cleanup()

    def test_serial_fallback_executes_all_tasks(self):
        script = self._make_fake_runner()
        manifest = self._manifest(["t1", "t2", "t3"])
        tasks = rsp.parse_task_manifest(manifest)
        record = self.root / "record.txt"
        with mock.patch.dict(os.environ, {"RECORD": str(record)}):
            rc, executed = rsp.run_serial_fallback(tasks, self._args(), threading.Lock(), time.perf_counter)
        self.assertEqual(rc, 0)
        self.assertEqual(executed, 3)
        self.assertEqual(record.read_text().splitlines(), ["t1", "t2", "t3"])

    def test_serial_fallback_propagates_failure(self):
        script = self._make_fake_runner(fail_testcase="t2")
        manifest = self._manifest(["t1", "t2", "t3"])
        tasks = rsp.parse_task_manifest(manifest)
        record = self.root / "record.txt"
        with mock.patch.dict(os.environ, {"RECORD": str(record)}):
            rc, executed = rsp.run_serial_fallback(tasks, self._args(), threading.Lock(), time.perf_counter)
        self.assertEqual(rc, 1)
        self.assertEqual(executed, 3)  # all tasks still attempted
        self.assertEqual(record.read_text().splitlines(), ["t1", "t2", "t3"])

    def _make_fake_runner(self, fail_testcase=None):
        """A fake run_st.py that records invocations and can fail on a testcase."""
        script = self.run_st_script
        script.write_text(
            "import sys, os\n"
            "tc = sys.argv[sys.argv.index('-t')+1]\n"
            "rec = os.environ.get('RECORD','')\n"
            "with open(rec,'a') as f:\n"
            "    f.write(tc + '\\n')\n"
            "sys.exit(1 if tc == %r else 0)\n" % (fail_testcase if fail_testcase else "__none__")
        )
        return str(script)

    def _manifest(self, tasks):
        p = self.root / "tasks.tsv"
        p.write_text("".join("%s\t%s\t0\n" % (t, ("F.case" + t)) for t in tasks))
        return str(p)

    def _args(self):
        return types.SimpleNamespace(
            workers=2,
            run_st_script=str(self.run_st_script),
            st_root=str(self.st_root),
            base_build_dir=str(self.base_build),
            auto_mode=False,
            best_effort=True,
            assume_available=False,
        )


class FullPoolLockingTest(unittest.TestCase):
    """P1: if a candidate is locked, keep trying later candidates from the pool."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.lock_dir = self.root / "locks"
        self.lock_dir.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def test_locks_skip_busy_candidate_and_take_later_one(self):
        if rsp.fcntl is None:
            self.skipTest("fcntl unavailable on this platform")
        # Pre-lock device 0 so it cannot be acquired.
        lk0 = rsp.lock_candidate(0, lock_dir=str(self.lock_dir))
        self.assertIsNotNone(lk0)
        try:
            # Devices 0,1,2 available; 0 locked -> should pick 1,2.
            lock1 = rsp.lock_candidate(1, lock_dir=str(self.lock_dir))
            lock2 = rsp.lock_candidate(2, lock_dir=str(self.lock_dir))
            self.assertIsNotNone(lock1)
            self.assertIsNotNone(lock2)
            self.assertEqual(lock1.device_id, 1)
            self.assertEqual(lock2.device_id, 2)
            # Locking 0 again must fail (still held).
            lock0_again = rsp.lock_candidate(0, lock_dir=str(self.lock_dir))
            self.assertIsNone(lock0_again)
            rsp.release_lock(lock1)
            rsp.release_lock(lock2)
        finally:
            rsp.release_lock(lk0)
        # After release, 0 becomes lockable again.
        lk0b = rsp.lock_candidate(0, lock_dir=str(self.lock_dir))
        self.assertIsNotNone(lk0b)
        rsp.release_lock(lk0b)


class BuildShParsingTest(unittest.TestCase):
    """P0: build.sh must consume --parallel and its value correctly."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.repo = Path(__file__).resolve().parents[2]

    def tearDown(self):
        self._tmp.cleanup()

    def test_parallel_parses_and_consumes(self):
        # Extract checkopts from build.sh and run it with --parallel=2.
        repo = str(self.repo).replace("\\", "/")
        script = self.root / "extract.sh"
        script.write_text(
            "#!/bin/bash\n"
            "PARALLEL_WORKERS=1\n"
            "source <(sed -n '/^checkopts()/,/^}/p' %s/build.sh)\n"
            "checkopts --run_simple --a3 --parallel=2\n"
            "echo \"PARALLEL_WORKERS=$PARALLEL_WORKERS\"\n" % repo
        )
        proc = subprocess.run([_BASH, str(script)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("PARALLEL_WORKERS=2", proc.stdout)

    def test_parallel_separate_form_parses(self):
        repo = str(self.repo).replace("\\", "/")
        script = self.root / "extract2.sh"
        script.write_text(
            "#!/bin/bash\n"
            "PARALLEL_WORKERS=1\n"
            "source <(sed -n '/^checkopts()/,/^}/p' %s/build.sh)\n"
            "checkopts --run_simple --a3 --parallel 2 --auto_mode\n"
            "echo \"PARALLEL_WORKERS=$PARALLEL_WORKERS\"\n" % repo
        )
        proc = subprocess.run([_BASH, str(script)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("PARALLEL_WORKERS=2", proc.stdout)


class MainFlowRegressionTest(unittest.TestCase):
    """P1: main() orchestration regression tests (mocked devices/runner)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.st_root = self.root / "st"
        self.st_root.mkdir()
        self.base_build = self.root / "base"
        self.base_build.mkdir()
        self.run_st_script = self.root / "run_st.py"
        self.manifest = self.root / "tasks.tsv"
        self.manifest.write_text("t1\tF.case1\t0\nt2\tF.case2\t0\n")

    def tearDown(self):
        rsp.reset_active_procs()
        self._tmp.cleanup()

    def test_no_device_best_effort_enters_serial_fallback(self):
        """main() with no devices and best-effort must run all tasks serially."""
        fake = self.run_st_script
        fake.write_text(
            "import sys, os\n"
            "rec=os.environ['RECORD']\n"
            "with open(rec,'a') as f: f.write('x\\n')\n"
            "sys.exit(0)\n"
        )
        record = self.root / "rec.txt"
        argv = self._base_argv() + ["--best-effort"]
        with mock.patch.object(sys, "argv", argv), \
             mock.patch.object(rsp, "select_devices", return_value=([], None)), \
             mock.patch.object(rsp, "query_npu_smi",
                               side_effect=rsp.DeviceDiscoveryError("no smi")), \
             mock.patch.dict(os.environ, {"RECORD": str(record)}):
            rc = rsp.main()
        self.assertEqual(rc, 0)
        # Serial fallback executed both tasks.
        self.assertEqual(record.read_text().count("x"), 2)

    def test_build_prep_exception_releases_locks_and_preserves_dir(self):
        """If worker build prep raises, locks release and run dir is preserved."""
        argv = self._base_argv()
        with mock.patch.object(sys, "argv", argv), \
             mock.patch.object(rsp, "select_devices", return_value=([0, 3], "npu-smi")), \
             mock.patch.object(rsp, "query_npu_smi", return_value=[
                 rsp.NpuStatus(0, True, False), rsp.NpuStatus(3, True, False),
             ]), \
             mock.patch.object(rsp, "lock_candidate") as lk, \
             mock.patch.object(rsp, "release_lock") as rl:
            lk.side_effect = [
                rsp.DeviceLock(0, fd=None, path=str(self.root / "l0")),
                rsp.DeviceLock(3, fd=None, path=str(self.root / "l3")),
            ]
            with self.assertRaises(OSError):
                with mock.patch.object(rsp, "prepare_worker_build",
                                       side_effect=OSError("boom")):
                    rsp.main()
            # Both locks must be released during cleanup.
            self.assertEqual(rl.call_count, 2)
            # The run dir is preserved on failure (not removed).
            run_dir = self.st_root / ".smoke-parallel" / str(os.getpid())
            self.assertTrue(run_dir.exists())

    def _base_argv(self):
        return _make_base_argv(
            self.manifest, self.st_root, self.base_build, self.run_st_script
        )


class NpuSmiNoProcessTableTest(unittest.TestCase):
    """P1: npu-smi with healthy rows but no process table must fail, not guess."""

    def test_healthy_row_without_process_table_raises(self):
        out = (
            "| NPU   Name                | Health |\n"
            "| 0     910B4               | OK     |\n"
        )
        with self.assertRaises(rsp.DeviceDiscoveryError):
            rsp.parse_npu_smi(out)


class SchedulerBoundaryTest(unittest.TestCase):
    """P0: insufficient ASCEND_RT_VISIBLE_DEVICES must not cross to npu-smi."""

    def test_inherited_insufficient_does_not_use_npu_smi(self):
        # inherited=[0] insufficient for 2, npu-smi=[0,3]. Must NOT pick 3.
        devices, source = rsp.select_devices(
            2,
            rsp.DeviceSources(
                explicit=None,
                inherited=[0],
                smi_statuses=[
                    rsp.NpuStatus(0, True, False),
                    rsp.NpuStatus(3, True, False),
                ],
            ),
            best_effort=True,
        )
        self.assertEqual(devices, [])
        self.assertEqual(source, "ASCEND_RT_VISIBLE_DEVICES")

    def test_explicit_insufficient_does_not_use_npu_smi(self):
        devices, source = rsp.select_devices(
            2,
            rsp.DeviceSources(
                explicit="0",
                inherited=[],
                smi_statuses=[
                    rsp.NpuStatus(0, True, False),
                    rsp.NpuStatus(3, True, False),
                ],
            ),
            best_effort=True,
        )
        self.assertEqual(devices, [])
        self.assertEqual(source, "PTO_ST_PARALLEL_DEVICES")


class ProcessGroupCleanupTest(unittest.TestCase):
    """P0: TERM->KILL escalation and process-group cleanup."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        rsp.reset_active_procs()
        self._tmp.cleanup()

    @unittest.skipIf(os.name == "nt", "process groups not used on Windows")

    def test_terminate_proc_group_escalates_to_kill(self):
        # Spawn a child in its own session that ignores SIGTERM.
        script = self.root / "ignore_term.py"
        script.write_text(
            "import signal, time, sys\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "sys.stdout.write('ready' + chr(10)); sys.stdout.flush()\n"
            "time.sleep(30)\n"
        )
        proc = subprocess.Popen([sys.executable, str(script)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, start_new_session=True)
        group = rsp.new_proc_group(proc)
        try:
            # Wait for it to signal readiness.
            proc.stdout.readline()
            # TERM to group, short timeout -> should escalate to KILL.
            exited = rsp.terminate_proc_gracefully(group, timeout=1)
            self.assertTrue(exited)
            # The process must be gone (KILLed).
            self.assertIsNotNone(proc.poll())
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    @unittest.skipIf(os.name == "nt", "process groups not used on Windows")

    def test_terminate_kills_grandchild_and_waits_for_group(self):
        """Leader spawns an ignore-TERM grandchild, then exits; group must be killed.

        The leader starts a grandchild in the same process group, then exits
        immediately. terminate_proc_gracefully must discover the group is still
        alive (grandchild), escalate to SIGKILL, and confirm the whole group is
        gone before returning True.
        """
        leader = self._make_leader_grandchild()
        proc = subprocess.Popen([sys.executable, str(leader)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, start_new_session=True)
        group = rsp.new_proc_group(proc)
        try:
            self._verify_grandchild_killed(proc, group)
        finally:
            if group is not None and rsp.group_exists(group):
                rsp.terminate_process_group(group, signal.SIGKILL)
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def _make_leader_grandchild(self):
        """Create a leader that spawns an ignore-TERM grandchild, return leader path."""
        grandchild = self.root / "grandchild.py"
        grandchild.write_text(
            "import signal, time, os, sys\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "sys.stdout.write('grandchild:' + str(os.getpid()) + chr(10))\n"
            "sys.stdout.flush()\n"
            "time.sleep(30)\n"
        )
        leader = self.root / "leader.py"
        leader.write_text(
            "import subprocess, sys, os\n"
            "p = subprocess.Popen([sys.executable, %r])\n"
            "sys.stdout.write('leader:' + str(os.getpid()) + ':' + str(os.getpgrp()) + chr(10))\n"
            "sys.stdout.flush()\n"
            "sys.exit(0)\n" % str(grandchild)
        )
        return leader

    def _verify_grandchild_killed(self, proc, group):
        """Drive the grandchild-kill scenario and assert the whole group is gone."""
        line = proc.stdout.readline()
        self.assertTrue(line.startswith("leader:"), line)
        _, leader_pid, pgid = line.strip().split(":")
        gline = proc.stdout.readline()
        self.assertTrue(gline.startswith("grandchild:"), gline)
        grandchild_pid = int(gline.strip().split(":")[1])
        proc.wait(timeout=5)
        self.assertTrue(rsp.group_exists(group))
        exited = rsp.terminate_proc_gracefully(group, timeout=2)
        self.assertTrue(exited)
        self.assertFalse(rsp.group_exists(group))
        try:
            os.kill(grandchild_pid, 0)
            grandchild_alive = True
        except OSError:
            grandchild_alive = False
        self.assertFalse(grandchild_alive)


class SuccessCleanupTest(unittest.TestCase):
    """P1: success path removes run dir; failure path preserves it."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.st_root = self.root / "st"
        self.st_root.mkdir()
        self.base_build = self.root / "base"
        self.base_build.mkdir()
        self.run_st_script = self.root / "run_st.py"
        self.manifest = self.root / "tasks.tsv"
        self.manifest.write_text("t1\tF.case1\t0\n")

    def tearDown(self):
        rsp.reset_active_procs()
        self._tmp.cleanup()

    def test_success_removes_run_dir(self):
        fake = self.run_st_script
        fake.write_text("import sys\nsys.exit(0)\n")
        # build_st built into base_build/bin; orchestrator links it into workers.
        base_bin = self.base_build / "bin"
        base_bin.mkdir()
        (base_bin / "fake_test").write_text("x")
        argv = self._base_argv()
        with mock.patch.object(sys, "argv", argv), \
             mock.patch.object(rsp, "select_devices", return_value=([0, 3], "npu-smi")), \
             mock.patch.object(rsp, "query_npu_smi", return_value=[
                 rsp.NpuStatus(0, True, False), rsp.NpuStatus(3, True, False),
             ]), \
             mock.patch.object(rsp, "lock_candidate") as lk:
            lk.side_effect = [
                rsp.DeviceLock(0, fd=None, path=str(self.root / "l0")),
                rsp.DeviceLock(3, fd=None, path=str(self.root / "l3")),
            ]
            rc = rsp.main()
        self.assertEqual(rc, 0)
        run_dir = self.st_root / ".smoke-parallel" / str(os.getpid())
        self.assertFalse(run_dir.exists())

    def _base_argv(self):
        return _make_base_argv(
            self.manifest, self.st_root, self.base_build, self.run_st_script
        )


class ManifestTrapTest(unittest.TestCase):
    """P1: run_st.sh EXIT trap removes the manifest directory."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.repo = Path(__file__).resolve().parents[2]

    def tearDown(self):
        self._tmp.cleanup()

    def test_exit_trap_removes_manifest_dir(self):
        # The trap syntax must be present in run_st.sh.
        content = (self.repo / "tests" / "run_st.sh").read_text(encoding="utf-8")
        self.assertIn("trap 'rm -rf \"$A3_PARALLEL_ROOT\"' EXIT", content)

    @unittest.skipUnless(shutil.which("bash"), "bash not available")

    def test_exit_trap_actually_removes_dir(self):
        # Behaviorally verify the EXIT trap removes the directory on exit.
        target = self.root / ".smoke-parallel" / "99999"
        script = self.root / "trap_test.sh"
        script.write_text(
            "#!/bin/bash\n"
            "A3_PARALLEL_ROOT=%s\n"
            "mkdir -p \"$A3_PARALLEL_ROOT\"\n"
            "trap 'rm -rf \"$A3_PARALLEL_ROOT\"' EXIT\n"
            "exit 3\n" % str(target).replace("\\", "/")
        )
        proc = subprocess.run([_BASH, str(script)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 3)
        self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
