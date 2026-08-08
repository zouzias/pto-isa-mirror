# --------------------------------------------------------------------------------
# coding=utf-8
# Host-only unit tests for the A3 parallel smoke orchestrator.
# No NPU binaries or device access are required.
# --------------------------------------------------------------------------------

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_st  # noqa: E402
import run_st_parallel as rsp  # noqa: E402


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
            2, explicit="3,5", inherited=[0, 1], smi_statuses=[]
        )
        self.assertEqual(devices, [3, 5])
        self.assertEqual(source, "PTO_ST_PARALLEL_DEVICES")

    def test_inherited_used_when_no_explicit(self):
        devices, source = rsp.select_devices(
            2, explicit=None, inherited=[0, 3], smi_statuses=[]
        )
        self.assertEqual(devices, [0, 3])
        self.assertEqual(source, "ASCEND_RT_VISIBLE_DEVICES")

    def test_npu_smi_used_when_no_lists(self):
        statuses = [rsp.NpuStatus(0, True, False), rsp.NpuStatus(3, True, False)]
        devices, source = rsp.select_devices(
            2, explicit=None, inherited=[], smi_statuses=statuses
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
            1, explicit=None, inherited=[], smi_statuses=statuses
        )
        self.assertEqual(devices, [3])

    def test_insufficient_strict_raises(self):
        statuses = [rsp.NpuStatus(0, True, False)]
        with self.assertRaises(rsp.DeviceDiscoveryError):
            rsp.select_devices(2, explicit=None, inherited=[], smi_statuses=statuses)

    def test_best_effort_falls_back(self):
        devices, source = rsp.select_devices(
            2, explicit=None, inherited=[], smi_statuses=None,
            assume_available=False, best_effort=True,
        )
        self.assertEqual(devices, [])
        self.assertIsNone(source)

    def test_dev_requires_opt_in(self):
        with unittest.mock.patch.object(rsp, "list_dev_davinci_ids", return_value=[0, 3]):
            # without assume_available -> strict error
            with self.assertRaises(rsp.DeviceDiscoveryError):
                rsp.select_devices(2, explicit=None, inherited=[], smi_statuses=None)
            # with assume_available -> /dev source
            devices, source = rsp.select_devices(
                2, explicit=None, inherited=[], smi_statuses=None, assume_available=True
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


if __name__ == "__main__":
    unittest.main()