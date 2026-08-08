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


if __name__ == "__main__":
    unittest.main()