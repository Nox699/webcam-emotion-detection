"""Regression checks for the Windows launcher runtime-check quoting failure."""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.check_runtime import main, validate_runtime

ROOT = Path(__file__).resolve().parent.parent


class BootstrapTests(unittest.TestCase):
    def test_supported_64_bit_versions(self):
        for minor in (11, 12, 13):
            with self.subTest(minor=minor):
                validate_runtime((3, minor), 8)

    def test_unsupported_versions_have_clear_errors(self):
        for version in ((2, 7), (3, 10), (3, 14), (4, 0)):
            with self.subTest(version=version):
                with self.assertRaisesRegex(RuntimeError, "3.11, 3.12, or 3.13"):
                    validate_runtime(version, 8)

    def test_32_bit_runtime_has_clear_error(self):
        with self.assertRaisesRegex(RuntimeError, "64-bit"):
            validate_runtime((3, 12), 4)

    def test_failure_returns_nonzero(self):
        with patch("scripts.check_runtime.validate_runtime", side_effect=RuntimeError("Use 64-bit Python")), \
             patch("scripts.check_runtime.sys.stderr") as stderr:
            self.assertEqual(main([]), 1)
            self.assertTrue(stderr.write.called)

    def test_check_executes_from_a_path_with_spaces(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime check with spaces.py"
            shutil.copyfile(ROOT / "scripts" / "check_runtime.py", path)
            for arguments in ([], ["--print-executable"]):
                result = subprocess.run([sys.executable, str(path), *arguments],
                                        capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), sys.executable if arguments else "")

    def test_powershell_uses_the_script_for_every_runtime_probe(self):
        # Regression guard: no Python source is transported as a native -c argument.
        source = (ROOT / "scripts" / "bootstrap_windows.ps1").read_text()
        self.assertNotIn(" -c ", source)
        self.assertNotIn("@('-c'", source)
        self.assertIn("Join-Path $PSScriptRoot 'check_runtime.py'", source)
        self.assertIn("-Arguments @($RuntimeCheck)", source)
        self.assertEqual(source.count("$RuntimeCheck '--print-executable'"), 2)


if __name__ == "__main__":
    unittest.main()
