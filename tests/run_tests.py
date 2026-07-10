#!/usr/bin/env python3
"""
Test runner for GUAC training system.

This script runs all available tests and provides a summary of results.
Can be run from the project root directory.

Usage:
    python tests/run_tests.py                    # Run all tests
    python tests/run_tests.py --test path        # Run specific test
    python tests/run_tests.py --list             # List available tests
"""

import sys
import os
import argparse
from pathlib import Path
import subprocess
import importlib.util

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
os.chdir(project_root)  # Set working directory to project root


class TestRunner:
    """Test runner for GUAC tests."""

    def __init__(self):
        self.tests_dir = project_root / "tests"
        self.available_tests = self._discover_tests()

    def _discover_tests(self):
        """Discover all test files in the tests directory."""
        test_files = []
        for test_file in self.tests_dir.glob("test_*.py"):
            if test_file.name != "test_runner.py":  # Skip this file
                test_name = test_file.stem.replace("test_", "")
                test_files.append({
                    "name": test_name,
                    "file": test_file,
                    "module": test_file.stem
                })
        return test_files

    def list_tests(self):
        """List all available tests."""
        print("Available Tests:")
        print("=" * 40)
        for test in self.available_tests:
            print(f"  • {test['name']:<20} - {test['file'].name}")

    def run_test(self, test_name):
        """Run a specific test by name."""
        test_info = None
        for test in self.available_tests:
            if test["name"] == test_name:
                test_info = test
                break

        if not test_info:
            print(f"❌ Test '{test_name}' not found!")
            return False

        print(f"🧪 Running {test_name} test...")
        print("-" * 50)

        try:
            # Run the test as a subprocess to capture output
            result = subprocess.run(
                [sys.executable, str(test_info["file"])],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                print(result.stdout)
                print(f"✅ {test_name} test PASSED")
                return True
            else:
                print(result.stdout)
                if result.stderr:
                    print("STDERR:", result.stderr)
                print(f"❌ {test_name} test FAILED (exit code: {result.returncode})")
                return False

        except subprocess.TimeoutExpired:
            print(f"⏰ {test_name} test TIMED OUT")
            return False
        except Exception as e:
            print(f"💥 {test_name} test CRASHED: {e}")
            return False

    def run_all_tests(self):
        """Run all available tests."""
        print("🚀 Running All GUAC Tests")
        print("=" * 50)

        results = {}
        for test in self.available_tests:
            print(f"\n{'='*20} {test['name'].upper()} {'='*20}")
            results[test["name"]] = self.run_test(test["name"])

        # Summary
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)

        passed = sum(1 for result in results.values() if result)
        total = len(results)

        for test_name, passed_test in results.items():
            status = "✅ PASS" if passed_test else "❌ FAIL"
            print(f"  {test_name:<20} {status}")

        print(f"\nPassed: {passed}/{total}")

        if passed == total:
            print("🎉 All tests PASSED!")
            return True
        else:
            print(f"⚠️  {total - passed} test(s) FAILED")
            return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GUAC Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "Usage:" in __doc__ else ""
    )

    parser.add_argument(
        "--test",
        type=str,
        help="Run specific test by name"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available tests"
    )

    return parser.parse_args()


def main():
    """Main test runner function."""
    args = parse_args()
    runner = TestRunner()

    if args.list:
        runner.list_tests()
        return

    if args.test:
        success = runner.run_test(args.test)
        sys.exit(0 if success else 1)

    # Run all tests
    success = runner.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()