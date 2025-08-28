#!/usr/bin/env python3
"""
Test Runner for YouTube Summarizer API
=====================================

Simple script to run pytest tests with proper configuration.
"""

import os
import subprocess
import sys
from pathlib import Path


def install_test_dependencies():
    """Install test dependencies if not already installed."""
    try:
        import httpx
        import pytest
        from fastapi.testclient import TestClient

        print("✅ All test dependencies are already installed")
        return True
    except ImportError:
        print("📦 Installing test dependencies...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements-test.txt"], check=True)
            print("✅ Test dependencies installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install test dependencies")
            return False


def run_tests(coverage=False, verbose=True):
    """Run the test suite."""
    cmd = [sys.executable, "-m", "pytest"]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(["--cov=app", "--cov=youtube_summarizer", "--cov-report=html", "--cov-report=term"])

    cmd.append("test_api.py")

    print(f"🧪 Running tests: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def main():
    """Main test runner function."""
    print("🚀 YouTube Summarizer API Test Runner")
    print("=" * 40)

    # Check if we're in the right directory
    if not Path("app.py").exists():
        print("❌ app.py not found. Please run from the project root directory.")
        sys.exit(1)

    # Install dependencies if needed
    if not install_test_dependencies():
        sys.exit(1)

    # Run tests
    print("\n🧪 Running API tests...")
    success = run_tests(coverage=False, verbose=True)

    if success:
        print("\n✅ All tests passed!")

        # Ask if user wants coverage report
        try:
            response = input("\n📊 Run with coverage report? (y/N): ").strip().lower()
            if response in ["y", "yes"]:
                print("\n📊 Running tests with coverage...")
                run_tests(coverage=True, verbose=False)
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
