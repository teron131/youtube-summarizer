#!/usr/bin/env python3
"""
Test Runner for YouTube Summarizer API
=====================================

Enhanced test runner with better organization and reporting.
"""

import os
import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """Check if test dependencies are installed."""
    try:
        import httpx
        import pytest
        import rich
        from fastapi.testclient import TestClient

        print("✅ All test dependencies are available")
        return True
    except ImportError as e:
        print(f"❌ Missing test dependency: {e}")
        print("📦 Install test dependencies with: pip install -r requirements-test.txt")
        return False


def run_tests(test_type="all", coverage=False, verbose=True):
    """Run specific test suites."""
    if not check_dependencies():
        return False

    # Prefer running via uv if available; also disable auto-loaded plugins for stability
    uv = os.environ.get("UV", "uv")
    use_uv = True
    try:
        subprocess.run([uv, "--version"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        use_uv = False

    cmd = [uv, "run", "pytest"] if use_uv else [sys.executable, "-m", "pytest"]

    if verbose:
        cmd.extend(["-v", "--tb=short"])

    if coverage:
        cmd.extend(["--cov=app", "--cov=youtube_summarizer", "--cov-report=html", "--cov-report=term", "--cov-report=term-missing"])

    # Test type selection
    if test_type == "meta_language":
        cmd.extend(["-k", "TestMetaLanguageAvoidance"])
        print("🎯 Running Meta-Language Avoidance Tests...")
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
        print("🔗 Running Integration Tests...")
    elif test_type == "streaming":
        cmd.extend(["-k", "streaming"])
        print("📡 Running Streaming Tests...")
    elif test_type == "health":
        cmd.extend(["-k", "health"])
        print("🏥 Running Health Tests...")
    elif test_type != "all":
        print(f"❓ Unknown test type: {test_type}. Running all tests...")

    # Determine which tests to run
    test_files = []
    if test_type == "all":
        test_files = ["tests/"]
    elif test_type == "api":
        test_files = ["tests/test_api.py"]
    elif test_type == "streaming":
        test_files = ["tests/test_streaming.py"]
    elif test_type == "unit":
        test_files = ["tests/test_api.py", "tests/test_streaming.py"]
        cmd.extend(["-m", "not integration"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
        test_files = ["tests/"]
    else:
        print(f"❌ Unknown test type: {test_type}")
        return False

    cmd.extend(test_files)

    print(f"🧪 Running {test_type} tests...")
    print(f"Command: {' '.join(cmd)}")

    try:
        env = os.environ.copy()
        # Avoid plugin import conflicts in varied environments
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = env.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
        result = subprocess.run(cmd, check=False, env=env)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def main():
    """Main test runner function."""
    print("🚀 YouTube Summarizer Test Suite")
    print("=" * 50)

    # Check if we're in the right directory
    project_root = Path(__file__).parent.parent
    if not (project_root / "app.py").exists():
        print("❌ app.py not found. Please run from the project root directory.")
        sys.exit(1)

    # Change to project root
    os.chdir(project_root)

    # Parse command line arguments
    test_type = "all"
    coverage = False

    if len(sys.argv) > 1:
        test_type = sys.argv[1]

    if "--coverage" in sys.argv:
        coverage = True

    print(f"📋 Test suite options:")
    print(f"   • all: Run all tests")
    print(f"   • api: Run API endpoint tests only")
    print(f"   • streaming: Run streaming tests only")
    print(f"   • meta_language: Run meta-language avoidance tests only")
    print(f"   • unit: Run unit tests only")
    print(f"   • integration: Run integration tests only")
    print(f"")
    print(f"📋 Environment Requirements:")
    print(f"   • Integration tests require API keys:")
    print(f"     - GEMINI_API_KEY or OPENROUTER_API_KEY (for summarization)")
    print(f"     - APIFY_API_KEY (for video scraping)")
    print(f"   • Unit tests run without external dependencies")
    print(f"")
    print(f"🎯 Running: {test_type} tests")

    # Run tests
    success = run_tests(test_type=test_type, coverage=coverage, verbose=True)

    if success:
        print("\n✅ All tests passed!")

        # Show coverage report location if generated
        if coverage:
            print("📊 Coverage report generated:")
            print("   • HTML: htmlcov/index.html")
            print("   • Terminal output above")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
