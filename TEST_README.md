# YouTube Summarizer API Testing Guide

This guide explains how to run the comprehensive test suite for the YouTube Summarizer API.

## 📋 Test Overview

The test suite includes comprehensive coverage of all API endpoints:

- **Health & Info Tests**: Root endpoint, health check
- **URL Validation Tests**: Valid/invalid YouTube URLs
- **Video Info Tests**: Metadata extraction via Apify API
- **Transcript Tests**: Multi-tier extraction (Apify → Gemini fallback)
- **Summary Tests**: AI-powered text summarization
- **Process Tests**: Complete video processing workflow
- **Generate Tests**: Master endpoint with full analysis
- **Helper Function Tests**: Direct testing of utility functions
- **Error Handling Tests**: Edge cases and error scenarios

## 🚀 Quick Start

### Option 1: Using the Test Runner Script
```bash
python run_tests.py
```

### Option 2: Using pytest directly
```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest test_api.py -v

# Run specific test class
pytest test_api.py::TestHealthAndInfo -v

# Run with coverage
pytest test_api.py --cov=app --cov=youtube_summarizer --cov-report=html
```

## 📁 Test Files

- **`test_api.py`** - Main test suite with comprehensive API coverage
- **`pytest.ini`** - Pytest configuration
- **`requirements-test.txt`** - Test dependencies
- **`run_tests.py`** - Interactive test runner script

## 🔧 Test Configuration

The tests use mocking to avoid actual API calls:

- **Apify API**: Mocked with `mock_youtube_scrapper_result` fixture
- **Gemini API**: Mocked with `mock_analysis_result` fixture
- **Environment Variables**: Patched for different scenarios

## 🧪 Test Categories

### Unit Tests
```bash
pytest test_api.py::TestHelperFunctions -v
```

### Integration Tests
```bash
pytest test_api.py::TestProcess -v
pytest test_api.py::TestGenerate -v
```

### Error Handling Tests
```bash
pytest test_api.py::TestErrorHandling -v
```

## 📊 Test Coverage

Run with coverage reporting:
```bash
pytest test_api.py --cov=app --cov=youtube_summarizer --cov-report=html --cov-report=term
```

This generates:
- Terminal coverage report
- HTML coverage report in `htmlcov/` directory

## 🔍 Test Scenarios Covered

### ✅ Success Cases
- Valid YouTube URL processing
- Successful Apify API responses
- Gemini fallback functionality
- Complete workflow execution
- Example mode responses

### ❌ Error Cases
- Invalid YouTube URLs
- Missing API keys
- API quota exceeded
- Network timeouts
- Invalid JSON requests

### 🔄 Fallback Testing
- Apify API failure → Gemini fallback
- Missing APIFY_API_KEY → Direct Gemini processing
- Various error conditions and recovery

## 🎯 Running Specific Tests

```bash
# Test specific endpoint
pytest test_api.py::TestVideoInfo::test_video_info_success -v

# Test error handling
pytest test_api.py::TestErrorHandling -v

# Test with specific markers (if added)
pytest test_api.py -m "not slow" -v
```

## 🛠️ Debugging Failed Tests

1. **Run with detailed output**:
   ```bash
   pytest test_api.py -v -s --tb=long
   ```

2. **Run single failing test**:
   ```bash
   pytest test_api.py::TestClass::test_method -v -s
   ```

3. **Check mock configurations**: Ensure mocks match actual API responses

## 📝 Adding New Tests

When adding new features:

1. **Add test fixtures** for new data structures
2. **Mock external dependencies** (APIs, file operations)
3. **Test both success and failure cases**
4. **Follow naming convention**: `test_<feature>_<scenario>`

## 🔐 Environment Variables for Testing

The tests automatically mock environment variables, but you can set real ones for integration testing:

```bash
export APIFY_API_KEY="your_test_key"
export GEMINI_API_KEY="your_test_key"
pytest test_api.py -v
```

## 📈 CI/CD Integration

For continuous integration, use:

```bash
# In your CI pipeline
pip install -r requirements-test.txt
pytest test_api.py --cov=app --cov=youtube_summarizer --cov-report=xml
```

This generates coverage reports compatible with most CI systems.