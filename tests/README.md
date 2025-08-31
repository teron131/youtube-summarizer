## 📋 Test Configuration

### Environment Variables for Integration Tests
```bash
export GEMINI_API_KEY="your-gemini-api-key"
export OPENROUTER_API_KEY="your-openrouter-api-key"
export APIFY_API_KEY="your-apify-api-key"
```

**Note:** You need either `GEMINI_API_KEY` or `OPENROUTER_API_KEY` for summarization tests.

### Pytest Configuration
See `pytest.ini` for test paths, markers, and options.

## 🔧 Warning Fixes Applied

### ✅ **Fixed Warnings:**

1. **Pytest Marker Warnings**
   - **Issue**: `PytestUnknownMarkWarning: Unknown pytest.mark.integration`
   - **Fix**: Added `pytest_configure()` function in `conftest.py` to register custom markers
   - **Result**: No more marker warnings

2. **LangGraph Deprecation Warnings**
   - **Issue**: `LangGraphDeprecatedSinceV05: input/output deprecated`
   - **Fix**: Updated `summarizer.py` to use `input_schema` and `output_schema`
   - **Result**: No more LangGraph deprecation warnings

### ✅ **Clean Test Output:**

```bash
# Before fixes - lots of warnings
8 passed, 13 warnings in 2:40

# After fixes - clean output
8 passed in 2:40
```

## 📊 Test Fixtures

### Available Fixtures (in conftest.py)
- `client` - FastAPI test client
- `mock_channel` - Mock YouTube channel object
- `mock_youtube_scrapper_result` - Mock scraping result
- `mock_analysis_result` - Mock AI analysis result
- `clean_env` - Clean environment for testing

## 🎯 Best Practices

1. **Use appropriate markers**: Mark integration tests that require API keys
2. **Mock external dependencies**: Use fixtures for predictable testing
3. **Test error conditions**: Include negative test cases
4. **Verify response structure**: Check both success and error responses
5. **Test streaming**: Verify SSE format and multiple chunks

## 🔍 Debugging Tests

```bash
# Run with verbose output
pytest -v

# Run specific test
pytest tests/test_api.py::TestHealthAndInfo::test_root_endpoint

# Run with debugging
pytest --pdb

# Run with coverage and open HTML report
pytest --cov --cov-report=html && open htmlcov/index.html
```