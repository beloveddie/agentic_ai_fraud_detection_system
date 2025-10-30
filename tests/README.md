# Test Suite

This directory contains comprehensive tests for the fraud detection agent system.

## Test Files

### Core Functionality Tests
- `test_integration.py` - Integration tests for the complete fraud detection pipeline
- `test_agent_import.py` - Tests for agent import and basic functionality

### Self-Healing System Tests  
- `test_self_healing.py` - Tests for basic self-healing capabilities
- `test_llm_healing.py` - Tests for LLM-powered error recovery
- `test_llm_healing_sandbox_safety.py` - Tests for enhanced sandbox safety
- `test_direct_healing.py` - Direct testing of healing functionality
- `test_error_visibility.py` - Tests for error reporting and visibility

## Running Tests

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Run Specific Test Categories
```bash
# Core functionality
python tests/test_integration.py

# Self-healing capabilities
python tests/test_self_healing.py
python tests/test_llm_healing.py

# Error handling
python tests/test_error_visibility.py
```

### Prerequisites
- Ensure virtual environment is activated
- Set OPENAI_API_KEY in .env file
- Run from project root directory

## Test Coverage

The test suite covers:
- ✅ ML model fraud detection accuracy
- ✅ SHAP explanation generation  
- ✅ Behavioral pattern analysis
- ✅ Sandbox code execution safety
- ✅ LLM-powered error recovery
- ✅ Dynamic code generation
- ✅ Error visibility and logging
- ✅ Agent tool integration