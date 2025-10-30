# 📁 Project Structure

```
fraud_agent/
├── 📄 README.md                    # Main project documentation
├── 📄 LICENSE                      # MIT license
├── 📄 CHANGELOG.md                 # Version history and changes
├── 📄 requirements.txt             # Python dependencies
├── 📄 setup.py                     # Setup and validation script
├── 📄 example_usage.py             # Usage examples and demos
├── 📄 .env.example                 # Environment configuration template
├── 📄 .gitignore                   # Git ignore patterns
├── 📄 generate_transaction_dataset.py  # Data generation script
├── 📄 main.py                      # Legacy main entry point
├── 
├── 📂 .github/                     # GitHub Actions CI/CD
│   └── 📂 workflows/
│       └── 📄 ci.yml               # Continuous integration
├── 
├── 📂 src/                         # Source code
│   ├── 📂 _agents/                 # AI agent implementations
│   │   └── 📄 fraud_analysis_agent.py  # Main fraud detection agent
│   ├── 📂 tools/                   # Agent tools and capabilities
│   │   ├── 📄 fraud_detector.py    # ML-based fraud detection
│   │   ├── 📄 behavioral_analyzer.py  # Customer behavior analysis
│   │   ├── 📄 data_loader.py       # Data loading and management
│   │   └── 📄 sandbox_runner.py    # Secure code execution engine
│   ├── 📂 examples/                # Code examples and tutorials
│   ├── 📄 eda.py                   # Exploratory data analysis
│   ├── 📄 preprocess.py            # Data preprocessing pipeline
│   └── 📄 model_baseline.py        # ML model training
├── 
├── 📂 data/                        # Transaction datasets
│   └── 📄 transactions.csv         # 10,000 synthetic transactions
├── 
├── 📂 tests/                       # Test suite
│   ├── 📄 README.md                # Test documentation
│   ├── 📄 test_integration.py      # Integration tests
│   ├── 📄 test_self_healing.py     # Self-healing tests
│   ├── 📄 test_llm_healing.py      # LLM healing tests
│   ├── 📄 test_error_visibility.py # Error handling tests
│   └── 📄 test_*.py                # Additional test files
├── 
├── 📂 docs/                        # Documentation
│   ├── 📄 INTEGRATION_README.md    # Integration guide
│   └── 📄 AGENT_USAGE.md           # Agent usage documentation
├── 
├── 📂 notebooks/                   # Jupyter notebooks
│   └── 📄 *.ipynb                  # Analysis and exploration notebooks
├── 
└── 📂 artifacts/                   # Model artifacts and outputs
    └── 📄 *.pkl                    # Trained models and preprocessors
```

## 🏗️ Architecture Overview

### Core Components

#### 🤖 AI Agent Layer (`src/_agents/`)
- **fraud_analysis_agent.py**: Main orchestrator using OpenAI o3-mini
- Coordinates multiple tools for comprehensive analysis
- Provides natural language interface for fraud detection

#### 🛠️ Tool Layer (`src/tools/`)
- **fraud_detector.py**: LightGBM model + SHAP explanations  
- **behavioral_analyzer.py**: Customer pattern analysis
- **data_loader.py**: Efficient data management
- **sandbox_runner.py**: Secure Python code execution with self-healing

#### 📊 ML Pipeline (`src/`)
- **preprocess.py**: Feature engineering and data preparation
- **model_baseline.py**: Model training and evaluation
- **eda.py**: Exploratory data analysis

### Data Flow

```
Transaction Input → Agent → Tools → Analysis Results
                     ↓
                ML Detector ← → Behavioral Analyzer
                     ↓              ↓
                SHAP Explanations + Risk Patterns
                     ↓
              Sandbox Code Execution (Optional)
                     ↓
              Comprehensive Fraud Assessment
```

### Key Features

1. **🔍 Multi-Modal Analysis**
   - Machine learning predictions
   - Behavioral pattern detection  
   - Dynamic code-based analytics
   - SHAP explanations for transparency

2. **🛡️ Self-Healing System**
   - Automatic error detection
   - LLM-powered code correction
   - Multiple fallback strategies
   - Comprehensive logging

3. **🔒 Security & Safety**
   - Sandboxed code execution
   - Input validation and sanitization
   - Resource limits and timeouts
   - Audit trails and monitoring

4. **📈 Performance & Scalability**
   - Optimized ML models (98.2% recall)
   - Efficient data processing
   - Concurrent analysis capability
   - Sub-2-second response times

### Integration Points

- **API Integration**: Via OpenAI Agents SDK
- **Data Sources**: CSV, databases, real-time streams
- **Output Formats**: JSON, structured reports, explanations
- **Monitoring**: Comprehensive logging and metrics