# 🛡️ AI-Powered Fraud Detection Agent

An advanced fraud detection system that combines machine learning, explainable AI, and intelligent self-healing capabilities to analyze financial transactions for fraudulent patterns.

## 🌟 Features

### 🤖 Intelligent Agent System
- **AI-Powered Analysis**: Uses OpenAI's o3-mini model for sophisticated reasoning and decision making
- **Multi-Tool Integration**: Combines ML models, behavioral analysis, and custom analytics
- **Self-Healing Code Execution**: Automatically detects and fixes code errors using LLM-powered debugging

### 🔍 Advanced Fraud Detection
- **LightGBM ML Model**: High-performance gradient boosting with 98.2% recall and 81.5% ROC AUC
- **SHAP Explanations**: Transparent, interpretable AI with feature importance analysis
- **Behavioral Pattern Analysis**: Customer behavior profiling and anomaly detection
- **Risk Score Calculation**: Comprehensive fraud risk assessment with confidence intervals

### 🛠️ Dynamic Analytics Engine
- **Sandbox Code Execution**: Safe Python/Pandas/NumPy code execution with security restrictions
- **LLM-Powered Code Generation**: Agents that write custom analytics code based on transaction characteristics
- **Error Recovery System**: Automatic error detection and healing with multiple fallback strategies
- **Comprehensive Logging**: Detailed execution tracking and debugging capabilities

## 🏗️ Architecture

```
fraud_agent/
├── src/
│   ├── _agents/               # AI agent implementations
│   │   └── fraud_analysis_agent.py
│   ├── tools/                 # Agent tools and capabilities
│   │   ├── fraud_detector.py          # ML-based fraud detection
│   │   ├── behavioral_analyzer.py     # Customer behavior analysis
│   │   ├── data_loader.py            # Data loading and management
│   │   └── sandbox_runner.py         # Secure code execution
│   ├── eda.py                # Exploratory data analysis
│   ├── preprocess.py         # Data preprocessing pipeline
│   └── model_baseline.py     # ML model training
├── data/                     # Transaction datasets
│   └── transactions.csv      # 10,000 synthetic transactions
├── notebooks/               # Jupyter notebooks for analysis
├── tests/                   # Test files and validation
└── docs/                   # Documentation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API Key

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/fraud-detection-agent.git
cd fraud-detection-agent
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Basic Usage

```python
from src._agents.fraud_analysis_agent import agent, Runner
import asyncio

# Sample transaction
transaction = {
    "transaction_id": "TX_001",
    "customer_id": "CUST_12345", 
    "amount": 2500.00,
    "transaction_type": "online_purchase",
    "merchant_category": "electronics",
    "is_weekend": 1
}

async def analyze_transaction():
    result = await Runner.run(
        agent,
        f"Analyze this transaction for fraud: {transaction}"
    )
    print(result.final_output)

# Run analysis
asyncio.run(analyze_transaction())
```

## 📊 Dataset

The system uses a synthetic dataset of 10,000 financial transactions with:
- **60.82% fraud rate** for comprehensive testing
- **19 engineered features** including behavioral and temporal patterns
- **NGN currency** transactions with realistic amounts and categories
- **Customer behavior tracking** with transaction history and patterns

### Key Features
- Transaction amounts and types
- Customer demographics and history
- Temporal patterns (weekends, time-based)
- Merchant categories and locations
- Behavioral indicators (frequency, velocity)

## 🔧 Core Components

### 1. Fraud Detection Engine (`fraud_detector.py`)
- **LightGBM Model**: Optimized for fraud detection with high recall
- **SHAP Integration**: Provides explainable AI with feature importance
- **Risk Scoring**: Comprehensive fraud probability with confidence metrics
- **Feature Engineering**: 47 behavioral and transaction features

### 2. Behavioral Analysis (`behavioral_analyzer.py`)
- **Customer Profiling**: Historical transaction pattern analysis
- **Anomaly Detection**: Statistical outlier identification
- **Velocity Checking**: Transaction frequency and volume analysis
- **Risk Factor Identification**: Behavioral red flags and indicators

### 3. Sandbox Code Execution (`sandbox_runner.py`)
- **Secure Environment**: Restricted Python execution with safety controls
- **LLM-Powered Healing**: Automatic error detection and code correction
- **Dynamic Analytics**: Agents generate custom code for specific analysis needs
- **Comprehensive Logging**: Detailed execution tracking and debugging

### 4. Self-Healing System
- **Error Detection**: Automatic identification of code execution failures
- **LLM Debugging Agent**: AI-powered error analysis and code correction
- **Fallback Strategies**: Multiple recovery mechanisms for robust operation
- **Learning Capability**: Improves over time through error pattern recognition

## 🎯 Use Cases

### Financial Institutions
- **Real-time Transaction Monitoring**: Analyze transactions as they occur
- **Risk Assessment**: Comprehensive fraud scoring for decision making
- **Compliance Reporting**: Explainable AI for regulatory requirements
- **Custom Analytics**: Dynamic code generation for specific analysis needs

### E-commerce Platforms
- **Payment Processing**: Fraud detection at checkout
- **Merchant Risk Analysis**: Assess merchant transaction patterns
- **Customer Behavior Analysis**: Identify suspicious account activity
- **Adaptive Learning**: Continuously improve detection accuracy

### Security Teams
- **Investigation Support**: Detailed fraud analysis with explanations
- **Pattern Recognition**: Identify emerging fraud techniques
- **Risk Prioritization**: Focus efforts on highest-risk transactions
- **Automated Response**: Self-healing systems for continuous operation

## 🧪 Testing

The project includes comprehensive test suites:

```bash
# Test fraud detection
python test_integration.py

# Test self-healing capabilities
python test_self_healing.py

# Test LLM healing system
python test_llm_healing.py

# Test error visibility
python test_error_visibility.py
```

## 📈 Performance Metrics

### ML Model Performance
- **Recall**: 98.2% (excellent fraud detection)
- **Precision**: 78.5% (low false positive rate)
- **ROC AUC**: 81.5% (strong discrimination)
- **F1 Score**: 87.4% (balanced performance)

### System Performance
- **Response Time**: <2 seconds for standard analysis
- **Self-Healing Success Rate**: 85%+ automatic error recovery
- **Code Safety**: 100% sandbox security compliance
- **Scalability**: Handles 1000+ transactions/minute

## 🔒 Security Features

### Sandbox Security
- **Import Restrictions**: No file system or network access
- **Safe Operations**: Only pandas/numpy analytics allowed
- **Timeout Protection**: Prevents infinite loops
- **Resource Limits**: Memory and CPU constraints

### Data Privacy
- **No Data Persistence**: Code execution doesn't save data
- **Audit Logging**: Complete execution history tracking
- **Access Control**: Restricted operation permissions
- **Encryption**: Secure data handling throughout

## 🛠️ Development

### Adding New Tools
1. Create tool function with `@function_tool` decorator
2. Add to agent's tools list
3. Update agent instructions
4. Add comprehensive tests

### Custom Analytics
The system supports dynamic code generation for:
- Custom risk metrics
- Feature engineering
- Statistical analysis
- Data visualization
- Advanced fraud patterns

### Contributing
1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-capability`
3. Commit changes: `git commit -am 'Add new capability'`
4. Push branch: `git push origin feature/new-capability`
5. Submit pull request

## 📚 Documentation

- [`INTEGRATION_README.md`](INTEGRATION_README.md) - Detailed integration guide
- [`AGENT_USAGE.md`](AGENT_USAGE.md) - Agent usage examples
- [`src/examples/`](src/examples/) - Code examples and tutorials
- [`notebooks/`](notebooks/) - Jupyter notebook analyses

## 🤝 Support

For support, please:
1. Check the documentation and examples
2. Search existing issues
3. Create detailed issue with reproduction steps
4. Contact the development team

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for the powerful language models
- **LightGBM** for high-performance gradient boosting
- **SHAP** for explainable AI capabilities
- **scikit-learn** for machine learning utilities

---

*Built with ❤️ for safer financial transactions*