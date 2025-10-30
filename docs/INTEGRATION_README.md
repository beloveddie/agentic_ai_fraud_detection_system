# 🚀 Integrated Fraud Analysis Agent System

A comprehensive AI-powered fraud detection system that combines machine learning models with behavioral pattern analysis and code-as-action capabilities.

## 🎯 **System Overview**

This integrated system provides:
- **ML-based fraud detection** using LightGBM with SHAP explanations
- **Behavioral pattern analysis** for customer behavior insights
- **Combined risk assessment** merging ML predictions with behavioral insights
- **Code-as-action paradigm** enabling dynamic analysis code generation
- **Comprehensive fraud analysis** with actionable recommendations

## 🏗️ **Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    Fraud Analysis Agent                     │
├─────────────────────────────────────────────────────────────┤
│  • OpenAI Agents SDK Integration                            │
│  • Multi-tool Orchestration                                 │
│  • Intelligent Analysis Workflow                            │
└─────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
┌───────────────▼─────┐ ┌───────▼───────┐ ┌─────▼─────────────┐
│   Fraud Detector    │ │   Behavioral  │ │   Data Loader     │
│                     │ │   Analyzer   │ │                   │
├─────────────────────┤ ├───────────────┤ ├───────────────────┤
│ • LightGBM Model    │ │ • Spending    │ │ • CSV Loading     │
│ • SHAP Explanations │ │   Patterns    │ │ • Customer History│
│ • Risk Scoring      │ │ • Temporal    │ │ • Data Filtering  │
│ • Confidence Levels │ │   Patterns    │ │ • Caching         │
│ • Recommendations   │ │ • Geographic  │ │ • Statistics      │
│                     │ │   Patterns    │ │                   │
│                     │ │ • Device/     │ │                   │
│                     │ │   Channel     │ │                   │
│                     │ │ • Frequency   │ │                   │
│                     │ │ • Risk        │ │                   │
│                     │ │   Indicators  │ │                   │
└─────────────────────┘ └───────────────┘ └───────────────────┘
```

## 🔧 **Core Components**

### 1. **Fraud Detector Tool** (`src/tools/fraud_detector.py`)
- **ML Model**: LightGBM classifier with preprocessing pipeline
- **SHAP Integration**: Explainable AI with feature contributions
- **Risk Assessment**: Probability scores, confidence levels, risk categories
- **Recommendations**: Actionable fraud mitigation suggestions

### 2. **Behavioral Analyzer Tool** (`src/tools/behavioral_analyzer.py`)
- **Spending Patterns**: Acceleration, consistency, unusual spikes
- **Temporal Patterns**: Time preferences, consistency, deviations
- **Geographic Patterns**: Location consistency, travel detection
- **Device/Channel Patterns**: Usage consistency, change rates
- **Transaction Type Patterns**: Preferences, diversity, loyalty
- **Frequency Patterns**: Transaction frequency, burst detection
- **Risk Indicators**: Behavioral risk scoring and anomaly detection

### 3. **Data Loader Utility** (`src/tools/data_loader.py`)
- **Customer History**: Historical transaction retrieval
- **Data Filtering**: Flexible filtering and sampling
- **Performance**: Caching and optimization
- **Statistics**: Dataset analysis and insights

### 4. **Fraud Analysis Agent** (`src/agents/fraud_analysis_agent.py`)
- **Multi-tool Integration**: Orchestrates all analysis tools
- **Intelligent Workflow**: Combines ML and behavioral insights
- **Comprehensive Analysis**: Provides unified fraud assessment
- **Actionable Output**: Clear recommendations and explanations

## 🚀 **Key Features**

### **Machine Learning Capabilities**
- ✅ LightGBM fraud detection model
- ✅ SHAP explanations for transparency
- ✅ Confidence scoring and risk levels
- ✅ Feature importance analysis
- ✅ Preprocessing pipeline integration

### **Behavioral Analysis Capabilities**
- ✅ Comprehensive pattern analysis across 6 domains
- ✅ Anomaly detection and risk scoring
- ✅ Customer profiling and stability assessment
- ✅ Data sufficiency and confidence tracking
- ✅ Rich structured data for code-as-action

### **Integration Features**
- ✅ Seamless tool orchestration
- ✅ Combined risk assessment
- ✅ Context-aware recommendations
- ✅ Confidence level aggregation
- ✅ Comprehensive fraud insights

### **Code-as-Action Ready**
- ✅ Rich structured behavioral data
- ✅ Raw pattern data preservation
- ✅ Extensible analysis framework
- ✅ Dynamic code generation support
- ✅ Custom analysis capabilities

## 📊 **Analysis Domains**

### **1. Spending Patterns**
- Average amounts (7d, 30d, 90d)
- Spending acceleration rates
- Consistency scoring
- Unusual spending spike detection
- Spending trend analysis

### **2. Temporal Patterns**
- Preferred transaction hours/days
- Time consistency scoring
- Unusual timing detection
- Night/weekend activity ratios
- Time deviation analysis

### **3. Geographic Patterns**
- Primary location identification
- Location diversity scoring
- Location consistency analysis
- Unusual location detection
- Impossible travel detection

### **4. Device/Channel Patterns**
- Preferred devices and channels
- Consistency scoring
- Change rate analysis
- New device detection
- Unusual usage patterns

### **5. Transaction Type Patterns**
- Preferred transaction types
- Merchant category preferences
- Diversity scoring
- Unusual type detection
- Merchant loyalty analysis

### **6. Frequency Patterns**
- Transaction frequency metrics
- Frequency trend analysis
- Acceleration detection
- Burst activity identification
- Consistency scoring

## 🎯 **Risk Assessment**

### **Risk Levels**
- **LOW**: Minimal risk indicators
- **MEDIUM**: Moderate risk factors present
- **HIGH**: Significant risk indicators
- **CRITICAL**: Multiple high-risk factors

### **Risk Factors**
- Unusual spending spikes
- High spending acceleration
- Inconsistent spending patterns
- Unusual timing patterns
- Geographic anomalies
- Device/channel changes
- Frequency spikes
- Burst activity

### **Confidence Scoring**
- Data sufficiency assessment
- Analysis confidence levels
- Quality indicators
- Reliability metrics

## 🔧 **Installation & Setup**

### **Prerequisites**
```bash
pip install -r requirements.txt
```

### **Required Dependencies**
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `lightgbm` - ML model
- `shap` - Explainable AI
- `scikit-learn` - ML utilities
- `openai-agents` - Agent framework
- `python-dotenv` - Environment variables

### **Model Artifacts**
Ensure the following files exist in `artifacts/`:
- `lgbm_model.joblib` - Trained LightGBM model
- `preprocessor.joblib` - Preprocessing pipeline
- `preprocess_meta.json` - Feature metadata

## 🚀 **Usage Examples**

### **1. Basic Agent Usage**
```python
from agents.fraud_analysis_agent import agent
from agents import Runner

# Run comprehensive fraud analysis
result = await Runner.run(agent, """
Please analyze this transaction for fraud risk:
- Transaction ID: TXN_001
- Amount: 15000 NGN
- Customer: CUST_123
- Type: transfer
- Channel: web
""")
```

### **2. Behavioral Analysis Only**
```python
from tools.behavioral_analyzer import analyze_behavioral_patterns

# Analyze behavioral patterns
result = await analyze_behavioral_patterns(
    transaction=current_transaction,
    customer_history=historical_transactions,
    analysis_window_days=30
)
```

### **3. Customer History Retrieval**
```python
from tools.data_loader import get_customer_history

# Get customer history
history = get_customer_history(
    customer_id="CUST_123",
    limit=50,
    exclude_current_transaction="TXN_001"
)
```

## 📈 **Performance & Scalability**

### **Optimization Features**
- **Caching**: Data loader with TTL-based caching
- **Batch Processing**: Efficient data handling
- **Memory Management**: Optimized data structures
- **Error Handling**: Robust fallback mechanisms

### **Scalability Considerations**
- **Modular Design**: Independent tool components
- **Extensible Framework**: Easy to add new analysis tools
- **Code-as-Action**: Dynamic analysis capabilities
- **API Ready**: Structured for service integration

## 🔍 **Testing & Validation**

### **Test Scripts**
- `test_integration.py` - Integration testing
- `demo_integrated_system.py` - Comprehensive demonstration

### **Test Coverage**
- ✅ Import validation
- ✅ Tool functionality testing
- ✅ Agent integration testing
- ✅ Behavioral analysis validation
- ✅ Data loader testing

## 🎯 **Future Enhancements**

### **Planned Features**
- **Network Analysis**: Transaction relationship analysis
- **Anomaly Detection**: Advanced statistical methods
- **Rule Engine**: Dynamic rule-based analysis
- **Real-time Processing**: Stream processing capabilities
- **API Integration**: RESTful service endpoints

### **Code-as-Action Extensions**
- **Custom Analysis Functions**: LLM-generated analysis code
- **Dynamic Feature Engineering**: Runtime feature creation
- **Adaptive Thresholds**: Self-tuning risk parameters
- **Custom Risk Models**: Domain-specific risk scoring

## 📚 **Documentation**

### **API Reference**
- `src/tools/fraud_detector.py` - Fraud detection API
- `src/tools/behavioral_analyzer.py` - Behavioral analysis API
- `src/tools/data_loader.py` - Data loading API
- `src/agents/fraud_analysis_agent.py` - Agent integration

### **Examples**
- `src/examples/example_usage.py` - Basic usage examples
- `demo_integrated_system.py` - Comprehensive demonstrations

## 🤝 **Contributing**

### **Development Setup**
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Ensure model artifacts are in place
4. Run tests: `python test_integration.py`

### **Code Standards**
- Follow PEP 8 style guidelines
- Include comprehensive docstrings
- Add type hints for all functions
- Write tests for new functionality

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 **Support**

For questions, issues, or contributions:
- Create an issue in the repository
- Check existing documentation
- Review example code and demos

---

**🎉 Ready to detect fraud with AI-powered behavioral analysis!**
