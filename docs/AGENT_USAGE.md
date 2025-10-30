# Fraud Analysis Agent Usage

## Running the Fraud Analysis Agent

The fraud analysis agent has been refactored to use the fraud detector tool. Here's how to run it:

### Option 1: Run from project root (Recommended)

```bash
# From the project root directory
python test_agent_import.py
```

### Option 2: Run the agent directly

```bash
# From the project root directory
python -m src.agents.fraud_analysis_agent
```

### Option 3: Run with specific Python executable

If you have multiple Python installations, you might need to specify the executable:

```bash
# Using python3
python3 -m src.agents.fraud_analysis_agent

# Or using full path
/path/to/python -m src.agents.fraud_analysis_agent
```

## What the Agent Does

The fraud analysis agent now includes:

1. **Fraud Detection Tool**: Uses machine learning with SHAP explanations
2. **Risk Assessment**: Provides detailed fraud risk scores and confidence levels
3. **Explanations**: Explains why transactions are flagged as fraudulent
4. **Recommendations**: Generates actionable recommendations for risk mitigation
5. **Feature Analysis**: Identifies top contributing factors to fraud risk

## Example Usage

The agent can analyze transactions and provide comprehensive fraud analysis including:

- Fraud probability score (0-1)
- Risk level (LOW/MEDIUM/HIGH)
- Confidence level
- Top risk factors with SHAP explanations
- Specific recommendations for risk mitigation

## Troubleshooting

If you encounter import errors:

1. Make sure you're running from the project root directory
2. Ensure all dependencies are installed: `pip install -r requirements.txt`
3. Make sure the model artifacts exist in the `artifacts/` directory
4. Check that SHAP is installed for enhanced explanations: `pip install shap`
