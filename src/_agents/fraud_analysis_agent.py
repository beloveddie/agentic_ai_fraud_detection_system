#!/usr/bin/env python3
"""
🛡️ AI-Powered Fraud Detection Agent

This module implements an advanced fraud detection agent that combines:
- Machine learning models for transaction risk scoring
- SHAP explanations for interpretable AI
- Behavioral pattern analysis for customer profiling
- Self-healing code execution for custom analytics

The agent uses OpenAI's o3-mini model for reasoning and decision making,
integrated with specialized tools for comprehensive fraud analysis.

Author: Fraud Detection Team
Version: 1.0.0
License: MIT
"""

from agents import Agent, Runner
import asyncio
import sys
import os

from dotenv import load_dotenv
load_dotenv()

# Add the src directory to the path so we can import from tools
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

try:
    from tools.fraud_detector import detect_fraud
    from tools.behavioral_analyzer import analyze_behavioral_patterns
    from tools.data_loader import get_customer_history
    from tools.sandbox_runner import execute_python_code_action, execute_python_code_action_with_healing
except ImportError as e:
    print(f"❌ Failed to import tools: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

tools = [detect_fraud, analyze_behavioral_patterns, get_customer_history, execute_python_code_action, execute_python_code_action_with_healing]

agent = Agent(name="fraud_analysis_agent",
              instructions="""
You are a Fraud Analysis Agent that specializes in detecting fraudulent transactions using advanced machine learning models, SHAP explanations, behavioral pattern analysis, and you can also execute custom Python analytics code safely in a sandbox.

Your capabilities include:
- Analyzing transaction data for fraud indicators using ML models
- Providing detailed fraud risk assessments with confidence scores
- Explaining fraud predictions using SHAP (SHapley Additive exPlanations)
- Analyzing customer behavioral patterns for fraud detection
- Generating actionable recommendations based on risk factors
- Identifying top contributing factors to fraud risk
- Combining ML predictions with behavioral insights for comprehensive analysis
- Executing custom Python/Pandas/Numpy analytics using the `execute_python_code_action` tool (see below for safe code policy)

When analyzing a transaction:
1. Use the detect_fraud tool to get ML-based fraud analysis with SHAP explanations
2. Use the analyze_behavioral_patterns tool to analyze customer behavior patterns
3. Use the get_customer_history tool to retrieve historical transactions for behavioral analysis
4. Use the execute_python_code_action tool for custom analytics, feature engineering, or advanced metrics — ALWAYS follow the safe code policy described below
5. Combine ML, behavioral, and code-action results for comprehensive fraud assessment
6. Explain the results in clear, actionable terms
7. Highlight the most important risk factors from both perspectives
8. Provide specific recommendations for risk mitigation
9. Explain confidence levels and what they mean

Behavioral Analysis Integration:
- Always retrieve customer history for behavioral analysis when possible
- Compare current transaction against customer's historical patterns
- Identify behavioral anomalies (unusual spending, timing, location, device usage)
- Combine behavioral risk scores, ML predictions, and (when needed) code-action analytics for enhanced accuracy
- Provide context-aware recommendations based on behavioral and code-driven analytical insights

Always provide detailed explanations of why a transaction might be flagged as fraudulent, including:
- ML model predictions with SHAP explanations
- Behavioral pattern deviations and anomalies
- Results from custom Python code actions (as needed), e.g., new metrics
- Specific features that contributed to the risk score
- Confidence levels and data sufficiency for all analyses
- Actionable recommendations for risk mitigation

Important! You have access to the tool `execute_python_code_action` for custom Python/Pandas/Numpy analytical code:
- Code MUST use only the provided input (`input_data`) and standard math/analytics libs (pd, np).
- No `import`, no OS/sys/file/network operations allowed. Code violating this is rejected and flagged.
- The final answer must always be assigned to a variable named `result`.
- Print to stdout if you want output captured.
- Use this tool for advanced, deep, or ad-hoc analysis directly on a transaction's input_data.
""",
              model="o3-mini",
              tools=tools)


# async def main():
#     # Example usage of the fraud analysis agent
#     print("🔍 Fraud Analysis Agent - Example Usage")
#     print("=" * 50)
    
#     # Test 1: Ask about capabilities
#     print("\n1. Testing agent capabilities:")
#     result = await Runner.run(agent, "what can you help me do?")
#     print(result.final_output)
    
#     # Test 2: Analyze a sample transaction with behavioral analysis
#     print("\n2. Analyzing a sample transaction with behavioral analysis:")
#     sample_transaction = {
#         "transaction_id": "TXN_001",
#         "customer_id": "CUST_0383", 
#         "amount": 15000.0,
#         "currency": "NGN",
#         "transaction_type": "transfer",
#         "channel": "web",
#         "merchant_category": "electronics",
#         "device_id": "DEV_456",
#         "ip_address": "192.168.1.100",
#         "location_country": "Nigeria",
#         "location_city": "Lagos",
#         "time": "2025-01-22 14:30:00",
#         "account_age_days": 45,
#         "num_prev_transactions_24h": 8,
#         "avg_transaction_amount_7d": 500.0,
#         "device_change_rate_30d": 0.8,
#         "is_foreign": 0,
#         "is_blacklisted_merchant": 0
#     }
    
#     analysis_prompt = f"""
#     Please perform a comprehensive fraud analysis for this transaction, including both ML-based fraud detection and behavioral pattern analysis:
    
#     Transaction Details:
#     - Transaction ID: {sample_transaction['transaction_id']}
#     - Customer ID: {sample_transaction['customer_id']}
#     - Amount: {sample_transaction['amount']} {sample_transaction['currency']}
#     - Type: {sample_transaction['transaction_type']}
#     - Channel: {sample_transaction['channel']}
#     - Merchant Category: {sample_transaction['merchant_category']}
#     - Device ID: {sample_transaction['device_id']}
#     - Location: {sample_transaction['location_city']}, {sample_transaction['location_country']}
#     - Time: {sample_transaction['time']}
#     - Account Age: {sample_transaction['account_age_days']} days
    
#     Please:
#     1. First retrieve the customer's transaction history for behavioral analysis
#     2. Perform ML-based fraud detection with SHAP explanations
#     3. Analyze behavioral patterns and anomalies
#     4. Combine both analyses for a comprehensive fraud assessment
#     5. Provide specific recommendations based on the combined insights
#     6. Explain the confidence levels and data sufficiency for both analyses
#     7. Use the `execute_python_code_action` tool to perform at least two custom analytical code.
#     8. Provide the results of the custom analytical code in the response.
#     9. Explain the results of the custom analytical code in the response.
#     """
    
#     result = await Runner.run(agent, analysis_prompt)
#     print(result.final_output)

# Dynamic analytical agent that generates Python code based on transaction characteristics
async def simple_agent():
    transaction_data = {
        "transaction_id": "TXN_001",
        "customer_id": "CUST_0383", 
        "amount": 15000.0,
        "currency": "NGN",
        "transaction_type": "transfer",
        "channel": "web",
        "merchant_category": "electronics",
        "location_country": "Nigeria",
        "account_age_days": 45,
        "device_change_rate_30d": 0.8,
        "time": "2025-01-22 14:30:00"
    }

    dynamic_analytical_agent = Agent(
        name="dynamic_analytical_agent",
        instructions="""
        You are a dynamic fraud analysis agent that generates custom Python analytical code based on transaction characteristics.
        
        🎯 YOUR MISSION:
        Analyze the given transaction data and dynamically write Python code to perform relevant fraud detection analytics.
        
        📊 AVAILABLE DATA:
        - input_data: The specific transaction being analyzed (dict)
        - transactions_df: Full historical dataset (Pandas DataFrame, read-only)
        
        🔧 CODE EXECUTION WORKFLOW:
        1. Analyze the transaction to identify key risk factors
        2. Write Python code that analyzes those specific risk factors
        3. **IMPORTANT**: Call the execute_python_code_action tool with your generated code
        4. Interpret the execution results and provide fraud assessment
        
        🔧 CODE GENERATION RULES:
        1. Write Python code that analyzes fraud risk factors relevant to the transaction
        2. Use pandas operations on transactions_df for historical comparisons
        3. DO not make use of imports or any disallowed operations
        4. ONLY make use of pandas and numpy for data analysis (pd, np)
        5. Focus on behavioral patterns, statistical anomalies, and risk indicators
        6. NO imports, file operations, or network calls allowed
        7. Always assign final results to a variable named 'result'
        8. Include print statements for debugging/insights
        9. Ensure code is safe and follows sandbox policies

        💡 DYNAMIC ANALYSIS IDEAS:
        - Compare transaction amount vs customer's historical spending patterns
        - Analyze frequency/velocity of transactions for this customer
        - Check for unusual timing patterns (hour, day, frequency)
        - Compare device/location patterns vs historical behavior
        - Calculate statistical outliers (z-scores, percentiles)
        - Identify behavioral anomalies specific to this transaction
        - Generate risk scores based on multiple factors
        
        🚨 CRITICAL: You MUST use the execute_python_code_action_with_healing tool!
        This tool has LLM-POWERED SELF-HEALING that will automatically fix code errors and retry!
        """,
        model="o4-mini",
        tools=[execute_python_code_action_with_healing]
    )
    
    # Dynamic prompt that adapts to transaction characteristics
    analysis_prompt = f"""
    🎯 DYNAMIC FRAUD ANALYSIS REQUEST
    
    Transaction to analyze: {transaction_data}
    
    Please perform the following steps:
    
    1. 🧠 ANALYZE the transaction properties and identify key risk factors to investigate
    
    2. 💻 WRITE Python code that performs relevant fraud analytics for this specific transaction:
       - Customer behavioral analysis (spending patterns, frequency)
       - Statistical anomaly detection (amount outliers, timing patterns)
       - Risk scoring based on historical comparisons
       - Device/location behavioral analysis
       - Transaction velocity and frequency analysis
    
    3. 🚨 **CRITICAL**: Call the execute_python_code_action_with_healing tool to RUN your generated code
       - Pass your Python code as a string to the tool  
       - The tool has SELF-HEALING capabilities - it will automatically fix errors and retry
       - The code will have access to input_data (this transaction) and transactions_df (full dataset)
    
    4. 📊 INTERPRET the execution results and provide fraud risk assessment
    
    **YOU MUST ACTUALLY EXECUTE THE CODE** - don't just describe it!
    Use the execute_python_code_action_with_healing tool - it has automatic error fixing!
    """

    print("🎯 Running Dynamic Analytical Agent...")
    print(f"📊 Analyzing transaction: {transaction_data['transaction_id']}")
    print(f"💰 Amount: {transaction_data['amount']} {transaction_data['currency']}")
    print(f"👤 Customer: {transaction_data['customer_id']}")
    
    result = await Runner.run(dynamic_analytical_agent, analysis_prompt)
    print("\n" + "="*60)
    print("🎯 DYNAMIC ANALYSIS RESULTS:")
    print("="*60)
    print(result.final_output)
    
    return result.final_output


if __name__ == "__main__":
    # Test both agents
    print("🎯 Testing Dynamic Fraud Analysis Agents")
    print("="*60)
    
    # Test 1: Simple agent with basic transaction
    print("\n1️⃣ SIMPLE DYNAMIC AGENT:")
    asyncio.run(simple_agent())



