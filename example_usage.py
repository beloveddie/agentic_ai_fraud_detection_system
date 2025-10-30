#!/usr/bin/env python3
"""
🎯 Fraud Detection Agent - Example Usage

This script demonstrates how to use the fraud detection agent 
for analyzing transactions and detecting fraudulent patterns.

Run this after setting up your environment and API key.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from dotenv import load_dotenv
    load_dotenv()
    
    from agents import Runner
    from _agents.fraud_analysis_agent import agent
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure to run: pip install -r requirements.txt")
    sys.exit(1)

# Sample transactions for testing
SAMPLE_TRANSACTIONS = [
    {
        "transaction_id": "TXN_001",
        "customer_id": "CUST_12345",
        "amount": 2500.00,
        "currency": "NGN",
        "transaction_type": "online_purchase",
        "channel": "web",
        "merchant_category": "electronics",
        "device_id": "DEV_456",
        "ip_address": "192.168.1.100",
        "is_weekend": 1,
        "hour": 14
    },
    {
        "transaction_id": "TXN_002", 
        "customer_id": "CUST_67890",
        "amount": 50000.0,
        "currency": "NGN",
        "transaction_type": "transfer",
        "channel": "mobile",
        "merchant_category": "gambling",
        "device_id": "DEV_999",
        "ip_address": "10.0.0.1",
        "is_weekend": 0,
        "hour": 3
    },
    {
        "transaction_id": "TXN_003",
        "customer_id": "CUST_11111",
        "amount": 150.0,
        "currency": "NGN", 
        "transaction_type": "purchase",
        "channel": "pos",
        "merchant_category": "grocery",
        "device_id": "DEV_123",
        "ip_address": "172.16.0.1",
        "is_weekend": 1,
        "hour": 10
    }
]

async def analyze_transaction(transaction_data, detailed=True):
    """
    Analyze a single transaction for fraud indicators
    
    Args:
        transaction_data: Dictionary with transaction details
        detailed: Whether to request detailed analysis
    
    Returns:
        Analysis result from the agent
    """
    print(f"\n🔍 Analyzing Transaction: {transaction_data['transaction_id']}")
    print(f"   Customer: {transaction_data['customer_id']}")
    print(f"   Amount: {transaction_data['amount']} {transaction_data['currency']}")
    print(f"   Type: {transaction_data['transaction_type']}")
    print("-" * 60)
    
    if detailed:
        prompt = f"""
        Please analyze this transaction for fraud indicators using all available tools:
        
        Transaction Details: {transaction_data}
        
        I need:
        1. ML-based fraud risk assessment with SHAP explanations
        2. Behavioral analysis compared to customer history  
        3. Risk factor identification and scoring
        4. Detailed recommendations for handling this transaction
        5. Confidence levels for all assessments
        
        Please provide a comprehensive analysis with clear explanations.
        """
    else:
        prompt = f"Analyze this transaction for fraud: {transaction_data}"
    
    try:
        result = await Runner.run(agent, prompt)
        return result.final_output
    except Exception as e:
        return f"❌ Analysis failed: {str(e)}"

async def demonstrate_capabilities():
    """Demonstrate the agent's capabilities"""
    print("🛡️ Fraud Detection Agent - Capability Demo")
    print("=" * 70)
    
    # Show agent capabilities
    print("\n1. 🤖 Agent Capabilities Overview")
    print("-" * 40)
    
    capabilities_result = await Runner.run(
        agent, 
        "What fraud detection capabilities do you have? Please be specific about your tools and analysis methods."
    )
    print(capabilities_result.final_output)
    
    return True

async def analyze_sample_transactions():
    """Analyze the sample transactions"""
    print("\n\n2. 📊 Sample Transaction Analysis")
    print("=" * 70)
    
    for i, transaction in enumerate(SAMPLE_TRANSACTIONS, 1):
        print(f"\n--- Transaction {i} ---")
        
        # Analyze each transaction
        analysis = await analyze_transaction(transaction, detailed=(i == 1))
        print(analysis)
        
        if i < len(SAMPLE_TRANSACTIONS):
            print(f"\n{'='*60}")

async def demonstrate_custom_analytics():
    """Demonstrate custom analytics code execution"""
    print("\n\n3. 🧮 Custom Analytics Demo")
    print("=" * 70)
    
    custom_analysis_prompt = f"""
    Using the execute_python_code_action tool, perform advanced analytics on this transaction:
    
    {SAMPLE_TRANSACTIONS[1]}
    
    Please calculate:
    1. Risk score based on amount and time patterns
    2. Behavioral anomaly indicators  
    3. Statistical comparison with dataset patterns
    4. Custom fraud probability metrics
    
    Generate Python code to analyze these patterns and provide insights.
    """
    
    try:
        result = await Runner.run(agent, custom_analysis_prompt)
        print("🧮 Custom Analytics Result:")
        print("-" * 40)
        print(result.final_output)
    except Exception as e:
        print(f"❌ Custom analytics failed: {e}")

async def main():
    """Main demonstration function"""
    print("🚀 Starting Fraud Detection Agent Demo")
    
    # Check if OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not set. Please configure your .env file.")
        print("   Copy .env.example to .env and add your API key.")
        return False
    
    try:
        # Run demonstrations
        await demonstrate_capabilities()
        await analyze_sample_transactions()
        await demonstrate_custom_analytics()
        
        print("\n\n✅ Demo completed successfully!")
        print("\n🎯 Next Steps:")
        print("   • Modify SAMPLE_TRANSACTIONS to test your own data")
        print("   • Explore different analysis prompts")
        print("   • Check out the documentation in docs/")
        print("   • Run the test suite: python -m pytest tests/ -v")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("   Check your API key and internet connection")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)