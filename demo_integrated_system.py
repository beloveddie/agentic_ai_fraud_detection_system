#!/usr/bin/env python3
"""
Comprehensive example demonstrating the integrated fraud analysis agent
with behavioral pattern analysis and code-as-action capabilities
"""

import sys
import os
import asyncio
import json
from datetime import datetime

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

async def demonstrate_integrated_fraud_analysis():
    """Demonstrate the integrated fraud analysis system"""
    
    print("🔍 Integrated Fraud Analysis Agent Demo")
    print("=" * 60)
    
    try:
        from agents.fraud_analysis_agent import agent
        from agents import Runner
        
        # Sample transaction for analysis
        sample_transaction = {
            "transaction_id": "TXN_DEMO_001",
            "customer_id": "CUST_DEMO_001", 
            "amount": 25000.0,
            "currency": "NGN",
            "transaction_type": "transfer",
            "channel": "web",
            "merchant_category": "luxury",
            "device_id": "DEV_NEW_001",
            "ip_address": "192.168.1.200",
            "location_country": "Nigeria",
            "location_city": "Abuja",
            "time": "2025-01-22 02:30:00",  # Unusual time
            "account_age_days": 15,  # New account
            "num_prev_transactions_24h": 12,  # High frequency
            "avg_transaction_amount_7d": 2000.0,
            "device_change_rate_30d": 0.9,  # High device change
            "is_foreign": 0,
            "is_blacklisted_merchant": 0
        }
        
        print("📋 Transaction Details:")
        print(f"   ID: {sample_transaction['transaction_id']}")
        print(f"   Customer: {sample_transaction['customer_id']}")
        print(f"   Amount: {sample_transaction['amount']} {sample_transaction['currency']}")
        print(f"   Type: {sample_transaction['transaction_type']}")
        print(f"   Channel: {sample_transaction['channel']}")
        print(f"   Time: {sample_transaction['time']}")
        print(f"   Account Age: {sample_transaction['account_age_days']} days")
        print(f"   Device Change Rate: {sample_transaction['device_change_rate_30d']}")
        
        # Comprehensive analysis prompt
        analysis_prompt = f"""
        Please perform a comprehensive fraud analysis for this transaction using all available tools:
        
        Transaction Details:
        - Transaction ID: {sample_transaction['transaction_id']}
        - Customer ID: {sample_transaction['customer_id']}
        - Amount: {sample_transaction['amount']} {sample_transaction['currency']}
        - Type: {sample_transaction['transaction_type']}
        - Channel: {sample_transaction['channel']}
        - Merchant Category: {sample_transaction['merchant_category']}
        - Device ID: {sample_transaction['device_id']}
        - Location: {sample_transaction['location_city']}, {sample_transaction['location_country']}
        - Time: {sample_transaction['time']}
        - Account Age: {sample_transaction['account_age_days']} days
        - Previous Transactions (24h): {sample_transaction['num_prev_transactions_24h']}
        - Device Change Rate (30d): {sample_transaction['device_change_rate_30d']}
        
        Please perform the following analysis:
        
        1. **Customer History Retrieval**: Get the customer's transaction history for behavioral analysis
        
        2. **ML-Based Fraud Detection**: Use the fraud detector tool to get:
           - Fraud probability score
           - SHAP explanations
           - Risk level assessment
           - Confidence scores
        
        3. **Behavioral Pattern Analysis**: Analyze customer behavior patterns including:
           - Spending patterns and acceleration
           - Temporal behavior (timing patterns)
           - Geographic patterns and location consistency
           - Device and channel usage patterns
           - Transaction frequency patterns
           - Behavioral risk indicators
        
        4. **Comprehensive Assessment**: Combine both analyses to provide:
           - Overall fraud risk assessment
           - Key risk factors from both ML and behavioral perspectives
           - Confidence levels and data sufficiency
           - Specific recommendations for risk mitigation
        
        5. **Actionable Insights**: Provide clear, actionable recommendations based on the combined analysis
        
        Please structure your response to clearly show:
        - ML model predictions and explanations
        - Behavioral pattern analysis results
        - Combined risk assessment
        - Specific recommendations
        - Confidence levels and data quality assessment
        """
        
        print(f"\n🤖 Running comprehensive fraud analysis...")
        print("   This may take a moment as the agent uses multiple tools...")
        
        # Run the analysis
        result = await Runner.run(agent, analysis_prompt)
        
        print(f"\n📊 Analysis Results:")
        print("=" * 60)
        print(result.final_output)
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

async def demonstrate_behavioral_analysis_only():
    """Demonstrate behavioral analysis in isolation"""
    
    print(f"\n🧠 Behavioral Analysis Demo (Standalone)")
    print("=" * 60)
    
    try:
        from tools.behavioral_analyzer import BehavioralAnalyzer
        
        analyzer = BehavioralAnalyzer()
        
        # Sample transaction with suspicious patterns
        suspicious_transaction = {
            "transaction_id": "TXN_SUSPICIOUS_001",
            "customer_id": "CUST_SUSPICIOUS_001",
            "amount": 50000.0,  # High amount
            "currency": "NGN",
            "transaction_type": "transfer",
            "channel": "web",
            "merchant_category": "luxury",
            "device_id": "DEV_UNKNOWN_001",  # New device
            "location_country": "USA",  # Foreign location
            "location_city": "New York",
            "time": "2025-01-22 03:00:00",  # Unusual time
            "account_age_days": 5  # Very new account
        }
        
        # Sample customer history showing normal patterns
        normal_history = [
            {
                "transaction_id": "TXN_NORMAL_001",
                "customer_id": "CUST_SUSPICIOUS_001",
                "amount": 2000.0,
                "currency": "NGN",
                "transaction_type": "purchase",
                "channel": "mobile",
                "merchant_category": "groceries",
                "device_id": "DEV_REGULAR_001",
                "location_country": "Nigeria",
                "location_city": "Lagos",
                "time": "2025-01-20 10:00:00"
            },
            {
                "transaction_id": "TXN_NORMAL_002",
                "customer_id": "CUST_SUSPICIOUS_001",
                "amount": 1500.0,
                "currency": "NGN",
                "transaction_type": "purchase",
                "channel": "mobile",
                "merchant_category": "groceries",
                "device_id": "DEV_REGULAR_001",
                "location_country": "Nigeria",
                "location_city": "Lagos",
                "time": "2025-01-19 14:30:00"
            },
            {
                "transaction_id": "TXN_NORMAL_003",
                "customer_id": "CUST_SUSPICIOUS_001",
                "amount": 3000.0,
                "currency": "NGN",
                "transaction_type": "transfer",
                "channel": "mobile",
                "merchant_category": "utilities",
                "device_id": "DEV_REGULAR_001",
                "location_country": "Nigeria",
                "location_city": "Lagos",
                "time": "2025-01-18 16:00:00"
            }
        ]
        
        print("📋 Analyzing suspicious transaction against normal customer history:")
        print(f"   Current Transaction: {suspicious_transaction['amount']} {suspicious_transaction['currency']} at {suspicious_transaction['time']}")
        print(f"   Customer History: {len(normal_history)} previous transactions")
        
        # Run behavioral analysis
        result = analyzer.analyze_behavioral_patterns(suspicious_transaction, normal_history)
        
        print(f"\n📊 Behavioral Analysis Results:")
        print(f"   Analysis Quality: {result['analysis_quality']}")
        print(f"   Data Sufficiency: {result['data_sufficiency_score']:.3f}")
        print(f"   Analysis Confidence: {result['analysis_confidence']:.3f}")
        
        print(f"\n💰 Spending Patterns:")
        spending = result['spending_patterns']
        print(f"   Avg Amount (7d): {spending['avg_amount_7d']}")
        print(f"   Avg Amount (30d): {spending['avg_amount_30d']}")
        print(f"   Spending Acceleration: {spending['spending_acceleration_7d']}")
        print(f"   Consistency Score: {spending['spending_consistency_score']}")
        print(f"   Unusual Spike: {spending['unusual_spending_spike']}")
        
        print(f"\n⏰ Temporal Patterns:")
        temporal = result['temporal_patterns']
        print(f"   Preferred Hours: {temporal['preferred_hours']}")
        print(f"   Time Consistency: {temporal['time_consistency_score']}")
        print(f"   Unusual Time: {temporal['unusual_time_transaction']}")
        print(f"   Night Ratio: {temporal['night_transaction_ratio']}")
        
        print(f"\n🌍 Geographic Patterns:")
        geo = result['geographic_patterns']
        print(f"   Primary Locations: {geo['primary_locations']}")
        print(f"   Location Consistency: {geo['location_consistency_score']}")
        print(f"   Unusual Location: {geo['unusual_location']}")
        print(f"   International Ratio: {geo['international_transaction_ratio']}")
        
        print(f"\n📱 Device/Channel Patterns:")
        device = result['device_channel_patterns']
        print(f"   Preferred Devices: {device['preferred_devices']}")
        print(f"   Device Consistency: {device['device_consistency_score']}")
        print(f"   New Device: {device['new_device_detected']}")
        print(f"   Device Change Rate: {device['device_change_rate']}")
        
        print(f"\n🎯 Risk Indicators:")
        risk = result['risk_indicators']
        print(f"   Behavioral Risk Score: {risk['behavioral_risk_score']}")
        print(f"   Risk Level: {risk['risk_level']}")
        print(f"   Anomaly Count: {risk['anomaly_count']}")
        print(f"   Top Risk Factors: {risk['top_risk_factors']}")
        
        print(f"\n💡 Behavioral Insights:")
        insights = result['behavioral_insights']
        print(f"   Customer Profile: {insights['customer_profile']}")
        print(f"   Behavioral Stability: {insights['behavioral_stability']}")
        print(f"   Fraud Likelihood: {insights['fraud_likelihood']}")
        print(f"   Recommendations: {insights['recommended_actions']}")
        print(f"   Summary: {insights['behavioral_summary']}")
        
        print(f"\n🔧 Raw Patterns (for code-as-action):")
        print(f"   Available patterns: {list(result['raw_patterns'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Behavioral analysis demo failed: {e}")
        return False

async def main():
    """Run all demonstrations"""
    
    print("🚀 Comprehensive Fraud Analysis Agent Demonstration")
    print("=" * 80)
    print("This demo showcases the integrated fraud analysis system with:")
    print("- ML-based fraud detection with SHAP explanations")
    print("- Behavioral pattern analysis")
    print("- Combined risk assessment")
    print("- Code-as-action capabilities")
    print("=" * 80)
    
    demos = [
        ("Integrated Fraud Analysis", demonstrate_integrated_fraud_analysis),
        ("Behavioral Analysis Standalone", demonstrate_behavioral_analysis_only)
    ]
    
    passed = 0
    total = len(demos)
    
    for demo_name, demo_func in demos:
        print(f"\n🎬 Running {demo_name} Demo...")
        try:
            success = await demo_func()
            if success:
                passed += 1
                print(f"✅ {demo_name} Demo COMPLETED")
            else:
                print(f"❌ {demo_name} Demo FAILED")
        except Exception as e:
            print(f"❌ {demo_name} Demo FAILED with exception: {e}")
    
    print(f"\n📊 Demo Results: {passed}/{total} demos completed successfully")
    
    if passed == total:
        print("🎉 All demonstrations completed successfully!")
        print("\n💡 Key Features Demonstrated:")
        print("   ✅ ML-based fraud detection with SHAP explanations")
        print("   ✅ Comprehensive behavioral pattern analysis")
        print("   ✅ Combined risk assessment")
        print("   ✅ Rich structured data for code-as-action")
        print("   ✅ Actionable recommendations")
        print("   ✅ Confidence scoring and data sufficiency")
    else:
        print("⚠️  Some demonstrations failed. Please check the errors above.")

if __name__ == "__main__":
    asyncio.run(main())
