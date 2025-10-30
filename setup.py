#!/usr/bin/env python3
"""
🚀 Fraud Detection Agent Setup Script

This script sets up the fraud detection agent environment and validates the installation.
"""

import os
import sys
import subprocess
import pkg_resources
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def check_virtual_environment():
    """Check if we're in a virtual environment"""
    print("\n🔧 Checking virtual environment...")
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
        return True
    else:
        print("⚠️  Virtual environment not detected. Consider using a virtual environment.")
        return False

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing requirements...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_env_file():
    """Check if .env file exists"""
    print("\n🔑 Checking environment configuration...")
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("✅ .env file found")
        return True
    elif env_example.exists():
        print("⚠️  .env file not found. Please copy .env.example to .env and configure it.")
        print("   cp .env.example .env")
        return False
    else:
        print("❌ Neither .env nor .env.example found")
        return False

def check_data_files():
    """Check if required data files exist"""
    print("\n📊 Checking data files...")
    data_file = Path("data/transactions.csv")
    
    if data_file.exists():
        print("✅ Transaction data found")
        return True
    else:
        print("⚠️  Transaction data not found. Run generate_transaction_dataset.py to create it.")
        return False

def validate_imports():
    """Validate that key modules can be imported"""
    print("\n🔍 Validating imports...")
    
    try:
        # Test core dependencies
        import pandas
        import numpy
        import lightgbm
        import shap
        print("✅ Core ML libraries available")
        
        # Test OpenAI agents
        from agents import Agent, Runner
        print("✅ OpenAI Agents SDK available")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def run_basic_test():
    """Run a basic functionality test"""
    print("\n🧪 Running basic functionality test...")
    
    try:
        # Add src to path for imports
        sys.path.insert(0, 'src')
        
        from tools.data_loader import load_csv, DATA_PATH
        
        # Test data loading
        if Path(DATA_PATH).exists():
            df = load_csv(DATA_PATH)
            print(f"✅ Data loaded successfully: {df.shape[0]} transactions")
            return True
        else:
            print("⚠️  Data file not found, but imports work")
            return True
            
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("=" * 70)
    print("🛡️  FRAUD DETECTION AGENT SETUP")
    print("=" * 70)
    
    checks = [
        ("Python Version", check_python_version),
        ("Virtual Environment", check_virtual_environment),
        ("Requirements Installation", install_requirements),
        ("Environment Configuration", check_env_file),
        ("Data Files", check_data_files),
        ("Import Validation", validate_imports),
        ("Basic Functionality", run_basic_test)
    ]
    
    results = {}
    for name, check_func in checks:
        results[name] = check_func()
    
    print("\n" + "=" * 70)
    print("📋 SETUP SUMMARY")
    print("=" * 70)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:.<50} {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 Setup completed successfully!")
        print("\n🚀 Next steps:")
        print("   1. Configure your .env file with OPENAI_API_KEY")
        print("   2. Generate transaction data: python generate_transaction_dataset.py")
        print("   3. Run the agent: python src/_agents/fraud_analysis_agent.py")
        print("   4. Run tests: python -m pytest tests/ -v")
    else:
        print("\n⚠️  Some checks failed. Please address the issues above.")
        print("   Refer to README.md for detailed setup instructions.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)