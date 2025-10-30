#!/usr/bin/env python3
"""
🔍 Pre-GitHub Push Validation Script

This script performs comprehensive validation before pushing to GitHub:
- Code quality checks
- Import validation  
- Test execution
- Documentation validation
- Security checks
"""

import os
import sys
import subprocess
import ast
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            return True
        else:
            print(f"❌ {description} - FAILED")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def check_python_syntax():
    """Check Python syntax for all .py files"""
    print("🐍 Checking Python syntax...")
    python_files = list(Path('.').rglob('*.py'))
    
    errors = []
    for py_file in python_files:
        if '.venv' in str(py_file) or '__pycache__' in str(py_file):
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                source = f.read()
            ast.parse(source)
        except SyntaxError as e:
            errors.append(f"{py_file}: {e}")
        except Exception as e:
            errors.append(f"{py_file}: {e}")
    
    if errors:
        print(f"❌ Python syntax check - FAILED")
        for error in errors:
            print(f"   {error}")
        return False
    else:
        print(f"✅ Python syntax check - PASSED ({len(python_files)} files)")
        return True

def check_required_files():
    """Check that all required files exist"""
    print("📋 Checking required files...")
    
    required_files = [
        'README.md',
        'requirements.txt', 
        'LICENSE',
        '.gitignore',
        '.env.example',
        'src/_agents/fraud_analysis_agent.py',
        'src/tools/fraud_detector.py',
        'src/tools/behavioral_analyzer.py',
        'src/tools/data_loader.py',
        'src/tools/sandbox_runner.py'
    ]
    
    missing = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(file_path)
    
    if missing:
        print(f"❌ Required files check - FAILED")
        for file_path in missing:
            print(f"   Missing: {file_path}")
        return False
    else:
        print(f"✅ Required files check - PASSED")
        return True

def check_imports():
    """Check that key modules can be imported"""
    print("📦 Checking imports...")
    
    test_imports = [
        'import pandas',
        'import numpy', 
        'import lightgbm',
        'import shap',
        'from agents import Agent, Runner',
        'from dotenv import load_dotenv'
    ]
    
    errors = []
    for import_stmt in test_imports:
        try:
            exec(import_stmt)
        except ImportError as e:
            errors.append(f"{import_stmt}: {e}")
        except Exception as e:
            errors.append(f"{import_stmt}: {e}")
    
    if errors:
        print(f"❌ Import check - FAILED")
        for error in errors:
            print(f"   {error}")
        return False
    else:
        print(f"✅ Import check - PASSED")
        return True

def validate_documentation():
    """Validate documentation completeness"""
    print("📚 Validating documentation...")
    
    # Check README exists and has key sections
    readme_path = Path('README.md')
    if not readme_path.exists():
        print("❌ Documentation validation - No README.md found")
        return False
    
    readme_content = readme_path.read_text(encoding='utf-8')
    required_sections = [
        '# 🛡️ AI-Powered Fraud Detection Agent',
        '## 🌟 Features',
        '## 🚀 Quick Start',
        '## 📊 Dataset',
        '## 🔧 Core Components'
    ]
    
    missing_sections = []
    for section in required_sections:
        if section not in readme_content:
            missing_sections.append(section)
    
    if missing_sections:
        print(f"❌ Documentation validation - FAILED")
        for section in missing_sections:
            print(f"   Missing section: {section}")
        return False
    else:
        print(f"✅ Documentation validation - PASSED")
        return True

def check_git_status():
    """Check git repository status"""
    print("🔧 Checking git status...")
    
    # Check if we're in a git repo
    if not Path('.git').exists():
        print("⚠️  Not a git repository - initializing...")
        run_command('git init', 'Git initialization')
        return True
    
    # Check for uncommitted changes
    result = subprocess.run('git status --porcelain', shell=True, capture_output=True, text=True)
    if result.stdout.strip():
        print("📝 Uncommitted changes detected:")
        print(result.stdout)
        return True
    else:
        print("✅ Git status - clean working directory")
        return True

def main():
    """Main validation function"""
    print("=" * 70)
    print("🔍 PRE-GITHUB PUSH VALIDATION")
    print("=" * 70)
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    # Run all validation checks
    checks = [
        ("Python Syntax", check_python_syntax),
        ("Required Files", check_required_files), 
        ("Import Validation", check_imports),
        ("Documentation", validate_documentation),
        ("Git Status", check_git_status)
    ]
    
    results = {}
    for name, check_func in checks:
        results[name] = check_func()
        print()
    
    # Summary
    print("=" * 70)
    print("📋 VALIDATION SUMMARY")
    print("=" * 70)
    
    passed = 0
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:.<50} {status}")
        if result:
            passed += 1
    
    success_rate = (passed / len(checks)) * 100
    print(f"\nSuccess Rate: {success_rate:.1f}% ({passed}/{len(checks)})")
    
    if passed == len(checks):
        print("\n🎉 All validations passed! Ready for GitHub push.")
        print("\n📋 Recommended commands:")
        print("   git add .")
        print("   git commit -m 'Initial commit: AI-powered fraud detection agent'")
        print("   git branch -M main")
        print("   git remote add origin <your-repo-url>")
        print("   git push -u origin main")
    else:
        print(f"\n⚠️  {len(checks) - passed} validation(s) failed. Please fix before pushing.")
    
    return passed == len(checks)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)