import sys
import io
import traceback
import contextlib
import multiprocessing
import signal
import builtins
import types
import re

from agents import function_tool
from .data_loader import load_csv, DATA_PATH

ALLOWED_MODULES = {
    "pd": __import__("pandas"),
    "np": __import__("numpy"),
}

ALLOWED_BUILTINS = {
    k: getattr(builtins, k)
    for k in [
        "abs", "all", "any", "bool", "bytes", "callable", "chr", "complex",
        "dict", "divmod", "enumerate", "filter", "float", "format", "frozenset",
        "hash", "hex", "int", "isinstance", "issubclass", "iter", "len", "list", "map",
        "max", "min", "next", "object", "oct", "ord", "pow", "range", "repr", "reversed",
        "round", "set", "slice", "str", "sum", "tuple", "zip", "print"
    ]
}

SAFE_GLOBALS = {
    "__builtins__": ALLOWED_BUILTINS,
    **ALLOWED_MODULES,
}

def _run_code_safely(code, user_globals, user_input, data_loader, q):
    # Assemble full globals
    sand_globals = dict(SAFE_GLOBALS)
    if user_globals:
        sand_globals.update(user_globals)
    if user_input:
        sand_globals["input_data"] = user_input
    if data_loader:
        sand_globals["data_loader"] = data_loader
    result = {"stdout": "", "stderr": "", "result": None, "error": None}
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            with contextlib.redirect_stderr(stderr):
                compiled = compile(code, "<sandbox>", "exec")
                exec(compiled, sand_globals)
                # Optionally, retrieve a conventionally named 'result' variable
                result_val = sand_globals.get("result", None)
                result["result"] = result_val
    except Exception:
        result["error"] = traceback.format_exc()
    result["stdout"] = stdout.getvalue()
    result["stderr"] = stderr.getvalue()
    q.put(result)

def run_in_sandbox(code, globals_dict=None, input_data=None, data_loader=None, timeout=100):
    """
    Run Python code string in a secure sandbox environment.
    Args:
        code: Python code as a string.
        globals_dict: Merge into sandbox global namespace.
        input_data: Input data available as 'input_data' in sandbox.
        data_loader: Data loader function, available as 'data_loader' in sandbox.
        timeout: Max execution time in seconds.
    Returns:
        dict: {"stdout", "stderr", "result", "error"}
    """
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_run_code_safely, args=(code, globals_dict, input_data, data_loader, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        return {"stdout": "", "stderr": "", "result": None, "error": f"Timeout: Code execution exceeded {timeout} seconds."}
    if not q.empty():
        return q.get()
    else:
        return {"stdout": "", "stderr": "", "result": None, "error": "No result returned from sandbox."}

def preprocess_llm_code(code: str) -> str:
    """
    Preprocess LLM-generated code for safety, normalization, and compatibility.
    Returns: sanitized code string
    Raises: ValueError if forbidden pattern detected
    """
    forbidden_patterns = [
        r"import\s+os", r"import\s+sys", r"import\s+subprocess", r"import\s+socket",
        r"open\(", r"eval\(", r"exec\(", r"__import__", r"importlib",
        r"from\s+os", r"from\s+sys", r"from\s+subprocess", r"from\s+socket",
        r"\.system\(", r"\.popen\(", r"\.remove\(", r"\.unlink\(", r"input\("
    ]
    for pat in forbidden_patterns:
        try:
            if re.search(pat, code):
                raise ValueError(f"Forbidden code pattern detected: {pat}")
        except re.error as rex:
            raise ValueError(f"Regex pattern error ('{pat}'): {rex}")
    # Remove 'import pandas' or 'import numpy' if found
    code = re.sub(r'(^|\n)\s*import (pandas|numpy)( as [a-zA-Z0-9_]+)?\s*', '\n', code)
    code = re.sub(r'(^|\n)\s*from (pandas|numpy).*', '\n', code)
    # Ensure 'result =' for last variable (wrap last line if not assignment)
    lines = [l for l in code.strip().split('\n') if l.strip()]
    
    # Check if code already has result assignment in the body
    has_result_assignment = any('result =' in line for line in lines)
    
    if lines and not has_result_assignment:
        # Only wrap if there's no result assignment anywhere and last line isn't structural
        last_line = lines[-1].strip()
        if (not last_line.startswith("result ") and 
            "result" not in last_line and 
            last_line not in ['', '}', ')', ']', 'except:', 'finally:']):
            lines[-1] = f"result = {lines[-1].strip()}"
    
    code = '\n'.join(lines)
    return code

def agent_run_code_action(code_from_llm: str, input_data: dict, timeout: int = 10) -> dict:
    """
    Preprocess and sandbox-run LLM code.
    The code will have access to:
       - 'input_data': the current transaction as a dict
       - 'transactions_df': the full dataset (ALL transactions) as a Pandas DataFrame (read only, do NOT modify or save)
    Args:
        code_from_llm: Python code string
        input_data: transaction dict
        timeout: Max execution time
    Returns: dict with stdout, stderr, result, error, preprocessed_code
    """
    print(f"🔄 Starting code preprocessing...")
    try:
        safe_code = preprocess_llm_code(code_from_llm)
        print(f"✅ Code preprocessing successful!")
        print(f"📝 Preprocessed code length: {len(safe_code)} characters")
    except Exception as e:
        print(f"❌ Code preprocessing failed: {str(e)}")
        return {
            "stdout": "", "stderr": "", "result": None,
            "error": f"Code preprocessing error: {str(e)}", "preprocessed_code": None
        }
    
    print(f"📊 Loading transactions dataset...")
    try:
        transactions_df = load_csv(DATA_PATH)
        print(f"✅ Dataset loaded successfully! Shape: {transactions_df.shape}")
        # Optionally: lock down as read-only if you want (transactions_df.flags.writeable = False)
        extra_globals = {"transactions_df": transactions_df}
    except Exception as e:
        print(f"❌ Dataset loading failed: {str(e)}")
        return {
            "stdout": "", "stderr": "", "result": None,
            "error": f"Failed to load transactions.csv: {str(e)}", "preprocessed_code": safe_code
        }
    
    print(f"🏃 Executing code in sandbox (timeout: {timeout}s)...")
    result = run_in_sandbox(safe_code, globals_dict=extra_globals, input_data=input_data, timeout=timeout)
    result["preprocessed_code"] = safe_code
    
    print(f"🎯 Sandbox execution completed!")
    return result

@function_tool(strict_mode=False)
async def execute_python_code_action(
    code: str,
    input_data: dict,
    timeout: int = 10
) -> dict:
    """
    Agent-safe Python code executor for OpenAI Agent SDK/function-calling interface.
    The code may use both:
      - input_data (dict): The current transaction
      - transactions_df (Pandas DataFrame): All transactions (read-only; do NOT modify or save)
    Args:
        code: Python code string. ABSOLUTELY NO imports/os/sys/file/network ops allowed; must assign to 'result' and use only provided inputs.
        input_data: See above.
        timeout: Seconds allowed to run before forced termination.
    Returns:
        Dict: stdout, stderr, result, error, preprocessed_code.
    """
    # 🎯 LOG: Tool execution started
    print(f"\n{'='*60}")
    print(f"🔧 EXECUTE_PYTHON_CODE_ACTION CALLED!")
    print(f"{'='*60}")
    print(f"📊 Input Data: {input_data.get('transaction_id', 'Unknown')} - Amount: {input_data.get('amount', 'Unknown')}")
    print(f"⏱️  Timeout: {timeout} seconds")
    print(f"💻 Code Length: {len(code)} characters")
    print(f"📝 Code Preview (first 200 chars):")
    print(f"   {code[:200]}{'...' if len(code) > 200 else ''}")
    print(f"{'='*60}")
    
    result = agent_run_code_action(code, input_data, timeout)
    
    # 🎯 LOG: Tool execution completed
    print(f"\n🎯 EXECUTION COMPLETED!")
    print(f"✅ Status: {'SUCCESS' if not result.get('error') else 'ERROR'}")
    
    if result.get('error'):
        print(f"💥 FINAL ERROR DETAILS:")
        print(f"{'='*50}")
        print(f"❌ Error: {result['error']}")
        if result.get('stderr'):
            print(f"📢 Stderr: {result['stderr']}")
        print(f"{'='*50}")
    
    if result.get('result'):
        print(f"📊 SUCCESS DETAILS:")
        print(f"{'='*50}")
        print(f"📊 Result Type: {type(result['result'])}")
        print(f"📊 Result Preview: {str(result['result'])[:300]}...")
        print(f"{'='*50}")
        
    if result.get('stdout'):
        print(f"� Execution Output:")
        print(f"{'='*50}")
        print(result['stdout'][:500] + ("..." if len(result.get('stdout', '')) > 500 else ""))
        print(f"{'='*50}")
        
    print(f"📝 Total Output Length: {len(result.get('stdout', ''))}")
    print(f"{'='*60}\n")
    
    return result


async def analyze_error_with_llm_agent(error_message: str, original_code: str, input_data: dict, attempt_number: int) -> dict:
    """
    🧠 LLM-POWERED ERROR ANALYSIS & SELF-HEALING AGENT
    
    Uses an AI agent specialized in code debugging and error fixing
    """
    from agents import Agent, Runner
    
    print(f"\n🧠 LLM SELF-HEALING AGENT ACTIVATED (Attempt {attempt_number})")
    print(f"{'='*60}")
    print(f"🔍 Error Analysis Starting...")
    print(f"📋 Error Summary:")
    print(f"   Message: {error_message[:200]}{'...' if len(error_message) > 200 else ''}")
    print(f"   Code Length: {len(original_code)} chars")
    print(f"   Transaction ID: {input_data.get('transaction_id', 'Unknown')}")
    print(f"{'='*60}")
    
    # Create specialized debugging agent
    code_healer_agent = Agent(
        name="code_healer_specialist", 
        model="o3-mini",  # Use o3-mini for fast, efficient debugging
        instructions="""
        🧠 You are a SPECIALIZED CODE DEBUGGING & HEALING AGENT.
        
        Your mission: Analyze Python code execution errors and generate WORKING fixes.
        
        � DEBUGGING EXPERTISE:
        - Pandas DataFrame operations and column handling
        - Data type conversions and safety checks
        - Error handling and defensive programming
        - Statistical analysis and fraud detection patterns
        
        📊 AVAILABLE CONTEXT:
        - input_data: Current transaction being analyzed (dict)
        - transactions_df: Full historical dataset (Pandas DataFrame)
        - Only pandas (pd) and numpy (np) are available - NO imports allowed
        
        🎯 HEALING APPROACH:
        1. ANALYZE the error message and identify root cause
        2. EXAMINE the failing code to understand intent
        3. GENERATE a corrected version that handles the error
        4. ENSURE the fix maintains the original analytical purpose
        5. ADD defensive programming patterns (null checks, type safety)
        
        🛠️ CODE REQUIREMENTS:
        - Must assign final result to 'result' variable
        - Use safe pandas operations (.get(), .fillna(), error='coerce')
        - Include meaningful print statements for debugging
        - Handle edge cases (empty DataFrames, missing columns, zero divisions)
        
        Be smart, analytical, and generate ROBUST code that won't fail again!
        """
    )
    
    # Create detailed healing prompt
    healing_prompt = f"""
    🚨 CODE EXECUTION ERROR - NEED HEALING!
    
    TRANSACTION CONTEXT:
    {input_data}
    
    FAILING CODE:
    ```python
    {original_code}
    ```
    
    ERROR MESSAGE:
    {error_message}
    
    ATTEMPT NUMBER: {attempt_number}
    
    🧠 ANALYSIS NEEDED:
    1. What went wrong in this code?
    2. What was the original analytical intent?
    3. How can we fix it while maintaining the purpose?
    
    🛠️ GENERATE CORRECTED CODE that:
    - Fixes the specific error
    - Maintains the original analytical purpose  
    - Adds defensive programming patterns
    - Handles edge cases gracefully
    - Provides meaningful fraud analysis results
    
    Return your analysis and the corrected Python code.
    Focus on making it ROBUST and ERROR-PROOF!
    """
    
    try:
        print(f"🔄 Running LLM healing analysis...")
        
        healing_result = await Runner.run(
            code_healer_agent, 
            healing_prompt
        )
        
        # Fix: Access the response correctly from RunResult
        if hasattr(healing_result, 'final_output'):
            healing_response = healing_result.final_output
        elif hasattr(healing_result, 'content'):
            healing_response = healing_result.content
        elif hasattr(healing_result, 'text'):
            healing_response = healing_result.text
        else:
            # Try to get string representation
            healing_response = str(healing_result) if healing_result else "No healing response"
        
        print(f"✅ LLM analysis completed!")
        print(f"📝 Healing response length: {len(healing_response)} characters")
        
        # Show LLM's error analysis (first part of response)
        analysis_preview = healing_response[:500] + "..." if len(healing_response) > 500 else healing_response
        print(f"\n🧠 LLM ERROR ANALYSIS:")
        print(f"{'='*50}")
        print(analysis_preview)
        print(f"{'='*50}")
        
        # Extract corrected code from LLM response
        corrected_code = extract_code_from_llm_response(healing_response)
        
        return {
            "healing_method": "llm_agent",
            "llm_analysis": healing_response,
            "corrected_code": corrected_code,
            "confidence": 0.9,  # High confidence in LLM fixes
            "attempt_number": attempt_number,
            "agent_model": "o3-mini"
        }
        
    except Exception as e:
        print(f"❌ LLM healing failed: {str(e)}")
        
        return {
            "healing_method": "llm_failed",
            "corrected_code": None,
            "confidence": 0.0,
            "attempt_number": attempt_number,
            "error": str(e)
        }


def extract_code_from_llm_response(llm_response: str) -> str:
    """
    🔍 Extract Python code from LLM response
    """
    import re
    
    # Look for code blocks
    code_patterns = [
        r'```python\n(.*?)\n```',
        r'```\n(.*?)\n```', 
        r'CORRECTED CODE:\s*\n(.*?)(?=\n\n|\Z)',
        r'Fixed code:\s*\n(.*?)(?=\n\n|\Z)'
    ]
    
    for pattern in code_patterns:
        matches = re.findall(pattern, llm_response, re.DOTALL | re.IGNORECASE)
        if matches:
            code = matches[0].strip()
            if len(code) > 20:  # Reasonable code length
                print(f"✅ Extracted code from LLM response ({len(code)} chars)")
                return code
    
    # Fallback: extract lines that look like Python code
    lines = llm_response.split('\n')
    code_lines = []
    in_code_block = False
    
    for line in lines:
        if any(keyword in line for keyword in ['=', 'if ', 'for ', 'def ', 'result', 'print(', 'transactions_df']):
            code_lines.append(line)
            in_code_block = True
        elif in_code_block and line.strip() == '':
            code_lines.append(line)
        elif in_code_block and not line.strip().startswith('#'):
            break
    
    if code_lines:
        extracted_code = '\n'.join(code_lines).strip()
        print(f"📝 Extracted code via pattern matching ({len(extracted_code)} chars)")
        return extracted_code
    
    # Ultimate fallback - return simple safe code
    print(f"⚠️ Could not extract code - using minimal fallback")
    return """
# Basic safe analysis when code extraction fails
try:
    result = {
        'transaction_amount': input_data.get('amount', 0),
        'customer_id': input_data.get('customer_id', 'unknown'),
        'analysis_method': 'extraction_fallback',
        'status': 'basic_analysis'
    }
except Exception as e:
    result = {'error': f'Basic analysis failed: {str(e)}'}
"""











@function_tool(strict_mode=False)
async def execute_python_code_action_with_healing(
    code: str,
    input_data: dict,
    timeout: int = 10,
    max_retries: int = 2
) -> dict:
    """
    🧠 SELF-HEALING Python Code Executor
    
    Automatically detects errors and attempts to fix them with agentic feedback loop
    """
    print(f"\n🧠 SELF-HEALING CODE EXECUTOR STARTED")
    print(f"{'='*60}")
    print(f"🔄 Max Retries: {max_retries}")
    
    execution_history = []
    current_code = code
    
    for attempt in range(max_retries + 1):
        print(f"\n🚀 ATTEMPT {attempt + 1}/{max_retries + 1}")
        print(f"{'='*40}")
        
        # Execute the current code
        result = agent_run_code_action(current_code, input_data, timeout)
        
        execution_history.append({
            "attempt": attempt + 1,
            "code_length": len(current_code),
            "success": not result.get('error'),
            "error": result.get('error'),
            "result_preview": str(result.get('result', ''))[:100]
        })
        
        # ✅ Success - return immediately
        if not result.get('error'):
            print(f"✅ SUCCESS on attempt {attempt + 1}!")
            result["execution_history"] = execution_history
            result["self_healing_used"] = attempt > 0
            return result
        
        # ❌ Error - show detailed error information before healing
        print(f"\n💥 EXECUTION ERROR DETECTED!")
        print(f"{'='*60}")
        print(f"🚨 Attempt: {attempt + 1}/{max_retries + 1}")
        print(f"🔍 Error Type: {type(result.get('error', '')).__name__}")
        print(f"📝 Full Error Message:")
        print(f"   {result.get('error', 'Unknown error')}")
        
        if result.get('stderr'):
            print(f"📢 Stderr Output:")
            print(f"   {result['stderr']}")
            
        if result.get('stdout'):
            print(f"📄 Stdout Before Error:")
            print(f"   {result['stdout']}")
            
        print(f"💻 Code Length: {len(current_code)} characters")
        print(f"📝 Code That Failed:")
        print(f"{'='*40}")
        # Show the failing code with line numbers
        for i, line in enumerate(current_code.split('\n'), 1):
            print(f"   {i:2d}: {line}")
        print(f"{'='*40}")
        
        if attempt < max_retries:
            print(f"\n🧠 INITIATING LLM SELF-HEALING...")
            
            # Use LLM agent for intelligent error analysis and fixing
            try:
                error_analysis = await analyze_error_with_llm_agent(
                    result['error'], current_code, input_data, attempt + 1
                )
                
                if error_analysis.get('corrected_code') and error_analysis['confidence'] > 0.6:
                    current_code = error_analysis['corrected_code']
                    print(f"🛠️ LLM generated corrected code for next attempt")
                    print(f"🎯 Fix Confidence: {error_analysis['confidence']:.1%}")
                    print(f"🧠 Healing Method: {error_analysis.get('healing_method', 'unknown')}")
                    
                    # Show detailed preview of the corrected code
                    print(f"\n🔧 CORRECTED CODE PREVIEW:")
                    print(f"{'='*50}")
                    code_lines = current_code.split('\n')[:10]  # Show first 10 lines
                    for i, line in enumerate(code_lines, 1):
                        print(f"   {i:2d}: {line}")
                    if len(current_code.split('\n')) > 10:
                        print(f"   ... ({len(current_code.split('\n')) - 10} more lines)")
                    print(f"{'='*50}")
                else:
                    print(f"🤷 LLM unable to generate reliable fix (confidence: {error_analysis.get('confidence', 0):.1%})")
                    break
                    
            except Exception as healing_error:
                print(f"❌ LLM healing failed: {str(healing_error)}")
                print(f"� No fallback available - stopping healing attempts")
                break
        else:
            print(f"🚨 Max retries exceeded - returning final error")
    
    # Final result with healing attempt history
    result["execution_history"] = execution_history
    result["self_healing_attempted"] = True
    result["final_status"] = "failed_after_healing"
    
    print(f"\n🎯 SELF-HEALING SUMMARY:")
    print(f"{'='*40}")
    for i, hist in enumerate(execution_history):
        status = "✅ SUCCESS" if hist["success"] else "❌ ERROR" 
        print(f"   Attempt {i+1}: {status}")
    print(f"{'='*40}")
    
    return result