# tools/data_loader.py
"""
Enhanced data loader utility for fraud detection system
- Loads transactions from CSV with robust error handling
- Provides transaction sampling and filtering capabilities
- Centralized data access with caching for performance
- Real-time transaction simulation capabilities
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timedelta
import json

from agents import function_tool

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DATA_PATH = os.path.join("data", "transactions.csv")

# Cache for loaded data to improve performance
_cached_data = None
_cache_timestamp = None
_cache_ttl = 300  # 5 minutes


def load_csv(path: str = DATA_PATH, use_cache: bool = True) -> pd.DataFrame:
    """
    Load transactions CSV with caching and enhanced error handling.
    
    Args:
        path: Path to CSV file
        use_cache: Whether to use cached data if available
        
    Returns:
        DataFrame with parsed datetime and properly typed columns
    """
    global _cached_data, _cache_timestamp
    
    # Check cache validity
    if use_cache and _cached_data is not None and _cache_timestamp is not None:
        if (datetime.now() - _cache_timestamp).seconds < _cache_ttl:
            return _cached_data.copy()
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Data file not found at {path}. Please ensure the CSV file exists.")
    
    try:
        # Load with proper data types
        df = pd.read_csv(path)
        
        # Parse datetime if time column exists
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors='coerce')
        
        # Ensure proper data types for key columns
        if "fraud_label" in df.columns:
            df["fraud_label"] = df["fraud_label"].astype(int)
        if "amount" in df.columns:
            df["amount"] = pd.to_numeric(df["amount"], errors='coerce')
        if "is_foreign" in df.columns:
            df["is_foreign"] = df["is_foreign"].astype(int)
        if "is_blacklisted_merchant" in df.columns:
            df["is_blacklisted_merchant"] = df["is_blacklisted_merchant"].astype(int)
            
        # Cache the data
        if use_cache:
            _cached_data = df.copy()
            _cache_timestamp = datetime.now()
            
        print(f"✅ Loaded {len(df):,} transactions from {path}")
        return df
        
    except Exception as e:
        raise ValueError(f"❌ Error loading CSV file: {str(e)}")


def clear_cache():
    """Clear the data cache to force reload on next access."""
    global _cached_data, _cache_timestamp
    _cached_data = None
    _cache_timestamp = None
    print("🗑️  Data cache cleared")


def get_transaction_by_id(transaction_id: str, path: str = DATA_PATH) -> Dict[str, Any]:
    """
    Get a specific transaction by its ID.
    
    Args:
        transaction_id: The transaction ID to find
        path: Path to CSV file
        
    Returns:
        Dictionary representation of the transaction
    """
    try:
        df = load_csv(path)
        row = df[df["transaction_id"].astype(str) == str(transaction_id)]
        
        if row.empty:
            raise KeyError(f"❌ Transaction ID '{transaction_id}' not found in dataset")
            
        transaction = row.iloc[0].to_dict()
        
        # Convert NaN values to None for JSON serialization
        for key, value in transaction.items():
            if pd.isna(value):
                transaction[key] = None
                
        print(f"✅ Found transaction: {transaction_id}")
        return transaction
        
    except Exception as e:
        raise ValueError(f"❌ Error retrieving transaction: {str(e)}")


def sample_transaction(idx: Optional[int] = None, path: str = DATA_PATH, 
                      fraud_only: bool = False, legitimate_only: bool = False) -> Dict[str, Any]:
    """
    Sample a transaction from the dataset with filtering options.
    
    Args:
        idx: Specific index to sample (if None, random sample)
        path: Path to CSV file
        fraud_only: If True, only sample fraudulent transactions
        legitimate_only: If True, only sample legitimate transactions
        
    Returns:
        Dictionary representation of the sampled transaction
    """
    try:
        df = load_csv(path)
        
        # Apply filters
        if fraud_only and "fraud_label" in df.columns:
            df = df[df["fraud_label"] == 1]
            if df.empty:
                raise ValueError("❌ No fraudulent transactions found in dataset")
                
        elif legitimate_only and "fraud_label" in df.columns:
            df = df[df["fraud_label"] == 0]
            if df.empty:
                raise ValueError("❌ No legitimate transactions found in dataset")
        
        # Sample transaction
        if idx is None:
            transaction = df.sample(1).iloc[0].to_dict()
            print(f"🎲 Randomly sampled transaction: {transaction.get('transaction_id', 'Unknown')}")
        else:
            if idx < 0 or idx >= len(df):
                raise IndexError(f"❌ Index {idx} out of bounds (dataset size: {len(df)})")
            transaction = df.iloc[idx].to_dict()
            print(f"📍 Sampled transaction at index {idx}: {transaction.get('transaction_id', 'Unknown')}")
        
        # Convert NaN values to None
        for key, value in transaction.items():
            if pd.isna(value):
                transaction[key] = None
                
        return transaction
        
    except Exception as e:
        raise ValueError(f"❌ Error sampling transaction: {str(e)}")


def get_dataset_stats(path: str = DATA_PATH) -> Dict[str, Any]:
    """
    Get comprehensive statistics about the dataset.
    
    Returns:
        Dictionary with dataset statistics
    """
    try:
        df = load_csv(path)
        
        stats = {
            "total_transactions": len(df),
            "date_range": {},
            "fraud_distribution": {},
            "amount_stats": {},
            "channels": {},
            "countries": {},
            "transaction_types": {}
        }
        
        # Date range analysis
        if "time" in df.columns and not df["time"].isna().all():
            stats["date_range"] = {
                "earliest": str(df["time"].min()),
                "latest": str(df["time"].max()),
                "span_days": (df["time"].max() - df["time"].min()).days
            }
        
        # Fraud distribution
        if "fraud_label" in df.columns:
            fraud_counts = df["fraud_label"].value_counts()
            total = len(df)
            stats["fraud_distribution"] = {
                "legitimate_count": int(fraud_counts.get(0, 0)),
                "fraud_count": int(fraud_counts.get(1, 0)),
                "fraud_rate": float(df["fraud_label"].mean()),
                "fraud_percentage": f"{df['fraud_label'].mean() * 100:.2f}%"
            }
        
        # Amount statistics
        if "amount" in df.columns:
            stats["amount_stats"] = {
                "mean": float(df["amount"].mean()),
                "median": float(df["amount"].median()),
                "min": float(df["amount"].min()),
                "max": float(df["amount"].max()),
                "std": float(df["amount"].std())
            }
        
        # Channel distribution
        if "channel" in df.columns:
            stats["channels"] = df["channel"].value_counts().to_dict()
            
        # Country distribution
        if "location_country" in df.columns:
            stats["countries"] = df["location_country"].value_counts().to_dict()
            
        # Transaction type distribution
        if "transaction_type" in df.columns:
            stats["transaction_types"] = df["transaction_type"].value_counts().to_dict()
        
        return stats
        
    except Exception as e:
        raise ValueError(f"❌ Error computing dataset statistics: {str(e)}")


def filter_transactions(path: str = DATA_PATH, **filters) -> List[Dict[str, Any]]:
    """
    Filter transactions based on specified criteria.
    
    Args:
        path: Path to CSV file
        **filters: Keyword arguments for filtering (e.g., fraud_label=1, channel='web')
        
    Returns:
        List of transactions matching the criteria
    """
    try:
        df = load_csv(path)
        
        # Apply filters
        for column, value in filters.items():
            if column in df.columns:
                if isinstance(value, (list, tuple)):
                    df = df[df[column].isin(value)]
                else:
                    df = df[df[column] == value]
            else:
                print(f"⚠️  Warning: Column '{column}' not found in dataset")
        
        # Convert to list of dictionaries
        transactions = []
        for _, row in df.iterrows():
            transaction = row.to_dict()
            # Convert NaN values to None
            for key, value in transaction.items():
                if pd.isna(value):
                    transaction[key] = None
            transactions.append(transaction)
        
        print(f"🔍 Found {len(transactions)} transactions matching filters: {filters}")
        return transactions
        
    except Exception as e:
        raise ValueError(f"❌ Error filtering transactions: {str(e)}")


@function_tool(strict_mode=False)
async def get_customer_history(customer_id: str, path: str = DATA_PATH, 
                        limit: Optional[int] = None, 
                        exclude_current_transaction: Optional[str] = None) -> list:
    """
    Get historical transactions for a specific customer.
    
    Args:
        customer_id: The customer ID to get history for
        path: Path to CSV file
        limit: Maximum number of transactions to return (None for all)
        exclude_current_transaction: Transaction ID to exclude from history
        
    Returns:
        List of historical transactions for the customer
    """
    try:
        df = load_csv(path)
        
        # Filter by customer ID
        customer_df = df[df["customer_id"].astype(str) == str(customer_id)]
        
        if customer_df.empty:
            print(f"⚠️  No transactions found for customer: {customer_id}")
            return []
        
        # Exclude current transaction if specified
        if exclude_current_transaction:
            customer_df = customer_df[customer_df["transaction_id"].astype(str) != str(exclude_current_transaction)]
        
        # Sort by time (most recent first)
        if "time" in customer_df.columns:
            customer_df = customer_df.sort_values("time", ascending=False)
        
        # Apply limit
        if limit:
            customer_df = customer_df.head(limit)
        
        # Convert to list of dictionaries
        transactions = []
        for _, row in customer_df.iterrows():
            transaction = row.to_dict()
            # Convert NaN values to None
            for key, value in transaction.items():
                if pd.isna(value):
                    transaction[key] = None
            transactions.append(transaction)
        
        print(f"📚 Found {len(transactions)} historical transactions for customer: {customer_id}")
        return transactions
        
    except Exception as e:
        raise ValueError(f"❌ Error retrieving customer history: {str(e)}")


def export_sample(output_path: str, sample_size: int = 100, path: str = DATA_PATH, 
                 stratify_fraud: bool = True) -> bool:
    """
    Export a sample of transactions to a new CSV file.
    
    Args:
        output_path: Path for the output CSV file
        sample_size: Number of transactions to sample
        path: Source CSV path
        stratify_fraud: Whether to maintain fraud ratio in sample
        
    Returns:
        True if successful
    """
    try:
        df = load_csv(path)
        
        if sample_size >= len(df):
            sample_df = df.copy()
            print(f"⚠️  Requested sample size ({sample_size}) >= dataset size ({len(df)}). Using full dataset.")
        elif stratify_fraud and "fraud_label" in df.columns:
            # Stratified sampling to maintain fraud ratio
            fraud_ratio = df["fraud_label"].mean()
            fraud_count = int(sample_size * fraud_ratio)
            legitimate_count = sample_size - fraud_count
            
            fraud_sample = df[df["fraud_label"] == 1].sample(min(fraud_count, (df["fraud_label"] == 1).sum()), random_state=42)
            legitimate_sample = df[df["fraud_label"] == 0].sample(min(legitimate_count, (df["fraud_label"] == 0).sum()), random_state=42)
            
            sample_df = pd.concat([fraud_sample, legitimate_sample]).sample(frac=1, random_state=42)
        else:
            # Random sampling
            sample_df = df.sample(sample_size, random_state=42)
        
        # Export to CSV
        sample_df.to_csv(output_path, index=False)
        print(f"✅ Exported {len(sample_df)} transactions to {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error exporting sample: {str(e)}")
        return False


# CLI interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Data Loader for Fraud Detection System")
    parser.add_argument("--action", choices=["stats", "sample", "filter", "export"], 
                       default="stats", help="Action to perform")
    parser.add_argument("--sample-idx", type=int, help="Specific index to sample")
    parser.add_argument("--fraud-only", action="store_true", help="Sample only fraudulent transactions")
    parser.add_argument("--legitimate-only", action="store_true", help="Sample only legitimate transactions")
    parser.add_argument("--export-size", type=int, default=100, help="Size of sample to export")
    parser.add_argument("--export-path", default="sample_transactions.csv", help="Path for exported sample")
    parser.add_argument("--transaction-id", help="Specific transaction ID to retrieve")
    
    args = parser.parse_args()
    
    try:
        if args.action == "stats":
            print("📊 Dataset Statistics")
            print("=" * 50)
            stats = get_dataset_stats()
            print(json.dumps(stats, indent=2))
            
        elif args.action == "sample":
            if args.transaction_id:
                print(f"🔍 Retrieving transaction: {args.transaction_id}")
                transaction = get_transaction_by_id(args.transaction_id)
            else:
                print("🎲 Sampling transaction...")
                transaction = sample_transaction(
                    idx=args.sample_idx,
                    fraud_only=args.fraud_only,
                    legitimate_only=args.legitimate_only
                )
            print(json.dumps(transaction, indent=2, default=str))
            
        elif args.action == "filter":
            print("🔍 Filter example - showing first 5 web transactions:")
            transactions = filter_transactions(channel="web")[:5]
            for i, tx in enumerate(transactions, 1):
                print(f"\n{i}. {tx['transaction_id']}: {tx['amount']} {tx.get('currency', 'NGN')}")
                
        elif args.action == "export":
            print(f"📤 Exporting {args.export_size} transactions...")
            success = export_sample(args.export_path, args.export_size)
            if success:
                print(f"✅ Sample exported successfully to {args.export_path}")
            else:
                print("❌ Export failed")
                
    except Exception as e:
        print(f"❌ Error: {str(e)}")