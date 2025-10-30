# examples/example_usage.py
"""
Demo: load a sample transaction, call the fraud_detector, and print prettified output.
"""

from tools.data_loader import sample_transaction, get_transaction_by_id
from tools.fraud_detector import get_default_detector
import json

def pretty_print(result):
    print("=== Fraud Detector Result ===")
    print(f"Score: {result['score']:.4f}")
    print(f"Label (threshold {result['threshold']}): {result['label']}")
    print("Top reasons (feature,score):")
    for feat, val in result["top_reasons"]:
        print(f"  - {feat}: {val}")
    print("Raw transaction keys:", list(result["raw"]["transaction"].keys())[:10])

if __name__ == "__main__":
    tx = sample_transaction()  # picks a random row from data/transactions.csv
    detector = get_default_detector()
    res = detector.predict(tx)
    pretty_print(res)
    # Also show JSON (for agent/tool consumption)
    print("\nJSON payload for agent:")
    print(json.dumps(res, indent=2))