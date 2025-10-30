"""
generate_transaction_dataset.py
--------------------------------
Synthetic Transaction Dataset Generator for Fraud Detection Agent

This script generates a realistic transaction dataset with labeled fraud instances.
Fraud labels are assigned probabilistically based on behavioral and contextual rules.

Author: Bass Major
Project: Transaction Analysis Agent for Fraud
Date: 2025-10-22
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os


def generate_synthetic_transactions(n_samples: int = 10000, seed: int = 42, save_path: str = "synthetic_transactions.csv"):
    """
    Generate a synthetic dataset representing customer transactions with fraud labels.

    Parameters
    ----------
    n_samples : int
        Number of transactions to generate.
    seed : int
        Random seed for reproducibility.
    save_path : str
        Path to save the generated CSV file.

    Returns
    -------
    pd.DataFrame
        Generated transaction dataset.
    """
    np.random.seed(seed)
    random.seed(seed)

    customers = [f"CUST_{i:04d}" for i in range(1, 2001)]
    devices = [f"DEV_{i:05d}" for i in range(1, 5001)]
    merchants = ['electronics', 'groceries', 'airtime', 'betting', 'luxury', 'clothing', 'transport', 'food']
    cities = ['Lagos', 'Abuja', 'Port Harcourt', 'London', 'Accra']
    countries = ['Nigeria', 'Ghana', 'UK', 'USA']
    channels = ['mobile', 'web', 'ATM', 'POS']
    txn_types = ['transfer', 'purchase', 'withdrawal']

    def random_time():
        start = datetime(2025, 10, 1)
        end = datetime(2025, 10, 21)
        return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

    data = []

    for i in range(n_samples):
        cust = random.choice(customers)
        amt = round(np.random.exponential(5000) + np.random.normal(0, 1000), 2)
        avg7d = max(1000, np.random.normal(5000, 2000))
        dev_rate = round(np.random.beta(2, 5), 2)
        freq24h = np.random.poisson(4)
        country = random.choice(countries)
        blacklisted = np.random.choice([0, 1], p=[0.97, 0.03])
        channel = random.choice(channels)
        fraud = 0

        # Fraud trigger conditions
        if (
            amt > 10 * avg7d
            or dev_rate > 0.7
            or country != 'Nigeria'
            or (blacklisted == 1 and channel in ['web', 'POS'])
            or freq24h > 15
        ):
            fraud = np.random.choice([1, 0], p=[0.8, 0.2])

        data.append({
            "transaction_id": f"TXN_{i + 1:05d}",
            "customer_id": cust,
            "amount": amt,
            "currency": "NGN",
            "transaction_type": random.choice(txn_types),
            "channel": channel,
            "merchant_category": random.choice(merchants),
            "device_id": random.choice(devices),
            "ip_address": f"197.210.{random.randint(0, 255)}.{random.randint(0, 255)}",
            "location_country": country,
            "location_city": random.choice(cities),
            "time": random_time(),
            "account_age_days": np.random.randint(30, 1500),
            "num_prev_transactions_24h": freq24h,
            "avg_transaction_amount_7d": avg7d,
            "device_change_rate_30d": dev_rate,
            "is_foreign": int(country != 'Nigeria'),
            "is_blacklisted_merchant": blacklisted,
            "fraud_label": fraud,
        })

    df = pd.DataFrame(data)
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    df.to_csv(save_path, index=False)

    fraud_ratio = df['fraud_label'].mean() * 100
    print(f"✅ Dataset generated successfully: {len(df):,} transactions")
    print(f"💡 Fraudulent transactions: {fraud_ratio:.2f}%")
    print(f"📁 Saved to: {os.path.abspath(save_path)}")

    return df


if __name__ == "__main__":
    generate_synthetic_transactions()
