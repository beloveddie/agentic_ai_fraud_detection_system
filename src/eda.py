# src/eda.py

"""
Exploratory Data Analysis (EDA) Script for Fraud Detection
----------------------------------------------------------
Loads transaction data, performs statistical summaries, 
and visualizes important relationships for fraud detection.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

plt.style.use("seaborn-v0_8")

DATA_PATH = os.path.join("data", "transactions.csv")

# ======================================================
# 1️⃣ LOAD DATA
# ======================================================
def load_data(path: str) -> pd.DataFrame:
    """Load the transaction dataset from CSV."""
    df = pd.read_csv(path)
    
    # Parse the datetime column and extract useful time features
    df['time'] = pd.to_datetime(df['time'])
    df['transaction_hour'] = df['time'].dt.hour
    df['transaction_day'] = df['time'].dt.day
    df['transaction_month'] = df['time'].dt.month
    df['transaction_weekday'] = df['time'].dt.dayofweek
    
    print(f"✅ Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")
    print(f"Columns: {list(df.columns)}\n")
    return df


# ======================================================
# 2️⃣ BASIC SUMMARY
# ======================================================
def summarize(df: pd.DataFrame):
    """Print dataset summary and missing values."""
    print("📊 Dataset Info:")
    print(df.info())
    print("\n🔍 Missing Values:")
    print(df.isnull().sum())
    print("\n📈 Descriptive Statistics:")
    print(df.describe(include="all"))


# ======================================================
# 3️⃣ DISTRIBUTIONS & VISUALS
# ======================================================
def visualize(df: pd.DataFrame):
    """Generate key EDA plots."""
    # Fraud balance
    plt.figure(figsize=(6, 4))
    sns.countplot(x="fraud_label", data=df, palette="coolwarm")
    plt.title("Fraud vs Non-Fraud Transactions")
    plt.xlabel("Fraud Label (0=Non-Fraud, 1=Fraud)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Transaction amount by fraud
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="fraud_label", y="amount", data=df, palette="coolwarm")
    plt.title("Transaction Amount Distribution by Fraud Label")
    plt.xlabel("Fraud Label")
    plt.ylabel("Amount")
    plt.tight_layout()
    plt.show()

    # Hourly pattern
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="transaction_hour", hue="fraud_label", bins=24, multiple="stack", palette="coolwarm")
    plt.title("Transaction Hour vs Fraud")
    plt.xlabel("Hour of Day")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Transaction type
    plt.figure(figsize=(7, 5))
    sns.countplot(x="transaction_type", hue="fraud_label", data=df, palette="coolwarm")
    plt.title("Fraud Rate by Transaction Type")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

    # Channel distribution
    plt.figure(figsize=(7, 5))
    sns.countplot(x="channel", hue="fraud_label", data=df, palette="coolwarm")
    plt.title("Fraud Rate by Channel")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

    # Merchant category
    plt.figure(figsize=(10, 5))
    sns.countplot(x="merchant_category", hue="fraud_label", data=df, palette="coolwarm")
    plt.title("Fraud Rate by Merchant Category")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Location country
    plt.figure(figsize=(8, 5))
    sns.countplot(x="location_country", hue="fraud_label", data=df, palette="coolwarm")
    plt.title("Fraud Rate by Country")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

    # Is foreign transaction
    plt.figure(figsize=(6, 4))
    sns.countplot(x="is_foreign", hue="fraud_label", data=df, palette="coolwarm")
    plt.title("Fraud Rate: Foreign vs Domestic Transactions")
    plt.xlabel("Is Foreign (0=Domestic, 1=Foreign)")
    plt.tight_layout()
    plt.show()

    # Account age vs fraud
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="fraud_label", y="account_age_days", data=df, palette="coolwarm")
    plt.title("Account Age Distribution by Fraud Label")
    plt.xlabel("Fraud Label")
    plt.ylabel("Account Age (Days)")
    plt.tight_layout()
    plt.show()

    # Correlation heatmap (numerical features only)
    plt.figure(figsize=(12, 8))
    numerical_cols = ['amount', 'account_age_days', 'num_prev_transactions_24h', 
                     'avg_transaction_amount_7d', 'device_change_rate_30d', 
                     'is_foreign', 'is_blacklisted_merchant', 'fraud_label',
                     'transaction_hour', 'transaction_weekday']
    corr = df[numerical_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.show()


# ======================================================
# 4️⃣ FRAUD INSIGHTS
# ======================================================
def fraud_insights(df: pd.DataFrame):
    """Quick ratio and pattern insights."""
    fraud_rate = df["fraud_label"].mean()
    print(f"🚨 Overall Fraud Rate: {fraud_rate:.2%}\n")

    # Fraud rate by transaction type
    fraud_by_type = df.groupby("transaction_type")["fraud_label"].mean().sort_values(ascending=False)
    print("📦 Fraud rate by transaction type:")
    for tx_type, rate in fraud_by_type.items():
        print(f"  {tx_type}: {rate:.2%}")
    print()

    # Fraud rate by channel
    fraud_by_channel = df.groupby("channel")["fraud_label"].mean().sort_values(ascending=False)
    print("📱 Fraud rate by channel:")
    for channel, rate in fraud_by_channel.items():
        print(f"  {channel}: {rate:.2%}")
    print()

    # Fraud rate by merchant category
    fraud_by_merchant = df.groupby("merchant_category")["fraud_label"].mean().sort_values(ascending=False)
    print("🏪 Fraud rate by merchant category:")
    for category, rate in fraud_by_merchant.items():
        print(f"  {category}: {rate:.2%}")
    print()

    # Fraud rate by country
    fraud_by_country = df.groupby("location_country")["fraud_label"].mean().sort_values(ascending=False)
    print("🌍 Fraud rate by country:")
    for country, rate in fraud_by_country.items():
        print(f"  {country}: {rate:.2%}")
    print()

    # Fraud rate by hour
    fraud_by_hour = df.groupby("transaction_hour")["fraud_label"].mean().sort_values(ascending=False)
    print("🕓 Top 10 fraud hours:")
    for hour, rate in fraud_by_hour.head(10).items():
        print(f"  Hour {hour}: {rate:.2%}")
    print()

    # Foreign vs domestic transactions
    foreign_fraud = df[df['is_foreign'] == 1]['fraud_label'].mean()
    domestic_fraud = df[df['is_foreign'] == 0]['fraud_label'].mean()
    print(f"🌐 Foreign transactions fraud rate: {foreign_fraud:.2%}")
    print(f"🏠 Domestic transactions fraud rate: {domestic_fraud:.2%}")
    print()

    # Blacklisted merchant impact
    blacklisted_fraud = df[df['is_blacklisted_merchant'] == 1]['fraud_label'].mean()
    regular_fraud = df[df['is_blacklisted_merchant'] == 0]['fraud_label'].mean()
    print(f"⚠️  Blacklisted merchant fraud rate: {blacklisted_fraud:.2%}")
    print(f"✅ Regular merchant fraud rate: {regular_fraud:.2%}")
    print()

    # Amount statistics
    fraud_amounts = df[df['fraud_label'] == 1]['amount']
    normal_amounts = df[df['fraud_label'] == 0]['amount']
    print("💰 Amount statistics:")
    print(f"  Average fraud amount: {fraud_amounts.mean():.2f} {df['currency'].iloc[0]}")
    print(f"  Average normal amount: {normal_amounts.mean():.2f} {df['currency'].iloc[0]}")
    print(f"  Median fraud amount: {fraud_amounts.median():.2f} {df['currency'].iloc[0]}")
    print(f"  Median normal amount: {normal_amounts.median():.2f} {df['currency'].iloc[0]}")


# ======================================================
# 🚀 MAIN EXECUTION
# ======================================================
if __name__ == "__main__":
    df = load_data(DATA_PATH)
    summarize(df)
    visualize(df)
    fraud_insights(df)
    print("\n✅ EDA Completed Successfully!")