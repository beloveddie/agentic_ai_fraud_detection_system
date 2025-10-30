# tools/fraud_detector.py
"""
Fraud detector tool
- Loads the trained pipeline (artifacts/lgbm_model.joblib)
- Loads preprocessing metadata (artifacts/preprocess_meta.json)
- Provides fraud detection with SHAP explanations
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import warnings

from typing_extensions import TypedDict

from agents import function_tool

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# SHAP for explainable AI
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

ARTIFACT_DIR = os.path.join("artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "lgbm_model.joblib")
META_PATH = os.path.join(ARTIFACT_DIR, "preprocess_meta.json")


# TypedDict definitions for OpenAI SDK format
class Transaction(TypedDict):
    """Transaction data structure for fraud detection."""
    transaction_id: str
    customer_id: str
    amount: float
    currency: str
    transaction_type: str
    channel: str
    merchant_category: str
    device_id: str
    ip_address: str
    location_country: str
    location_city: str
    time: str
    account_age_days: int
    num_prev_transactions_24h: int
    avg_transaction_amount_7d: float
    device_change_rate_30d: float
    is_foreign: int
    is_blacklisted_merchant: int


class FraudPrediction(TypedDict):
    """Fraud prediction result structure."""
    score: float
    label: int
    risk_level: str
    threshold: float
    confidence: float
    top_reasons: List[Tuple[str, float]]
    recommendations: List[str]
    model_details: Dict[str, Any]


class FraudDetector:
    def __init__(self, model_path: str = MODEL_PATH, meta_path: str = META_PATH, 
                 probability_threshold: float = 0.5, enable_shap: bool = True):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Preprocess meta not found at {meta_path}")

        self.pipeline = joblib.load(model_path)  # pipeline: preprocessor -> clf
        with open(meta_path, "r") as f:
            self.meta = json.load(f)

        # feature names *after* preprocessing (e.g., numeric + onehot feature names)
        self.feature_names = self.meta.get("feature_names", None)
        self.expected_input_cols = self.meta.get("numeric_features", []) + self.meta.get("categorical_features", [])
        self.threshold = probability_threshold

        # attempt to access the underlying classifier for importances
        try:
            self.clf = self.pipeline.named_steps.get("clf", None)
        except Exception:
            self.clf = None

        # Initialize SHAP explainer
        self.shap_explainer = None
        self.enable_shap = enable_shap and SHAP_AVAILABLE
        if self.enable_shap and self.clf is not None:
            self._initialize_shap_explainer()

    def _initialize_shap_explainer(self):
        """Initialize SHAP explainer for model interpretability."""
        try:
            print("🔍 Initializing SHAP explainer for enhanced fraud explanation...")
            
            # For tree-based models like LightGBM, use TreeExplainer
            if hasattr(self.clf, 'booster_'):
                self.shap_explainer = shap.TreeExplainer(self.clf)
                print("✅ SHAP TreeExplainer initialized successfully")
            else:
                # Fallback to other explainer types if needed
                print("⚠️  Using fallback SHAP explainer")
                # We'll initialize this when we have background data
                self.shap_explainer = None
                
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize SHAP explainer: {e}")
            self.shap_explainer = None
            self.enable_shap = False

    def _normalize_transaction(self, tx: Dict[str, Any]) -> pd.DataFrame:
        """
        Create a single-row DataFrame with expected input columns for the preprocessor.
        Handle datetime parsing and feature engineering to match training pipeline.
        """
        # Import feature engineering function
        try:
            from src.preprocess import basic_feature_engineering
        except ImportError:
            # Fallback if import fails
            basic_feature_engineering = None
        
        # Create row with expected columns
        row = {}
        
        # Handle datetime if provided
        if "time" in tx and tx["time"] is not None:
            try:
                if isinstance(tx["time"], str):
                    time_obj = pd.to_datetime(tx["time"])
                else:
                    time_obj = tx["time"]
                row["time"] = time_obj
                row["transaction_hour"] = time_obj.hour
                row["transaction_day"] = time_obj.day
                row["transaction_month"] = time_obj.month
                row["transaction_weekday"] = time_obj.dayofweek
            except Exception:
                # Use current time as fallback
                now = datetime.now()
                row["time"] = now
                row["transaction_hour"] = now.hour
                row["transaction_day"] = now.day
                row["transaction_month"] = now.month
                row["transaction_weekday"] = now.weekday()
        
        # Copy all provided fields
        for key, value in tx.items():
            if key != "time":  # Skip time as we handled it above
                row[key] = value
        
        # Fill missing columns with sensible defaults
        for col in self.expected_input_cols:
            if col not in row:
                if col in ["amount", "amount_abs", "amount_log", "amount_vs_avg_ratio"]:
                    row[col] = 0.0
                elif col in ["account_age_days", "num_prev_transactions_24h", "avg_transaction_amount_7d"]:
                    row[col] = 0.0
                elif col in ["device_change_rate_30d"]:
                    row[col] = 0.0
                elif col in ["is_foreign", "is_blacklisted_merchant", "high_freq_24h", "high_device_change", "is_negative_amount"]:
                    row[col] = 0
                elif col in ["hour_of_day", "transaction_hour", "transaction_day", "transaction_month", "transaction_weekday"]:
                    row[col] = 0
                elif col in ["transaction_type", "channel", "merchant_category", "location_country", "location_city", "currency"]:
                    row[col] = "unknown"
                elif col == "account_age_category":
                    row[col] = "unknown"
                else:
                    row[col] = 0.0
        
        # Create DataFrame
        df = pd.DataFrame([row])
        
        # Apply feature engineering if available
        if basic_feature_engineering:
            try:
                df = basic_feature_engineering(df)
            except Exception as e:
                print(f"Warning: Feature engineering failed: {e}")
                # Manually add key engineered features
                if "amount" in df.columns:
                    df["amount_abs"] = df["amount"].abs()
                    df["amount_log"] = np.log1p(df["amount_abs"])
                    df["is_negative_amount"] = (df["amount"] < 0).astype(int)
                
                if "avg_transaction_amount_7d" in df.columns and "amount" in df.columns:
                    df["amount_vs_avg_ratio"] = df["amount_abs"] / (df["avg_transaction_amount_7d"] + 1e-9)
                
                if "num_prev_transactions_24h" in df.columns:
                    df["high_freq_24h"] = (df["num_prev_transactions_24h"] > 5).astype(int)
                
                if "device_change_rate_30d" in df.columns:
                    df["high_device_change"] = (df["device_change_rate_30d"] > 0.5).astype(int)
        
        return df

    def predict(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict fraud for a single transaction dict with SHAP explanations.

        Returns:
            {
                "score": 0.78,
                "label": 1,
                "risk_level": "HIGH/MEDIUM/LOW",
                "threshold": 0.5,
                "confidence": 0.85,
                "top_reasons": [("device_change_rate_30d", 0.21), ...],
                "shap_explanation": {"feature_contributions": [...], "base_value": 0.5},
                "recommendations": ["Review device activity", ...],
                "model_details": {...},
                "raw": {"transaction": transaction}
            }
        """
        try:
            df_input = self._normalize_transaction(transaction)

            # Predict probability
            proba = float(self.pipeline.predict_proba(df_input)[:, 1][0])
            label = int(proba >= self.threshold)
            
            # Calculate confidence (distance from threshold)
            confidence = abs(proba - 0.5) * 2  # Scale to 0-1
            
            # Determine risk level
            if proba >= 0.7:
                risk_level = "HIGH"
            elif proba >= 0.4:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            # Build explanation using SHAP if available, fallback to feature importance
            top_reasons, shap_explanation = self._get_enhanced_explanations(df_input, transaction)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(proba, transaction, top_reasons)

            result = {
                "score": round(proba, 4),
                "label": label,
                "risk_level": risk_level,
                "threshold": self.threshold,
                "confidence": round(confidence, 4),
                "top_reasons": top_reasons[:5],  # Top 5 reasons
                "shap_explanation": shap_explanation,
                "recommendations": recommendations,
                "model_details": {
                    "model_type": "LightGBM",
                    "features_used": len(self.feature_names) if self.feature_names else "unknown",
                    "preprocessing_applied": True,
                    "shap_enabled": self.enable_shap,
                    "explainer_type": "TreeExplainer" if self.shap_explainer else "FeatureImportance"
                },
                "raw": {"transaction": transaction}
            }
            
            return result
            
        except Exception as e:
            # Fallback response for errors
            return {
                "score": 0.5,
                "label": 0,
                "risk_level": "UNKNOWN",
                "threshold": self.threshold,
                "confidence": 0.0,
                "top_reasons": [],
                "shap_explanation": None,
                "recommendations": ["Review transaction manually due to processing error"],
                "error": str(e),
                "model_details": {"error": True},
                "raw": {"transaction": transaction}
            }

    def _get_enhanced_explanations(self, df_input: pd.DataFrame, transaction: Dict[str, Any]) -> Tuple[List[Tuple[str, float]], Optional[Dict[str, Any]]]:
        """Get enhanced explanations using SHAP if available, otherwise fallback to feature importance."""
        
        shap_explanation = None
        explanations = []
        
        # Try SHAP explanation first
        if self.enable_shap and self.shap_explainer is not None:
            try:
                # Get preprocessed features for SHAP
                X_processed = self.pipeline.named_steps['preprocessor'].transform(df_input)
                
                # Calculate SHAP values
                shap_values = self.shap_explainer.shap_values(X_processed)
                
                # Handle different SHAP output formats
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    # Binary classification - use positive class
                    shap_values_fraud = shap_values[1][0]  # First (and only) sample, fraud class
                elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 2:
                    # Single array for binary classification
                    shap_values_fraud = shap_values[0]  # First sample
                else:
                    # Fallback
                    shap_values_fraud = shap_values[0] if isinstance(shap_values, np.ndarray) else shap_values
                
                # Get base value (expected prediction)
                if hasattr(self.shap_explainer, 'expected_value'):
                    if isinstance(self.shap_explainer.expected_value, np.ndarray):
                        base_value = float(self.shap_explainer.expected_value[1] if len(self.shap_explainer.expected_value) > 1 else self.shap_explainer.expected_value[0])
                    else:
                        base_value = float(self.shap_explainer.expected_value)
                else:
                    base_value = 0.5  # Default for binary classification
                
                # Create feature contributions list
                feature_contributions = []
                if self.feature_names and len(shap_values_fraud) == len(self.feature_names):
                    for i, (feature, shap_val) in enumerate(zip(self.feature_names, shap_values_fraud)):
                        feature_contributions.append({
                            "feature": feature,
                            "shap_value": float(shap_val),
                            "feature_value": float(X_processed[0][i]) if i < len(X_processed[0]) else 0.0,
                            "contribution": "increases_fraud" if shap_val > 0 else "decreases_fraud"
                        })
                    
                    # Sort by absolute SHAP value for top reasons
                    explanations = [(contrib["feature"], abs(contrib["shap_value"])) 
                                  for contrib in feature_contributions]
                    explanations = sorted(explanations, key=lambda x: x[1], reverse=True)
                
                # Create SHAP explanation object
                shap_explanation = {
                    "base_value": base_value,
                    "prediction": base_value + sum(shap_values_fraud),
                    "feature_contributions": sorted(feature_contributions, 
                                                  key=lambda x: abs(x["shap_value"]), reverse=True),
                    "explanation_method": "SHAP_TreeExplainer",
                    "top_positive_factors": [contrib for contrib in feature_contributions 
                                           if contrib["shap_value"] > 0.001][:5],
                    "top_negative_factors": [contrib for contrib in feature_contributions 
                                           if contrib["shap_value"] < -0.001][:5]
                }
                
                print("✅ SHAP explanation generated successfully")
                
            except Exception as e:
                print(f"⚠️  Warning: SHAP explanation failed: {e}")
                # Fallback to feature importance
                explanations = self._get_feature_explanations(df_input, transaction)
        
        # Fallback to feature importance if SHAP failed or not available
        if not explanations:
            explanations = self._get_feature_explanations(df_input, transaction)
            
        return explanations, shap_explanation

    def _get_feature_explanations(self, df_input: pd.DataFrame, transaction: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Extract meaningful feature explanations for the prediction."""
        explanations = []
        
        # Use feature importances if available
        if self.clf is not None and hasattr(self.clf, "feature_importances_") and self.feature_names:
            importances = np.array(self.clf.feature_importances_, dtype=float)
            if len(importances) == len(self.feature_names):
                # Get top important features
                idx = np.argsort(importances)[::-1]
                for i in idx[:10]:
                    fname = self.feature_names[i]
                    importance = float(importances[i])
                    explanations.append((fname, importance))
        
        # Fallback: use heuristic explanations based on transaction values
        if not explanations:
            amt = transaction.get("amount", 0)
            avg7 = transaction.get("avg_transaction_amount_7d", 1)
            
            explanations = [
                ("amount_vs_avg_ratio", float(amt) / (float(avg7) + 1e-9)),
                ("device_change_rate_30d", float(transaction.get("device_change_rate_30d", 0))),
                ("account_age_days", float(transaction.get("account_age_days", 0))),
                ("is_foreign", float(transaction.get("is_foreign", 0))),
                ("is_blacklisted_merchant", float(transaction.get("is_blacklisted_merchant", 0))),
                ("num_prev_transactions_24h", float(transaction.get("num_prev_transactions_24h", 0))),
                ("amount", float(amt))
            ]
            explanations = sorted(explanations, key=lambda x: abs(x[1]), reverse=True)
        
        return explanations

    def _generate_recommendations(self, score: float, transaction: Dict[str, Any], reasons: List[Tuple[str, float]]) -> List[str]:
        """Generate actionable recommendations based on the fraud score and reasons."""
        recommendations = []
        
        if score >= 0.8:
            recommendations.append("🚨 BLOCK TRANSACTION - High fraud probability")
            recommendations.append("📞 Contact customer immediately for verification")
        elif score >= 0.6:
            recommendations.append("⚠️ REVIEW REQUIRED - Elevated fraud risk")
            recommendations.append("🔍 Perform additional verification checks")
        elif score >= 0.4:
            recommendations.append("👀 MONITOR - Medium risk transaction")
        else:
            recommendations.append("✅ APPROVE - Low fraud risk")
        
        # Specific recommendations based on top reasons
        top_reason = reasons[0][0] if reasons else ""
        
        if "device_change" in top_reason:
            recommendations.append("📱 Verify device and location with customer")
        if "amount" in top_reason:
            recommendations.append("💰 Verify transaction amount with customer")
        if "foreign" in top_reason:
            recommendations.append("🌍 Verify international transaction legitimacy")
        if "blacklisted" in top_reason:
            recommendations.append("🚫 Review merchant blacklist status")
        if "account_age" in top_reason:
            recommendations.append("👤 Enhanced verification for new account")
        
        return recommendations

    def get_shap_summary(self, transactions: List[Dict[str, Any]], max_samples: int = 100) -> Optional[Dict[str, Any]]:
        """
        Generate SHAP summary for multiple transactions.
        
        Args:
            transactions: List of transaction dictionaries
            max_samples: Maximum number of samples to analyze
            
        Returns:
            SHAP summary with feature importance and interaction effects
        """
        if not self.enable_shap or self.shap_explainer is None:
            return None
            
        try:
            print(f"🔍 Generating SHAP summary for {min(len(transactions), max_samples)} transactions...")
            
            # Limit samples for performance
            sample_transactions = transactions[:max_samples]
            
            # Process all transactions
            processed_data = []
            for tx in sample_transactions:
                df_input = self._normalize_transaction(tx)
                X_processed = self.pipeline.named_steps['preprocessor'].transform(df_input)
                processed_data.append(X_processed[0])
            
            X_batch = np.array(processed_data)
            
            # Calculate SHAP values for batch
            shap_values = self.shap_explainer.shap_values(X_batch)
            
            # Handle different output formats
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values_fraud = shap_values[1]  # Fraud class
            else:
                shap_values_fraud = shap_values
            
            # Calculate feature importance summary
            mean_abs_shap = np.mean(np.abs(shap_values_fraud), axis=0)
            feature_importance = []
            
            if self.feature_names and len(mean_abs_shap) == len(self.feature_names):
                for feature, importance in zip(self.feature_names, mean_abs_shap):
                    feature_importance.append({
                        "feature": feature,
                        "mean_abs_shap": float(importance),
                        "mean_shap": float(np.mean(shap_values_fraud[:, self.feature_names.index(feature)]))
                    })
            
            feature_importance = sorted(feature_importance, key=lambda x: x["mean_abs_shap"], reverse=True)
            
            summary = {
                "total_samples_analyzed": len(sample_transactions),
                "feature_importance_ranking": feature_importance[:15],  # Top 15 features
                "shap_statistics": {
                    "mean_prediction": float(np.mean(np.sum(shap_values_fraud, axis=1))),
                    "std_prediction": float(np.std(np.sum(shap_values_fraud, axis=1))),
                    "max_feature_impact": float(np.max(np.abs(shap_values_fraud))),
                    "most_impactful_feature": feature_importance[0]["feature"] if feature_importance else "unknown"
                },
                "top_fraud_drivers": [f for f in feature_importance if f["mean_shap"] > 0.01][:5],
                "top_fraud_preventers": [f for f in feature_importance if f["mean_shap"] < -0.01][:5]
            }
            
            print("✅ SHAP summary generated successfully")
            return summary
            
        except Exception as e:
            print(f"❌ Error generating SHAP summary: {e}")
            return None


# convenience factory
_detector: FraudDetector = None


def get_default_detector() -> FraudDetector:
    global _detector
    if _detector is None:
        _detector = FraudDetector()
    return _detector


# OpenAI SDK format function tool
@function_tool
async def detect_fraud(transaction: Transaction) -> FraudPrediction:
    """Detect fraud in a transaction using machine learning with SHAP explanations.

    Args:
        transaction: The transaction data to analyze for fraud indicators.
        
    Returns:
        FraudPrediction containing score, risk level, explanations, and recommendations.
    """
    detector = get_default_detector()
    result = detector.predict(transaction)
    
    # Convert to the expected TypedDict format
    return FraudPrediction(
        score=result["score"],
        label=result["label"],
        risk_level=result["risk_level"],
        threshold=result["threshold"],
        confidence=result["confidence"],
        top_reasons=result["top_reasons"],
        recommendations=result["recommendations"],
        model_details=result["model_details"]
    )


# CLI quick test
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test the fraud detector with SHAP explanations")
    parser.add_argument("--sample_idx", type=int, default=0, help="Sample row index to test")
    parser.add_argument("--threshold", type=float, default=0.5, help="Fraud detection threshold")
    parser.add_argument("--enable-shap", action="store_true", default=True, help="Enable SHAP explanations")
    parser.add_argument("--shap-summary", action="store_true", help="Generate SHAP summary for multiple transactions")
    args = parser.parse_args()

    # Enhanced sample transactions for testing
    sample_transactions = [
        {
            "transaction_id": "TXN_demo_1",
            "customer_id": "CUST_test",
            "amount": 12000,
            "currency": "NGN",
            "transaction_type": "transfer",
            "channel": "web",
            "merchant_category": "electronics",
            "device_id": "DEV_test",
            "ip_address": "192.168.1.100",
            "location_country": "Nigeria",
            "location_city": "Lagos",
            "time": "2025-10-22 14:30:00",
            "account_age_days": 45,
            "num_prev_transactions_24h": 8,
            "avg_transaction_amount_7d": 500,
            "device_change_rate_30d": 0.8,
            "is_foreign": 0,
            "is_blacklisted_merchant": 0
        },
        {
            "transaction_id": "TXN_demo_2", 
            "customer_id": "CUST_normal",
            "amount": 850,
            "currency": "NGN",
            "transaction_type": "purchase",
            "channel": "mobile",
            "merchant_category": "groceries",
            "device_id": "DEV_regular",
            "ip_address": "192.168.1.200",
            "location_country": "Nigeria",
            "location_city": "Abuja",
            "time": "2025-10-22 10:15:00",
            "account_age_days": 730,
            "num_prev_transactions_24h": 2,
            "avg_transaction_amount_7d": 900,
            "device_change_rate_30d": 0.1,
            "is_foreign": 0,
            "is_blacklisted_merchant": 0
        },
        {
            "transaction_id": "TXN_demo_3",
            "customer_id": "CUST_suspicious",
            "amount": 25000,
            "currency": "NGN",
            "transaction_type": "transfer",
            "channel": "web",
            "merchant_category": "luxury",
            "device_id": "DEV_new",
            "ip_address": "192.168.1.300",
            "location_country": "USA",
            "location_city": "Lagos",
            "time": "2025-10-22 02:00:00",
            "account_age_days": 5,
            "num_prev_transactions_24h": 15,
            "avg_transaction_amount_7d": 200,
            "device_change_rate_30d": 0.9,
            "is_foreign": 1,
            "is_blacklisted_merchant": 1
        }
    ]
    
    async def test_fraud_detector():
        try:
            detector = get_default_detector()
            
            if args.shap_summary:
                print("🔍 Generating SHAP Summary Analysis")
                print("=" * 60)
                summary = detector.get_shap_summary(sample_transactions)
                if summary:
                    print(json.dumps(summary, indent=2))
                else:
                    print("❌ SHAP summary not available")
            else:
                sample_idx = min(args.sample_idx, len(sample_transactions) - 1)
                
                print(f"🧪 Testing Fraud Detector with SHAP Explanations - Sample {sample_idx + 1}")
                print("=" * 70)
                
                sample_tx = sample_transactions[sample_idx]
                # Convert to TypedDict format for the new function
                tx_dict = Transaction(**sample_tx)
                result = await detect_fraud(tx_dict)
                
                print(f"💰 Transaction: {sample_tx['transaction_type']} of {sample_tx['amount']} {sample_tx['currency']}")
                print(f"🎯 Fraud Score: {result['score']:.4f}")
                print(f"🏷️  Label: {'FRAUD' if result['label'] == 1 else 'LEGITIMATE'}")
                print(f"📊 Risk Level: {result['risk_level']}")
                print(f"🎚️  Confidence: {result['confidence']:.4f}")
                
                print(f"\n🔍 Top Risk Factors:")
                for i, (feature, value) in enumerate(result['top_reasons'][:5], 1):
                    print(f"   {i}. {feature}: {value:.4f}")
                
                # Get the full result with SHAP explanation from the detector
                full_result = detector.predict(sample_tx)
                if full_result.get('shap_explanation'):
                    shap_exp = full_result['shap_explanation']
                    print(f"\n🧠 SHAP Explanation:")
                    print(f"   Base Value: {shap_exp['base_value']:.4f}")
                    print(f"   Prediction: {shap_exp['prediction']:.4f}")
                    
                    print(f"\n   Top Fraud Drivers (Positive SHAP):")
                    for i, contrib in enumerate(shap_exp['top_positive_factors'][:3], 1):
                        print(f"   {i}. {contrib['feature']}: +{contrib['shap_value']:.4f}")
                    
                    print(f"\n   Top Fraud Preventers (Negative SHAP):")
                    for i, contrib in enumerate(shap_exp['top_negative_factors'][:3], 1):
                        print(f"   {i}. {contrib['feature']}: {contrib['shap_value']:.4f}")
                
                print(f"\n💡 Recommendations:")
                for i, rec in enumerate(result['recommendations'], 1):
                    print(f"   {i}. {rec}")
                
                print("\n" + "=" * 70)
            
        except Exception as e:
            print(f"❌ Error testing fraud detector: {e}")
            print(f"💡 Make sure the model has been trained and SHAP is installed:")
            print("   pip install shap")
            print("   python src/model_baseline.py")
    
    # Run the async test
    import asyncio
    asyncio.run(test_fraud_detector())