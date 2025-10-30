# src/model_baseline.py
"""
Baseline model training script for fraud detection.

- Loads raw data via src.preprocess
- Splits into train/test (stratified)
- Constructs a pipeline: preprocessor -> LightGBM classifier
- Trains and evaluates model (ROC AUC, precision, recall, confusion matrix)
- Saves trained pipeline to artifacts/lgbm_model.joblib and evaluation report
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import lightgbm as lgb

# Add the parent directory to the path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import load_data, basic_feature_engineering, prepare_data, train_test_split_with_stratify, PREPROCESSOR_PATH, ARTIFACT_DIR

ARTIFACT_MODEL_PATH = os.path.join(ARTIFACT_DIR, "lgbm_model.joblib")
EVAL_REPORT_PATH = os.path.join(ARTIFACT_DIR, "eval_report.json")


def build_model_pipeline(preprocessor, lgb_params=None):
    """Build an optimized LightGBM pipeline for fraud detection."""
    if lgb_params is None:
        # Optimized parameters for fraud detection (imbalanced dataset)
        lgb_params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "n_jobs": -1,
            "class_weight": "balanced",  # Handle imbalanced classes
            "verbose": -1
        }

    lgbm_clf = lgb.LGBMClassifier(**lgb_params)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", lgbm_clf)
    ])
    return pipeline


def evaluate_model(pipeline, X_test: pd.DataFrame, y_test: pd.Series, threshold=0.5):
    """Enhanced model evaluation with multiple thresholds and detailed metrics."""
    preds_proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (preds_proba >= threshold).astype(int)

    # Core metrics
    roc = roc_auc_score(y_test, preds_proba)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds).tolist()
    report = classification_report(y_test, preds, zero_division=0, output_dict=True)

    # Additional fraud-specific metrics
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    fraud_detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Calculate metrics at different thresholds for optimal threshold selection
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    threshold_metrics = {}
    
    for t in thresholds:
        t_preds = (preds_proba >= t).astype(int)
        t_prec = precision_score(y_test, t_preds, zero_division=0)
        t_rec = recall_score(y_test, t_preds, zero_division=0)
        t_f1 = f1_score(y_test, t_preds, zero_division=0)
        threshold_metrics[f"threshold_{t}"] = {
            "precision": float(t_prec),
            "recall": float(t_rec),
            "f1": float(t_f1)
        }

    metrics = {
        "primary_metrics": {
            "roc_auc": float(roc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "fraud_detection_rate": float(fraud_detection_rate),
            "false_positive_rate": float(false_positive_rate)
        },
        "confusion_matrix": {
            "matrix": cm,
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "threshold_analysis": threshold_metrics,
        "classification_report": report,
        "model_threshold": float(threshold)
    }
    return metrics


def get_feature_importance(pipeline, feature_names, top_n=20):
    """Extract and format feature importance from the trained model."""
    try:
        # Get feature importance from LightGBM
        importance = pipeline.named_steps['clf'].feature_importances_
        
        # Create importance dataframe
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance.head(top_n).to_dict('records')
    except Exception as e:
        print(f"Warning: Could not extract feature importance: {e}")
        return []


def print_evaluation_summary(metrics):
    """Print a beautiful summary of model performance."""
    print("\n" + "="*60)
    print("🎯 FRAUD DETECTION MODEL EVALUATION SUMMARY")
    print("="*60)
    
    primary = metrics['primary_metrics']
    cm = metrics['confusion_matrix']
    
    print(f"📊 ROC AUC Score:           {primary['roc_auc']:.4f}")
    print(f"🎯 Precision:               {primary['precision']:.4f}")
    print(f"📈 Recall:                  {primary['recall']:.4f}")
    print(f"⚖️  F1 Score:                {primary['f1']:.4f}")
    print(f"🚨 Fraud Detection Rate:    {primary['fraud_detection_rate']:.4f}")
    print(f"⚠️  False Positive Rate:     {primary['false_positive_rate']:.4f}")
    
    print(f"\n📋 Confusion Matrix:")
    print(f"   True Negatives:  {cm['true_negatives']:,}")
    print(f"   False Positives: {cm['false_positives']:,}")
    print(f"   False Negatives: {cm['false_negatives']:,}")
    print(f"   True Positives:  {cm['true_positives']:,}")
    
    print(f"\n🎚️  Threshold Analysis:")
    for thresh_name, thresh_metrics in metrics['threshold_analysis'].items():
        threshold = thresh_name.split('_')[1]
        print(f"   @ {threshold}: Precision={thresh_metrics['precision']:.3f}, Recall={thresh_metrics['recall']:.3f}, F1={thresh_metrics['f1']:.3f}")
    print("="*60)


def main(test_size: float = 0.2):
    """Main training and evaluation pipeline."""
    print("🚀 Starting Fraud Detection Model Training...")
    
    # Load raw dataframe
    df = load_data()
    print(f"📊 Loaded dataset with {len(df)} transactions")
    
    # Check class distribution
    fraud_rate = df["fraud_label"].mean()
    print(f"🚨 Fraud rate in dataset: {fraud_rate:.2%}")

    # Build train/test split using stratification  
    X_train_raw, X_test_raw, y_train, y_test = train_test_split_with_stratify(
        df, target_col="fraud_label", test_size=test_size
    )
    print(f"📈 Train size: {len(y_train):,}, Test size: {len(y_test):,}")
    print(f"🎯 Train fraud rate: {y_train.mean():.2%}, Test fraud rate: {y_test.mean():.2%}")

    # Prepare data and fit preprocessor on training data only (avoid data leakage)
    df_train = X_train_raw.copy()
    df_train["fraud_label"] = y_train.values
    X_train_processed, y_train_processed, preprocessor, feature_names = prepare_data(
        df_train, target_col="fraud_label", save_preprocessor=True
    )
    print(f"🔧 Preprocessor fitted and saved with {len(feature_names)} features")

    # Build optimized model pipeline
    pipeline = build_model_pipeline(preprocessor)
    print("🏗️  Built LightGBM pipeline with balanced class weights")

    # Get the feature columns that the preprocessor expects
    expected_features = []
    for feature_list in [preprocessor.transformers[0][2], preprocessor.transformers[1][2]]:
        expected_features.extend(feature_list)
    
    # Train the model
    print("🎯 Training model...")
    pipeline.fit(X_train_raw[expected_features], y_train)
    print("✅ Model training completed!")

    # Evaluate on test set
    print("📊 Evaluating model on test set...")
    metrics = evaluate_model(pipeline, X_test_raw[expected_features], y_test)
    
    # Print beautiful summary
    print_evaluation_summary(metrics)
    
    # Get feature importance
    feature_importance = get_feature_importance(pipeline, feature_names, top_n=15)
    if feature_importance:
        print("\n🏆 TOP 15 MOST IMPORTANT FEATURES:")
        print("-" * 50)
        for i, feat in enumerate(feature_importance, 1):
            print(f"{i:2d}. {feat['feature']:<30} {feat['importance']:.4f}")

    # Save model pipeline and metrics
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(pipeline, ARTIFACT_MODEL_PATH)
    
    # Enhanced metrics with feature importance
    enhanced_metrics = metrics.copy()
    enhanced_metrics['feature_importance'] = feature_importance
    enhanced_metrics['dataset_info'] = {
        'total_samples': len(df),
        'train_samples': len(y_train),
        'test_samples': len(y_test),
        'fraud_rate': float(fraud_rate),
        'features_count': len(feature_names)
    }
    
    with open(EVAL_REPORT_PATH, "w") as f:
        json.dump(enhanced_metrics, f, indent=2)

    print(f"\n💾 Model pipeline saved to: {ARTIFACT_MODEL_PATH}")
    print(f"📄 Evaluation report saved to: {EVAL_REPORT_PATH}")
    print("🎉 Training pipeline completed successfully!")
    
    return pipeline, metrics


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()