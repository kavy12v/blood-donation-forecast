"""
Blood Donation Forecast - Machine Learning Project
Slash Mark Machine Learning Internship - Task 4

This script performs end-to-end data processing, exploratory data analysis,
feature engineering (log-normalization), model training, benchmarking,
and evaluation for predicting blood donor return.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for generating plots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def run_pipeline():
    print("=" * 70)
    print("🩸 BLOOD DONATION FORECAST - MACHINE LEARNING PIPELINE 🩸")
    print("=" * 70)

    # Create plots directory if it does not exist
    os.makedirs("plots", exist_ok=True)

    # 1. Load Data
    data_path = os.path.join("datasets", "transfusion.data")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    print(f"\n[Step 1] Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Rename Target Column
    target_orig_name = "whether he/she donated blood in March 2007"
    df.rename(columns={target_orig_name: "target"}, inplace=True)
    print("\n[Step 2] Renamed target column to 'target'")
    print(df.head())

    # 3. Target Incidence (Class Distribution)
    incidence = df["target"].value_counts(normalize=True)
    print("\n[Step 3] Target Incidence:")
    print(f"  Class 0 (Did not donate): {incidence[0]*100:.2f}% ({df['target'].value_counts()[0]} donors)")
    print(f"  Class 1 (Donated blood) : {incidence[1]*100:.2f}% ({df['target'].value_counts()[1]} donors)")

    # Plot 1: Target Incidence
    plt.figure(figsize=(6, 4))
    sns.countplot(x="target", data=df, hue="target", palette=["#3498db", "#e74c3c"], legend=False)
    plt.title("Target Class Distribution (0: No Donation, 1: Donated)", fontsize=12)
    plt.xlabel("Target (Donated in March 2007)")
    plt.ylabel("Number of Donors")
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "01_target_incidence.png"), dpi=300)
    plt.close()

    # Plot 2: Correlation Heatmap
    plt.figure(figsize=(7, 5))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Feature Correlation Matrix", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "02_feature_correlations.png"), dpi=300)
    plt.close()

    # 4. Train-Test Split (Stratified to maintain class ratio)
    X = df.drop(columns="target")
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"\n[Step 4] Stratified Split: Train set = {len(X_train)} samples, Test set = {len(X_test)} samples")

    # 5. Check Feature Variance & Log Normalization
    print("\n[Step 5] Checking Feature Variances before normalization:")
    orig_var = X_train.var().round(2)
    for col, var in orig_var.items():
        print(f"  - {col:25s}: {var:>12.2f}")

    # Prepare Log-Normalized Data
    col_to_normalize = "Monetary (c.c. blood)"
    X_train_normed = X_train.copy()
    X_test_normed = X_test.copy()

    for d in [X_train_normed, X_test_normed]:
        d["monetary_log"] = np.log(d[col_to_normalize])
        d.drop(columns=col_to_normalize, inplace=True)

    print("\nChecking Feature Variances after log-normalization:")
    norm_var = X_train_normed.var().round(2)
    for col, var in norm_var.items():
        print(f"  - {col:25s}: {var:>12.2f}")

    # Plot 3: Variance Comparison
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Variance Before Log-Transform")
    plt.bar(range(len(orig_var)), orig_var.values, color="#e67e22")
    plt.xticks(range(len(orig_var)), orig_var.index, rotation=45, ha='right')
    plt.yscale("log")
    plt.ylabel("Variance (log scale)")

    plt.subplot(1, 2, 2)
    plt.title("Variance After Log-Transform")
    plt.bar(range(len(norm_var)), norm_var.values, color="#2ecc71")
    plt.xticks(range(len(norm_var)), norm_var.index, rotation=45, ha='right')
    plt.yscale("log")
    plt.ylabel("Variance (log scale)")
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "03_variance_log_transformation.png"), dpi=300)
    plt.close()

    # 6. Train Models and Benchmark
    print("\n[Step 6] Training Machine Learning Models...")

    models = {
        "Baseline Logistic Regression": (
            LogisticRegression(solver="liblinear", random_state=42),
            X_train,
            X_test
        ),
        "Normalized Logistic Regression": (
            LogisticRegression(solver="liblinear", random_state=42),
            X_train_normed,
            X_test_normed
        ),
        "Random Forest Classifier": (
            RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42),
            X_train_normed,
            X_test_normed
        ),
        "XGBoost Classifier": (
            XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.08, eval_metric="logloss", random_state=42),
            X_train_normed,
            X_test_normed
        )
    }

    results = []
    roc_curves = {}

    plt.figure(figsize=(8, 6))

    for name, (model, xtr, xte) in models.items():
        model.fit(xtr, y_train)
        y_pred = model.predict(xte)
        y_proba = model.predict_proba(xte)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results.append({
            "Model": name,
            "ROC AUC": auc,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        })

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curves[name] = (fpr, tpr, auc)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})")

    # Plot 4: Combined ROC Curves
    plt.plot([0, 1], [0, 1], 'k--', label="Random Chance (AUC = 0.5000)")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)")
    plt.title("ROC Curves Comparison for Blood Donation Forecast", fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "04_roc_auc_curves.png"), dpi=300)
    plt.close()

    # Results Table
    results_df = pd.DataFrame(results).sort_values(by="ROC AUC", ascending=False)
    print("\n" + "=" * 70)
    print("🏆 MODEL PERFORMANCE BENCHMARK 🏆")
    print("=" * 70)
    print(results_df.to_string(index=False))

    # Best model selection
    best_model_name = results_df.iloc[0]["Model"]
    print(f"\nTop Performing Model: {best_model_name}")

    # Plot 5: Confusion Matrix for Normalized Logistic Regression
    best_logreg = models["Normalized Logistic Regression"][0]
    y_pred_best = best_logreg.predict(X_test_normed)
    cm = confusion_matrix(y_test, y_pred_best)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred: No", "Pred: Donated"],
                yticklabels=["Actual: No", "Actual: Donated"])
    plt.title("Confusion Matrix (Normalized Logistic Regression)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "05_confusion_matrix.png"), dpi=300)
    plt.close()

    # Plot 6: Feature Coefficients for Logistic Regression
    feature_names = X_train_normed.columns
    coeffs = best_logreg.coef_[0]
    plt.figure(figsize=(6, 4))
    colors = ['#e74c3c' if c < 0 else '#2ecc71' for c in coeffs]
    plt.barh(feature_names, coeffs, color=colors)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title("Feature Importance / Coefficients (Normalized LogReg)", fontsize=11)
    plt.xlabel("Coefficient Value")
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "06_feature_importance.png"), dpi=300)
    plt.close()

    # 7. Classification Report
    print("\nDetailed Classification Report (Normalized Logistic Regression):")
    print(classification_report(y_test, y_pred_best, target_names=["No Donation (0)", "Donated (1)"]))

    # 8. Demonstration Donor Prediction
    print("=" * 70)
    print("🔮 INTERACTIVE PREDICTION EXAMPLE")
    print("=" * 70)
    sample_donors = pd.DataFrame([
        {"Recency (months)": 2, "Frequency (times)": 10, "Time (months)": 28, "monetary_log": np.log(2500)},
        {"Recency (months)": 16, "Frequency (times)": 2, "Time (months)": 16, "monetary_log": np.log(500)},
        {"Recency (months)": 1, "Frequency (times)": 24, "Time (months)": 77, "monetary_log": np.log(6000)}
    ])
    probabilities = best_logreg.predict_proba(sample_donors)[:, 1]
    predictions = best_logreg.predict(sample_donors)

    for i, (prob, pred) in enumerate(zip(probabilities, predictions), 1):
        status = "LIKELY TO DONATE (1)" if pred == 1 else "UNLIKELY TO DONATE (0)"
        print(f"Donor {i}: Predicted Probability = {prob*100:.1f}% -> Status: {status}")

    print("\n[Complete] Pipeline executed successfully! All visual plots saved in 'plots/' directory.")

if __name__ == "__main__":
    run_pipeline()
