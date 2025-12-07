"""Random Forest Classifier for IVF Response Prediction

This module trains a Random Forest model to predict patient response to
ovarian stimulation in IVF treatments. It includes:
- 5-fold stratified cross-validation
- Feature importance analysis
- SHAP explainability
- Model persistence

Author: [Your Name]
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import sys
import os
import matplotlib.pyplot as plt
import shap  

# Add src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import MODEL_FILE, PLOTS_DIR
from preprocessing.data_preprocessing import preprocess_and_update_dataset

# === DATA LOADING ===
# Load preprocessed data (returns two datasets: multi-cycle and independent)
df1, df2 = preprocess_and_update_dataset()

# === FEATURE AND TARGET SEPARATION ===
# X: Features (all columns except patient_id and target)
# y: Target variable (Patient Response: 0=low, 1=optimal, 2=high)
X = df2.drop(['patient_id', 'Patient Response'], axis=1)
y = df2['Patient Response']

# === TRAIN/TEST SPLIT ===
# Stratified split maintains class distribution in both sets
# This is crucial for imbalanced medical datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 80% train, 20% test
    random_state=42,    # Reproducibility
    stratify=y          # Maintain class proportions
)

# === MODEL INITIALIZATION ===
# Random Forest chosen for:
# - Handles non-linear relationships
# - Built-in feature importance
# - Robust to outliers
# - No feature scaling required
model = RandomForestClassifier(
    n_estimators=100,           # 100 decision trees
    max_depth=10,               # Prevent overfitting
    class_weight='balanced',    # Handle class imbalance
    random_state=42             # Reproducibility
)

# === CROSS-VALIDATION ===
# 5-fold stratified CV provides robust performance estimate
print("\n=== Cross-Validation (5-Fold) ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"Individual folds: {[f'{score:.3f}' for score in cv_scores]}")

# Train final model on full training set
model.fit(X_train, y_train)

# Evaluate on test set
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\n=== Final Test Set Evaluation ===")
print(f"Test Accuracy: {test_accuracy:.3f}")
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Low', 'Optimal', 'High']))

# Probabilistic Output Example
print("\n=== Sample Prediction (Probabilistic) ===")
sample_patient_df = pd.DataFrame(X_test.iloc[0].values.reshape(1, -1), columns=X.columns)
probs = model.predict_proba(sample_patient_df)
print(f"Patient {X_test.index[0]}:")
print(f"  Low Response:     {probs[0][0]*100:.1f}%")
print(f"  Optimal Response: {probs[0][1]*100:.1f}%")
print(f"  High Response:    {probs[0][2]*100:.1f}%")

# 8. SHAP Explainability (MANDATORY for Assignment)
print("\n--- Generating SHAP Explanations ---")
# Create an explainer object
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary Plot (Global Importance) - Shows which features matter most
plt.figure()
shap.summary_plot(shap_values, X_test, class_names=['Low', 'Optimal', 'High'], show=False)
shap_summary_path = os.path.join(PLOTS_DIR, "shap_summary_plot.png")
plt.savefig(shap_summary_path, bbox_inches='tight')
print(f"Saved SHAP summary plot to '{shap_summary_path}'")

# Feature Importance Analysis
print("\n=== Feature Importance Analysis ===")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.to_string(index=False))

# Save feature importance
feature_importance_path = os.path.join(PLOTS_DIR, "feature_importance.csv")
feature_importance.to_csv(feature_importance_path, index=False)
print(f"\nFeature importance saved to {feature_importance_path}")

# Visualize feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
feature_plot_path = os.path.join(PLOTS_DIR, "feature_importance.png")
plt.savefig(feature_plot_path, bbox_inches='tight')
print(f"Feature importance plot saved to {feature_plot_path}")

# Save the model
joblib.dump(model, MODEL_FILE)
print(f"\nModel saved to {MODEL_FILE}")

plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Summary")
shap_beeswarm_path = os.path.join(PLOTS_DIR, "shap_beeswarm_high.png")
plt.savefig(shap_beeswarm_path, bbox_inches='tight')
print(f"SHAP plot saved to {shap_beeswarm_path}")