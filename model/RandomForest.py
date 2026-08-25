"""
RandomForest.py
Purpose: Runs Random Forest directly with the best hyperparameters found previously.
Evaluates the model and exports the .pkl files for GUI integration.
"""

import os
import time
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)

# 1. IMPORT EVERYTHING FROM YOUR PREPROCESSING FILE
from preprocessing.preprocessing import (
    X_train, 
    X_test, 
    y_train, 
    y_test, 
    preprocessor,
    binary_cols, 
    ordinal_cols, 
    nominal_cols, 
    numeric_cols
)

start_time = time.time()
RANDOM_STATE = 42

# --------------------------------------------------------------------------
# 2. BEST HYPERPARAMETERS
# --------------------------------------------------------------------------
BEST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 20,
    "max_features": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "criterion": "entropy"
}

# --------------------------------------------------------------------------
# SEVERITY-ORDERED TARGET ENCODER
# --------------------------------------------------------------------------
SEVERITY_ORDER = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]

class SeverityOrderedTargetEncoder:
    """Encodes the target in explicit obesity-severity order."""
    def __init__(self, categories):
        self.classes_ = np.array(categories)
        self._to_code = {c: i for i, c in enumerate(categories)}
        self._to_label = {i: c for i, c in enumerate(categories)}

    def fit_transform(self, y):
        return np.array([self._to_code[v] for v in y])

    def transform(self, y):
        return np.array([self._to_code[v] for v in y])

    def inverse_transform(self, y):
        return np.array([self._to_label[int(i)] for i in y])

# Initialize the encoder for the graphs and .pkl export
target_encoder = SeverityOrderedTargetEncoder(SEVERITY_ORDER)

# --------------------------------------------------------------------------
# 3. FIT RANDOM FOREST WITH BEST HYPERPARAMETERS
# --------------------------------------------------------------------------
best_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
                **BEST_PARAMS
            ),
        ),
    ]
)

print("\nFitting final Random Forest model...")
best_model.fit(X_train, y_train)

# --------------------------------------------------------------------------
# 4. EVALUATION
# --------------------------------------------------------------------------
final_preds = best_model.predict(X_test)
final_probs = best_model.predict_proba(X_test)

final_acc = accuracy_score(y_test, final_preds)
weighted_roc_auc = roc_auc_score(y_test, final_probs, multi_class="ovr", average="weighted")

print(f"\nTest accuracy: {final_acc:.4f}")
print(f"Weighted-average ROC-AUC (OvR): {weighted_roc_auc:.4f}")
print("\nClassification report:\n")
print(classification_report(y_test, final_preds, target_names=target_encoder.classes_, digits=4))

# Ensure output directories exist
os.makedirs("results/graphs", exist_ok=True)
os.makedirs("saved_model", exist_ok=True)

# Generate Confusion Matrix
cm = confusion_matrix(y_test, final_preds)
fig, ax = plt.subplots(figsize=(9, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_encoder.classes_)
disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix — Random Forest (Best Params)")
plt.tight_layout()
plt.savefig("results/graphs/rf_confusion_matrix.png", dpi=150)
plt.close()

# ==========================================================================
# 4.5. SAMPLE DECISION TREE GRAPH
# ==========================================================================
print("\nGenerating sample decision tree graph...")

# 1. Recover feature names from the new ColumnTransformer structure
# Binary and Ordinal columns keep their original names
bin_feature_names = binary_cols
ord_feature_names = ordinal_cols

# Nominal columns get expanded by OneHotEncoder
nom_ohe = best_model.named_steps["preprocessor"].named_transformers_["nom"]
nom_feature_names = list(nom_ohe.get_feature_names_out(nominal_cols))

# Combine all names in the EXACT order they appear in the ColumnTransformer
all_feature_names = bin_feature_names + ord_feature_names + nom_feature_names + numeric_cols

# 2. Extract the very first decision tree (index 0) from the 300 tuned trees
sample_tree = best_model.named_steps["classifier"].estimators_[0]

# 3. Plot the tree
plt.figure(figsize=(25, 12))
plot_tree(
    sample_tree,
    feature_names=all_feature_names,
    class_names=list(target_encoder.classes_),
    filled=True,
    rounded=True,
    max_depth=3,  # Capped at 3 so the image is readable in a Word document
    fontsize=9
)

plt.title("Sample Decision Tree from Tuned Random Forest (Max Depth = 3)")
plt.tight_layout()

tree_graph_path = "results/graphs/rf_sample_tree.png"
plt.savefig(tree_graph_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved sample decision tree to {tree_graph_path}")

# --------------------------------------------------------------------------
# 5. SAVE THE FINAL MODEL FOR GUI
# --------------------------------------------------------------------------
joblib.dump(best_model, "saved_model/random_forest_model.pkl")
joblib.dump(target_encoder, "saved_model/rf_target_encoder.pkl")
print("\nSaved trained pipeline to saved_model/random_forest_model.pkl")
print("Saved target label encoder to saved_model/rf_target_encoder.pkl")

# ==========================================================================
# 6. RUNNING TIME
# ==========================================================================
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTotal Execution Time: {elapsed_time:.2f} seconds")