"""
RandomForest_Tuning.py
Purpose: Hyperparameter tuning using RandomizedSearchCV, generating tuning graphs, 
and saving both the best parameters and all individual parameter results to a text file.
"""

import os
import sys
import contextlib
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

# ============================================================
# PROGRESS BAR HELPER
# ============================================================
import joblib
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

# ============================================================
# PROJECT PATHS & IMPORTS
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from preprocessing.preprocessing import preprocess_data

GRAPH_PATH = os.path.join(PROJECT_ROOT, "results", "graphs")
RESULT_FILE = os.path.join(PROJECT_ROOT, "results", "random_forest_tuning_results.txt")
os.makedirs(GRAPH_PATH, exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("Loading and preprocessing data for tuning...")
(X_train, X_test, y_train, y_test, preprocessor) = preprocess_data()

# ============================================================
# 2. SETUP PIPELINE & TUNING PARAMETERS
# ============================================================
rf_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1))
])

param_distributions = {
    "classifier__n_estimators": [100, 200, 300, 500, 700, 1000],
    "classifier__max_depth": [10, 20, 30, 50, 75, 100, None],
    "classifier__max_features": ["sqrt", "log2", None],
    "classifier__criterion": ["gini", "entropy", "log_loss"],
    "classifier__min_samples_split": [2, 5, 10, 15],
    "classifier__min_samples_leaf": [1, 2, 4, 8]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=param_distributions,
    n_iter=100,
    scoring="accuracy",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=0, # Set to 0 so tqdm progress bar displays cleanly
    return_train_score=True
)

# ============================================================
# 3. RUN TUNING WITH PROGRESS BAR
# ============================================================
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

print("\nStarting hyperparameter tuning...")
total_fits = random_search.n_iter * random_search.cv.n_splits

with tqdm_joblib(tqdm(desc="Tuning Progress", total=total_fits)):
    random_search.fit(X_train, y_train)

# ============================================================
# 4. SAVE BEST RESULTS AND ALL PARAMETER RESULTS
# ============================================================
print("\nSaving best parameters and all parameter results to text file...")
cv_results = pd.DataFrame(random_search.cv_results_)

parameters_to_plot = {
    "classifier__n_estimators": "Number of Trees",
    "classifier__max_depth": "Max Depth",
    "classifier__max_features": "Max Features",
    "classifier__min_samples_split": "Min Samples Split",
    "classifier__min_samples_leaf": "Min Samples Leaf",
    "classifier__criterion": "Splitting Criterion"
}

with open(RESULT_FILE, "w") as file:
    file.write("RANDOM FOREST TUNING RESULTS\n")
    file.write("=" * 50 + "\n\n")
    
    # --- PART 1: FINAL TUNING RESULT (BEST) ---
    file.write("1. FINAL TUNING RESULT (BEST COMBINATION)\n")
    file.write("-" * 50 + "\n")
    file.write(f"Best CV Accuracy: {random_search.best_score_:.4f}\n\n")
    file.write("Best Parameters:\n")
    for param, value in random_search.best_params_.items():
        file.write(f"  {param}: {value}\n")
    file.write("\n\n")
    
    # --- PART 2: ALL PARAMETER RESULTS ---
    file.write("2. ALL PARAMETER RESULTS (AVERAGE CV ACCURACY)\n")
    file.write("-" * 50 + "\n")

    for param_col, param_title in parameters_to_plot.items():
        search_col = f"param_{param_col}"
        
        # Extract and group by parameter value to find the mean test score
        temp_df = cv_results[[search_col, "mean_test_score"]].copy()
        temp_df[search_col] = temp_df[search_col].fillna("None").astype(str)
        grouped_df = temp_df.groupby(search_col)["mean_test_score"].mean().reset_index()
        
        # Sort the values logically
        def sort_key(x):
            try: return float(x)
            except ValueError: return float('inf')
                
        grouped_df['sort_val'] = grouped_df[search_col].apply(sort_key)
        grouped_df = grouped_df.sort_values('sort_val').drop('sort_val', axis=1)

        # Write to file
        file.write(f"{param_title} ({param_col})\n")
        for _, row in grouped_df.iterrows():
            file.write(f"  {row[search_col]}: {row['mean_test_score']:.4f}\n")
        file.write("\n")

# ============================================================
# 5. GENERATE TUNING LINE GRAPHS
# ============================================================
print("\nGenerating hyperparameter tuning line graphs...")

for param_col, param_title in parameters_to_plot.items():
    search_col = f"param_{param_col}"
    temp_df = cv_results[[search_col, "mean_test_score"]].copy()
    temp_df[search_col] = temp_df[search_col].fillna("None").astype(str)
    
    grouped_df = temp_df.groupby(search_col)["mean_test_score"].mean().reset_index()
    
    def sort_key(x):
        try:
            return float(x)
        except ValueError:
            return float('inf')
            
    grouped_df['sort_val'] = grouped_df[search_col].apply(sort_key)
    grouped_df = grouped_df.sort_values('sort_val').drop('sort_val', axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(grouped_df[search_col], grouped_df["mean_test_score"], marker="o", linestyle="-", color="b")
    plt.title(f"Accuracy vs {param_title}")
    plt.xlabel(param_title)
    plt.ylabel("Mean CV Accuracy")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    
    safe_name = param_col.replace('classifier__', '')
    save_filename = os.path.join(GRAPH_PATH, f"tuning_{safe_name}.png")
    plt.savefig(save_filename, dpi=300, bbox_inches="tight")
    plt.close()

print("\nProcess Complete! Check the 'results' folder.")