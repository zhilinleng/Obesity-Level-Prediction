"""
Random Forest Classifier
Estimation of Obesity Levels Based on Eating Habits
and Physical Condition

Dataset:
Estimation of Obesity Levels Based on Eating Habits
and Physical Condition

Model:
Random Forest Classifier

Pipeline:
1. Load and preprocess data
2. Train baseline Random Forest
3. Hyperparameter tuning using RandomizedSearchCV
4. 5-Fold Stratified Cross-Validation
5. Generate parameter tuning line graphs
6. Train final tuned Random Forest
7. Evaluate final model
8. Generate confusion matrix
9. Generate feature importance
10. Generate model performance graph
11. Save results
12. Save final model
"""


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import sys
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# ============================================================
# 2. PROJECT PATH
# ============================================================

# Get the root folder of the project
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# Allow Python to find the preprocessing folder
sys.path.insert(
    0,
    PROJECT_ROOT
)


# Import preprocessing function
from preprocessing.preprocessing import preprocess_data


# ============================================================
# 3. OUTPUT FOLDERS
# ============================================================

RESULT_PATH = os.path.join(
    PROJECT_ROOT,
    "results"
)

GRAPH_PATH = os.path.join(
    RESULT_PATH,
    "graphs"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_model",
    "random_forest_model.pkl"
)

RESULT_FILE = os.path.join(
    RESULT_PATH,
    "random_forest_results.txt"
)


# Create folders automatically
os.makedirs(
    GRAPH_PATH,
    exist_ok=True
)

os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)


# ============================================================
# 4. START PROGRAM
# ============================================================

print("\n")
print("=" * 70)
print("RANDOM FOREST CLASSIFIER")
print("ESTIMATION OF OBESITY LEVELS")
print("=" * 70)


# ============================================================
# 5. LOAD AND PREPROCESS DATA
# ============================================================

print("\n")
print("=" * 70)
print("PREPROCESSING DATA")
print("=" * 70)


(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
) = preprocess_data()


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 6. STRATIFIED CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


print("\nCross-validation:")
print("Method: Stratified K-Fold")
print("Number of folds: 5")


# ============================================================
# 7. BASELINE RANDOM FOREST
# ============================================================

print("\n")
print("=" * 70)
print("BASELINE RANDOM FOREST")
print("=" * 70)


baseline_pipeline = Pipeline([
    (
        "preprocessing",
        preprocessor
    ),

    (
        "classifier",
        RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        )
    )
])


print("\nTraining baseline Random Forest...")


baseline_pipeline.fit(
    X_train,
    y_train
)


# Predict test data
baseline_pred = baseline_pipeline.predict(
    X_test
)


# Calculate baseline performance
baseline_accuracy = accuracy_score(
    y_test,
    baseline_pred
)

baseline_precision = precision_score(
    y_test,
    baseline_pred,
    average="weighted"
)

baseline_recall = recall_score(
    y_test,
    baseline_pred,
    average="weighted"
)

baseline_f1 = f1_score(
    y_test,
    baseline_pred,
    average="weighted"
)


print(
    f"\nBaseline Accuracy:  "
    f"{baseline_accuracy:.4f}"
)

print(
    f"Baseline Precision: "
    f"{baseline_precision:.4f}"
)

print(
    f"Baseline Recall:    "
    f"{baseline_recall:.4f}"
)

print(
    f"Baseline F1-score:  "
    f"{baseline_f1:.4f}"
)


# ============================================================
# 8. RANDOM FOREST PIPELINE FOR TUNING
# ============================================================

print("\n")
print("=" * 70)
print("RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 70)


rf_pipeline = Pipeline([
    (
        "preprocessing",
        preprocessor
    ),

    (
        "classifier",
        RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        )
    )
])


# ============================================================
# 9. HYPERPARAMETER SEARCH SPACE
# ============================================================

param_distributions = {

    # --------------------------------------------------------
    # Number of trees
    # --------------------------------------------------------

    "classifier__n_estimators": [
        100,
        200,
        300,
        500,
        700,
        1000
    ],

    # --------------------------------------------------------
    # Maximum tree depth
    # --------------------------------------------------------

    "classifier__max_depth": [
        10,
        20,
        30,
        50,
        75,
        100,
        None
    ],

    # --------------------------------------------------------
    # Number of features considered at each split
    # --------------------------------------------------------

    "classifier__max_features": [
        "sqrt",
        "log2",
        None
    ],

    # --------------------------------------------------------
    # Splitting criterion
    # --------------------------------------------------------

    "classifier__criterion": [
        "gini",
        "entropy",
        "log_loss"
    ],

    # --------------------------------------------------------
    # Minimum samples required to split an internal node
    # --------------------------------------------------------

    "classifier__min_samples_split": [
        2,
        5,
        10,
        15
    ],

    # --------------------------------------------------------
    # Minimum samples required at a leaf node
    # --------------------------------------------------------

    "classifier__min_samples_leaf": [
        1,
        2,
        4,
        8
    ]
}


# ============================================================
# 10. RANDOMIZED SEARCH CV
# ============================================================

random_search = RandomizedSearchCV(

    estimator=rf_pipeline,

    param_distributions=param_distributions,

    # Test 100 random parameter combinations
    n_iter=100,

    # Optimisation metric
    scoring="accuracy",

    # 5-Fold Stratified Cross-Validation
    cv=cv,

    random_state=42,

    # Use all CPU cores
    n_jobs=-1,

    verbose=1,

    return_train_score=True
)


print("\nRandomizedSearchCV settings:")
print("Random parameter combinations: 100")
print("Cross-validation: 5-Fold Stratified")
print("Evaluation metric: Accuracy")


print("\nStarting hyperparameter tuning...")


random_search.fit(
    X_train,
    y_train
)


# ============================================================
# 11. BEST HYPERPARAMETERS
# ============================================================

print("\n")
print("=" * 70)
print("BEST RANDOM FOREST HYPERPARAMETERS")
print("=" * 70)


print("\nBest parameters:")


for parameter, value in (
    random_search.best_params_.items()
):

    print(
        f"{parameter}: {value}"
    )


print(
    "\nBest Mean CV Accuracy: "
    f"{random_search.best_score_:.4f}"
)


# ============================================================
# 12. FINAL MODEL
# ============================================================

final_model = (
    random_search.best_estimator_
)


# ============================================================
# 13. FINAL TEST PREDICTION
# ============================================================

print("\n")
print("=" * 70)
print("FINAL MODEL EVALUATION")
print("=" * 70)


y_pred = final_model.predict(
    X_test
)

y_pred_proba = final_model.predict_proba(
    X_test
)

# ============================================================
# 14. FINAL PERFORMANCE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

roc_auc = roc_auc_score(
    y_test,
    y_pred_proba,
    multi_class="ovr"
)

print(
    f"\nFinal Accuracy:  "
    f"{accuracy:.4f}"
)

print(
    f"Final Precision: "
    f"{precision:.4f}"
)

print(
    f"Final Recall:    "
    f"{recall:.4f}"
)

print(
    f"Final F1-score:  "
    f"{f1:.4f}"
)

print(
    f"Final ROC-AUC:   "
    f"{roc_auc:.4f}"
)

# ============================================================
# 15. CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)


report = classification_report(
    y_test,
    y_pred
)


print(report)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

print("\nGenerating confusion matrix...")


cm = confusion_matrix(
    y_test,
    y_pred,
    labels=final_model.classes_
)


plt.figure(
    figsize=(11, 8)
)


sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=final_model.classes_,
    yticklabels=final_model.classes_
)


plt.title(
    "Random Forest Confusion Matrix"
)

plt.xlabel(
    "Predicted Obesity Level"
)

plt.ylabel(
    "Actual Obesity Level"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.yticks(
    rotation=0
)

plt.tight_layout()


plt.savefig(
    os.path.join(
        GRAPH_PATH,
        "08_confusion_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.show()

plt.close()


# ============================================================
# 17. FEATURE IMPORTANCE
# ============================================================

print("\nGenerating feature importance graph...")


# Get fitted preprocessing step
preprocessor_fitted = (
    final_model
    .named_steps[
        "preprocessing"
    ]
)


# Get fitted Random Forest
rf_fitted = (
    final_model
    .named_steps[
        "classifier"
    ]
)


# Get feature names after encoding
feature_names = (
    preprocessor_fitted
    .get_feature_names_out()
)


# Get feature importance
feature_importances = (
    rf_fitted
    .feature_importances_
)


# Create DataFrame
feature_importance_df = pd.DataFrame({

    "Feature":
        feature_names,

    "Importance":
        feature_importances

})


# Sort from highest to lowest
feature_importance_df = (
    feature_importance_df
    .sort_values(
        by="Importance",
        ascending=False
    )
)


# Select top 15 features
top_features = (
    feature_importance_df
    .head(15)
)


plt.figure(
    figsize=(10, 7)
)


plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)


plt.title(
    "Top 15 Random Forest Feature Importances"
)

plt.xlabel(
    "Feature Importance"
)

plt.ylabel(
    "Feature"
)

plt.tight_layout()


plt.savefig(
    os.path.join(
        GRAPH_PATH,
        "09_feature_importance.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.show()

plt.close()


# ============================================================
# 18. MODEL PERFORMANCE GRAPH
# ============================================================

print("\nGenerating model performance graph...")


metrics = {

    "Accuracy":
        accuracy,

    "Precision":
        precision,

    "Recall":
        recall,

    "F1-Score":
        f1

}


plt.figure(
    figsize=(8, 6)
)


plt.bar(
    metrics.keys(),
    metrics.values()
)


plt.title(
    "Random Forest Model Performance"
)

plt.ylabel(
    "Score"
)

plt.ylim(
    0,
    1
)

plt.tight_layout()


plt.savefig(
    os.path.join(
        GRAPH_PATH,
        "10_model_performance.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.show()

plt.close()


# ============================================================
# 19. PARAMETER TUNING FUNCTIONS
# ============================================================

def plot_numerical_parameter(
    parameter_name,
    parameter_values,
    title,
    xlabel,
    filename
):
    """
    Evaluate a numerical Random Forest parameter
    using 5-Fold Stratified Cross-Validation
    and generate a line graph.
    """

    scores = []


    print("\n")
    print(
        f"Evaluating {parameter_name}..."
    )


    for value in parameter_values:

        temp_pipeline = Pipeline([

            (
                "preprocessing",
                preprocessor
            ),

            (
                "classifier",
                RandomForestClassifier(

                    random_state=42,

                    n_jobs=-1,

                    **{
                        parameter_name:
                            value
                    }

                )
            )

        ])


        cv_scores = cross_val_score(

            temp_pipeline,

            X_train,

            y_train,

            cv=cv,

            scoring="accuracy",

            n_jobs=-1

        )


        mean_score = (
            cv_scores.mean()
        )


        scores.append(
            mean_score
        )


        print(
            f"{parameter_name} = "
            f"{value}: "
            f"{mean_score:.4f}"
        )


    # Create graph
    plt.figure(
        figsize=(8, 6)
    )


    plt.plot(

        parameter_values,

        scores,

        marker="o",

        linewidth=2

    )


    plt.title(
        title
    )

    plt.xlabel(
        xlabel
    )

    plt.ylabel(
        "Mean Cross-Validation Accuracy"
    )


    plt.grid(
        True,
        alpha=0.3
    )


    plt.tight_layout()


    plt.savefig(

        os.path.join(

            GRAPH_PATH,

            filename

        ),

        dpi=300,

        bbox_inches="tight"

    )


    plt.show()

    plt.close()


    return scores


# ============================================================
# 20. CATEGORICAL PARAMETER GRAPH FUNCTION
# ============================================================

def plot_categorical_parameter(

    parameter_name,

    parameter_values,

    title,

    xlabel,

    filename

):
    """
    Evaluate a categorical Random Forest parameter
    using 5-Fold Stratified Cross-Validation
    and generate a line graph.
    """

    scores = []


    print("\n")
    print(
        f"Evaluating {parameter_name}..."
    )


    for value in parameter_values:

        temp_pipeline = Pipeline([

            (
                "preprocessing",
                preprocessor
            ),

            (
                "classifier",
                RandomForestClassifier(

                    random_state=42,

                    n_jobs=-1,

                    **{
                        parameter_name:
                            value
                    }

                )

            )

        ])


        cv_scores = cross_val_score(

            temp_pipeline,

            X_train,

            y_train,

            cv=cv,

            scoring="accuracy",

            n_jobs=-1

        )


        mean_score = (
            cv_scores.mean()
        )


        scores.append(
            mean_score
        )


        print(
            f"{parameter_name} = "
            f"{value}: "
            f"{mean_score:.4f}"
        )


    # Create graph
    plt.figure(
        figsize=(8, 6)
    )


    plt.plot(

        range(
            len(parameter_values)
        ),

        scores,

        marker="o",

        linewidth=2

    )


    # Put parameter names on x-axis
    plt.xticks(

        range(
            len(parameter_values)
        ),

        [
            str(value)
            for value in parameter_values
        ]

    )


    plt.title(
        title
    )

    plt.xlabel(
        xlabel
    )

    plt.ylabel(
        "Mean Cross-Validation Accuracy"
    )


    plt.grid(
        True,
        alpha=0.3
    )


    plt.tight_layout()


    plt.savefig(

        os.path.join(

            GRAPH_PATH,

            filename

        ),

        dpi=300,

        bbox_inches="tight"

    )


    plt.show()

    plt.close()


    return scores


# ============================================================
# 21. PARAMETER TUNING GRAPHS
# ============================================================

print("\n")
print("=" * 70)
print("PARAMETER TUNING ANALYSIS")
print("=" * 70)


# ============================================================
# 21.1 n_estimators
# ============================================================

n_estimators_values = [

    100,
    200,
    300,
    500,
    700,
    1000

]


n_estimators_scores = plot_numerical_parameter(

    parameter_name="n_estimators",

    parameter_values=n_estimators_values,

    title=(
        "Effect of n_estimators "
        "on Random Forest Accuracy"
    ),

    xlabel=(
        "Number of Trees (n_estimators)"
    ),

    filename=(
        "02_n_estimators_tuning.png"
    )

)


# ============================================================
# 21.2 max_depth
# ============================================================

max_depth_values = [

    10,
    20,
    30,
    50,
    75,
    100

]


max_depth_scores = plot_numerical_parameter(

    parameter_name="max_depth",

    parameter_values=max_depth_values,

    title=(
        "Effect of max_depth "
        "on Random Forest Accuracy"
    ),

    xlabel=(
        "Maximum Tree Depth (max_depth)"
    ),

    filename=(
        "03_max_depth_tuning.png"
    )

)


# ============================================================
# 21.3 max_features
# ============================================================

max_features_values = [

    "sqrt",
    "log2",
    None

]


max_features_scores = plot_categorical_parameter(

    parameter_name="max_features",

    parameter_values=max_features_values,

    title=(
        "Effect of max_features "
        "on Random Forest Accuracy"
    ),

    xlabel=(
        "Features Considered at Each Split"
    ),

    filename=(
        "04_max_features_tuning.png"
    )

)


# ============================================================
# 21.4 criterion
# ============================================================

criterion_values = [

    "gini",
    "entropy",
    "log_loss"

]


criterion_scores = plot_categorical_parameter(

    parameter_name="criterion",

    parameter_values=criterion_values,

    title=(
        "Effect of criterion "
        "on Random Forest Accuracy"
    ),

    xlabel=(
        "Splitting Criterion"
    ),

    filename=(
        "05_criterion_tuning.png"
    )

)


# ============================================================
# 21.5 min_samples_split
# ============================================================

min_samples_split_values = [

    2,
    5,
    10,
    15

]


min_samples_split_scores = plot_numerical_parameter(

    parameter_name="min_samples_split",

    parameter_values=min_samples_split_values,

    title=(
        "Effect of min_samples_split "
        "on Random Forest Accuracy"
    ),

    xlabel=(
        "Minimum Samples Required to Split"
    ),

    filename=(
        "06_min_samples_split_tuning.png"
    )

)


# ============================================================
# 21.6 min_samples_leaf
# ============================================================

min_samples_leaf_values = [

    1,
    2,
    4,
    8

]


min_samples_leaf_scores = plot_numerical_parameter(

    parameter_name="min_samples_leaf",

    parameter_values=min_samples_leaf_values,

    title=(
        "Effect of min_samples_leaf "
        "on Random Forest Accuracy"
    ),

    xlabel=(
        "Minimum Samples Required at Leaf"
    ),

    filename=(
        "07_min_samples_leaf_tuning.png"
    )

)


# ============================================================
# 22. BEST INDIVIDUAL PARAMETER VALUES
# ============================================================

print("\n")
print("=" * 70)
print("BEST INDIVIDUAL PARAMETER VALUES")
print("=" * 70)


# ------------------------------------------------------------
# n_estimators
# ------------------------------------------------------------

best_n_estimators_index = (
    n_estimators_scores.index(
        max(n_estimators_scores)
    )
)


print("\nn_estimators:")

print(
    f"Best value = "
    f"{n_estimators_values[best_n_estimators_index]}"
)

print(
    f"Accuracy = "
    f"{n_estimators_scores[best_n_estimators_index]:.4f}"
)


# ------------------------------------------------------------
# max_depth
# ------------------------------------------------------------

best_max_depth_index = (
    max_depth_scores.index(
        max(max_depth_scores)
    )
)


print("\nmax_depth:")

print(
    f"Best value = "
    f"{max_depth_values[best_max_depth_index]}"
)

print(
    f"Accuracy = "
    f"{max_depth_scores[best_max_depth_index]:.4f}"
)


# ------------------------------------------------------------
# max_features
# ------------------------------------------------------------

best_max_features_index = (
    max_features_scores.index(
        max(max_features_scores)
    )
)


print("\nmax_features:")

print(
    f"Best value = "
    f"{max_features_values[best_max_features_index]}"
)

print(
    f"Accuracy = "
    f"{max_features_scores[best_max_features_index]:.4f}"
)


# ------------------------------------------------------------
# criterion
# ------------------------------------------------------------

best_criterion_index = (
    criterion_scores.index(
        max(criterion_scores)
    )
)


print("\ncriterion:")

print(
    f"Best value = "
    f"{criterion_values[best_criterion_index]}"
)

print(
    f"Accuracy = "
    f"{criterion_scores[best_criterion_index]:.4f}"
)


# ------------------------------------------------------------
# min_samples_split
# ------------------------------------------------------------

best_min_samples_split_index = (
    min_samples_split_scores.index(
        max(min_samples_split_scores)
    )
)


print("\nmin_samples_split:")

print(
    f"Best value = "
    f"{min_samples_split_values[best_min_samples_split_index]}"
)

print(
    f"Accuracy = "
    f"{min_samples_split_scores[best_min_samples_split_index]:.4f}"
)


# ------------------------------------------------------------
# min_samples_leaf
# ------------------------------------------------------------

best_min_samples_leaf_index = (
    min_samples_leaf_scores.index(
        max(min_samples_leaf_scores)
    )
)


print("\nmin_samples_leaf:")

print(
    f"Best value = "
    f"{min_samples_leaf_values[best_min_samples_leaf_index]}"
)

print(
    f"Accuracy = "
    f"{min_samples_leaf_scores[best_min_samples_leaf_index]:.4f}"
)


# ============================================================
# 23. SAVE RESULTS TO TXT
# ============================================================

print("\n")
print("=" * 70)
print("SAVING RESULTS")
print("=" * 70)


with open(
    RESULT_FILE,
    "w"
) as file:

    file.write(
        "RANDOM FOREST CLASSIFIER RESULTS\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )


    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    file.write(
        "BASELINE RANDOM FOREST\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    file.write(
        f"Accuracy:  "
        f"{baseline_accuracy:.4f}\n"
    )

    file.write(
        f"Precision: "
        f"{baseline_precision:.4f}\n"
    )

    file.write(
        f"Recall:    "
        f"{baseline_recall:.4f}\n"
    )

    file.write(
        f"F1-score:  "
        f"{baseline_f1:.4f}\n\n"
    )


    # --------------------------------------------------------
    # Final tuned model
    # --------------------------------------------------------

    file.write(
        "FINAL TUNED RANDOM FOREST\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    file.write(
        f"Accuracy:  "
        f"{accuracy:.4f}\n"
    )

    file.write(
        f"Precision: "
        f"{precision:.4f}\n"
    )

    file.write(
        f"Recall:    "
        f"{recall:.4f}\n"
    )

    file.write(
        f"F1-score:  "
        f"{f1:.4f}\n\n"
    )


    # --------------------------------------------------------
    # Best CV score
    # --------------------------------------------------------

    file.write(
        "BEST RANDOMIZEDSEARCHCV SCORE\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    file.write(
        f"Mean CV Accuracy: "
        f"{random_search.best_score_:.4f}\n\n"
    )


    # --------------------------------------------------------
    # Best combination of parameters
    # --------------------------------------------------------

    file.write(
        "BEST RANDOMIZEDSEARCHCV PARAMETERS\n"
    )

    file.write(
        "-" * 70 + "\n"
    )


    for parameter, value in (
        random_search.best_params_.items()
    ):

        file.write(
            f"{parameter}: {value}\n"
        )


    file.write("\n\n")


    # ========================================================
    # Individual Parameter Analysis
    # ========================================================

    file.write(
        "INDIVIDUAL PARAMETER ANALYSIS\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )


    # --------------------------------------------------------
    # n_estimators
    # --------------------------------------------------------

    file.write(
        "n_estimators\n"
    )

    for value, score in zip(
        n_estimators_values,
        n_estimators_scores
    ):

        file.write(
            f"{value}: "
            f"{score:.4f}\n"
        )


    file.write("\n")


    # --------------------------------------------------------
    # max_depth
    # --------------------------------------------------------

    file.write(
        "max_depth\n"
    )

    for value, score in zip(
        max_depth_values,
        max_depth_scores
    ):

        file.write(
            f"{value}: "
            f"{score:.4f}\n"
        )


    file.write("\n")


    # --------------------------------------------------------
    # max_features
    # --------------------------------------------------------

    file.write(
        "max_features\n"
    )

    for value, score in zip(
        max_features_values,
        max_features_scores
    ):

        file.write(
            f"{value}: "
            f"{score:.4f}\n"
        )


    file.write("\n")


    # --------------------------------------------------------
    # criterion
    # --------------------------------------------------------

    file.write(
        "criterion\n"
    )

    for value, score in zip(
        criterion_values,
        criterion_scores
    ):

        file.write(
            f"{value}: "
            f"{score:.4f}\n"
        )


    file.write("\n")


    # --------------------------------------------------------
    # min_samples_split
    # --------------------------------------------------------

    file.write(
        "min_samples_split\n"
    )

    for value, score in zip(
        min_samples_split_values,
        min_samples_split_scores
    ):

        file.write(
            f"{value}: "
            f"{score:.4f}\n"
        )


    file.write("\n")


    # --------------------------------------------------------
    # min_samples_leaf
    # --------------------------------------------------------

    file.write(
        "min_samples_leaf\n"
    )

    for value, score in zip(
        min_samples_leaf_values,
        min_samples_leaf_scores
    ):

        file.write(
            f"{value}: "
            f"{score:.4f}\n"
        )


    file.write("\n\n")


    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    file.write(
        "CLASSIFICATION REPORT\n"
    )

    file.write(
        "=" * 70 + "\n"
    )

    file.write(
        report
    )


# ============================================================
# 24. SAVE FINAL MODEL
# ============================================================

print("\nSaving final Random Forest model...")


joblib.dump(
    final_model,
    MODEL_PATH
)


# ============================================================
# 25. FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("RANDOM FOREST PROCESS COMPLETED")
print("=" * 70)


print(
    "\nFinal Test Performance:"
)

print(
    f"Accuracy:  {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall:    {recall:.4f}"
)

print(
    f"F1-score:  {f1:.4f}"
)


print(
    "\nResults saved to:"
)

print(
    RESULT_FILE
)


print(
    "\nGraphs saved to:"
)

print(
    GRAPH_PATH
)


print(
    "\nModel saved to:"
)

print(
    MODEL_PATH
)


print("\n")
print("=" * 70)