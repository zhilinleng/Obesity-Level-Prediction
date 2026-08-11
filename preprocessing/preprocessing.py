import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# FILE PATH
# ============================================================

DATA_PATH = "data/ObesityDataSet_raw_and_data_sinthetic.csv"

GRAPH_PATH = "results/graphs"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """
    Load the obesity dataset from CSV.
    """

    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("DATASET LOADING")
    print("=" * 60)

    print(f"Dataset shape: {df.shape}")

    return df


# ============================================================
# DATA EXPLORATION
# ============================================================

def explore_data(df):
    """
    Display basic information about the dataset.
    """

    print("\n" + "=" * 60)
    print("DATA EXPLORATION")
    print("=" * 60)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset information:")
    print(df.info())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nNumber of duplicate rows:")
    print(df.duplicated().sum())

    print("\nTarget distribution:")
    print(df["NObeyesdad"].value_counts())


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(df):
    """
    Remove duplicate records from the dataset.
    """

    before = len(df)

    df = df.drop_duplicates().reset_index(drop=True)

    after = len(df)

    print("\n" + "=" * 60)
    print("DUPLICATE REMOVAL")
    print("=" * 60)

    print(f"Rows before removing duplicates: {before}")
    print(f"Rows after removing duplicates:  {after}")
    print(f"Duplicates removed:              {before - after}")

    return df


# ============================================================
# CLASS DISTRIBUTION GRAPH
# ============================================================

def create_class_distribution_graph(df):
    """
    Create graph showing the distribution of obesity classes.
    """

    os.makedirs(GRAPH_PATH, exist_ok=True)

    plt.figure(figsize=(10, 6))

    sns.countplot(
        data=df,
        x="NObeyesdad",
        order=df["NObeyesdad"].value_counts().index
    )

    plt.title("Distribution of Obesity Levels")
    plt.xlabel("Obesity Level")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            GRAPH_PATH,
            "01_class_distribution.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


# ============================================================
# PREPROCESSING
# ============================================================

def prepare_data(df):
    """
    Prepare features and target.
    The preprocessing transformer is returned without
    fitting it, so that it can be safely fitted inside
    the Random Forest pipeline during cross-validation.
    """

    # Separate features and target
    X = df.drop("NObeyesdad", axis=1)
    y = df["NObeyesdad"]

    # Identify categorical features
    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    # Identify numerical features
    numerical_features = X.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    print("\n" + "=" * 60)
    print("FEATURE INFORMATION")
    print("=" * 60)

    print("\nCategorical features:")
    for feature in categorical_features:
        print(f"- {feature}")

    print("\nNumerical features:")
    for feature in numerical_features:
        print(f"- {feature}")

    # One-Hot Encoding
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        ],
        remainder="passthrough"
    )

    return X, y, preprocessor


# ============================================================
# TRAIN-TEST SPLIT
# ============================================================

def split_data(X, y):
    """
    Split dataset into training and testing sets.
    Stratification maintains the class distribution.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\n" + "=" * 60)
    print("TRAIN-TEST SPLIT")
    print("=" * 60)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    print("\nTraining class distribution:")
    print(y_train.value_counts())

    print("\nTesting class distribution:")
    print(y_test.value_counts())

    return X_train, X_test, y_train, y_test


# ============================================================
# MAIN PREPROCESSING FUNCTION
# ============================================================

def preprocess_data():
    """
    Complete preprocessing process.
    """

    df = load_data()

    explore_data(df)

    df = remove_duplicates(df)

    create_class_distribution_graph(df)

    X, y, preprocessor = prepare_data(df)

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )