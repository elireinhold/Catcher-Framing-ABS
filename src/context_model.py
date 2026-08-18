from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "model_data_2025_2026.csv"
OUTPUT_FILE = PROCESSED_DIR / "context_predictions_2025_2026.csv"


# ============================================================
# Load data
# ============================================================

def load_data():
    print("=" * 60)
    print("MODEL 2: LOCATION + PITCH CONTEXT")
    print("=" * 60)

    data = pd.read_csv(INPUT_FILE)

    print(f"\nLoaded {len(data):,} pitches.")

    return data


# ============================================================
# Prepare variables
# ============================================================

def prepare_variables(data):
    data = data.copy()

    # Location variables will receive polynomial features.
    location_features = [
        "plate_x",
        "plate_z",
    ]

    # Other numeric pitch/context variables.
    numeric_features = [
        "release_speed",
        "pfx_x",
        "pfx_z",
        "balls",
        "strikes",
    ]

    categorical_features = [
        "pitch_type",
        "stand",
        "p_throws",
    ]

    required = [
        "called_strike",
        "game_date",
        "pitcher",
        "batter",
        "catcher",
    ]

    data = data.dropna(subset=required).copy()

    for column in location_features + numeric_features:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

    return (
        data,
        location_features,
        numeric_features,
        categorical_features,
    )


# ============================================================
# Chronological split
# ============================================================

def chronological_split(data):
    data = data.sort_values("game_date").reset_index(drop=True)

    split_index = int(len(data) * 0.80)

    train = data.iloc[:split_index].copy()
    test = data.iloc[split_index:].copy()

    return train, test


# ============================================================
# Build model
# ============================================================

def build_model(
    location_features,
    numeric_features,
    categorical_features,
):

    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------
    #
    # This is the important change.
    #
    # Instead of assuming the relationship between location
    # and called-strike probability is linear, we allow the
    # model to learn a nonlinear strike-zone surface.
    #
    location_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "polynomial",
            PolynomialFeatures(
                degree=3,
                include_bias=False
            )
        ),
        (
            "scaler",
            StandardScaler()
        ),
    ])

    # --------------------------------------------------------
    # OTHER NUMERIC FEATURES
    # --------------------------------------------------------

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
    ])

    # --------------------------------------------------------
    # CATEGORICAL FEATURES
    # --------------------------------------------------------

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        ),
    ])

    # --------------------------------------------------------
    # COMBINE FEATURES
    # --------------------------------------------------------

    preprocessor = ColumnTransformer([
        (
            "location",
            location_pipeline,
            location_features
        ),
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
    ])

    # --------------------------------------------------------
    # LOGISTIC REGRESSION
    # --------------------------------------------------------

    model = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=3000
            )
        ),
    ])

    return model


# ============================================================
# Evaluate model
# ============================================================

def evaluate_model(model, X_test, y_test):

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    print("\n" + "=" * 60)
    print("MODEL 2 EVALUATION")
    print("=" * 60)

    print(
        f"\nAccuracy:  "
        f"{accuracy_score(y_test, predictions):.4f}"
    )

    print(
        f"Log Loss:  "
        f"{log_loss(y_test, probabilities):.4f}"
    )

    print(
        f"ROC AUC:   "
        f"{roc_auc_score(y_test, probabilities):.4f}"
    )

    return probabilities


# ============================================================
# Catcher residual analysis
# ============================================================

def catcher_analysis(test, probabilities):

    results = test.copy()

    results["expected_strike_probability"] = probabilities

    results["residual"] = (
        results["called_strike"]
        - results["expected_strike_probability"]
    )

    summary = (
        results
        .groupby("catcher")
        .agg(
            pitches=("called_strike", "size"),

            actual_strike_rate=(
                "called_strike",
                "mean"
            ),

            expected_strike_rate=(
                "expected_strike_probability",
                "mean"
            ),

            average_residual=(
                "residual",
                "mean"
            ),
        )
        .reset_index()
    )

    summary = summary[
        summary["pitches"] >= 50
    ].copy()

    summary = summary.sort_values(
        "average_residual",
        ascending=False
    )

    print("\n" + "=" * 60)
    print("MODEL 2 CATCHER ANALYSIS")
    print("=" * 60)

    print(
        "\nThese residuals are still exploratory."
        "\nThey are NOT Bayesian catcher framing estimates."
    )

    print(
        f"\nCatchers with >= 50 pitches: "
        f"{len(summary)}"
    )

    print("\nHighest positive residuals:")

    print(
        summary.head(15)
        .round(4)
        .to_string(index=False)
    )

    print("\nLowest residuals:")

    print(
        summary.tail(15)
        .sort_values(
            "average_residual"
        )
        .round(4)
        .to_string(index=False)
    )

    return results, summary


# ============================================================
# Season analysis
# ============================================================

def season_analysis(results):

    print("\n" + "=" * 60)
    print("2025 vs 2026 MODEL 2 COMPARISON")
    print("=" * 60)

    summary = (
        results
        .groupby("season")
        .agg(
            pitches=("called_strike", "size"),

            actual_strike_rate=(
                "called_strike",
                "mean"
            ),

            expected_strike_rate=(
                "expected_strike_probability",
                "mean"
            ),

            average_residual=(
                "residual",
                "mean"
            ),
        )
        .reset_index()
    )

    print(
        summary
        .round(4)
        .to_string(index=False)
    )


# ============================================================
# Main
# ============================================================

def main():

    data = load_data()

    (
        data,
        location_features,
        numeric_features,
        categorical_features,
    ) = prepare_variables(data)

    print("\nPreparing variables...")

    print("\nModel features:")

    print("  Location:")
    for column in location_features:
        print(f"    - {column} [polynomial degree 3]")

    print("  Numeric:")
    for column in numeric_features:
        print(f"    - {column}")

    print("  Categorical:")
    for column in categorical_features:
        print(f"    - {column}")

    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    train, test = chronological_split(data)

    print(
        "\nCreating chronological train/test split..."
    )

    print(
        f"Training pitches: {len(train):,}"
    )

    print(
        f"Testing pitches:  {len(test):,}"
    )

    print(
        f"\nTraining dates:\n"
        f"  {train['game_date'].min()} -> "
        f"{train['game_date'].max()}"
    )

    print(
        f"Testing dates:\n"
        f"  {test['game_date'].min()} -> "
        f"{test['game_date'].max()}"
    )

    # --------------------------------------------------------
    # MODEL DATA
    # --------------------------------------------------------

    feature_columns = (
        location_features
        + numeric_features
        + categorical_features
    )

    X_train = train[feature_columns]
    y_train = train["called_strike"].astype(int)

    X_test = test[feature_columns]
    y_test = test["called_strike"].astype(int)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\nTraining Model 2...")

    model = build_model(
        location_features,
        numeric_features,
        categorical_features
    )

    model.fit(
        X_train,
        y_train
    )

    print("Model training complete.")

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    probabilities = evaluate_model(
        model,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # CATCHER ANALYSIS
    # --------------------------------------------------------

    results, summary = catcher_analysis(
        test,
        probabilities
    )

    # --------------------------------------------------------
    # SEASON SUMMARY
    # --------------------------------------------------------

    season_analysis(results)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nSaved predictions to:\n"
        f"  {OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL 2 COMPLETE")
    print("=" * 60)

    print(
        "\nModel 2 accounts for:"
        "\n  - Nonlinear pitch location"
        "\n  - Pitch type"
        "\n  - Velocity"
        "\n  - Pitch movement"
        "\n  - Count"
        "\n  - Batter handedness"
        "\n  - Pitcher handedness"
    )

    print(
        "\nThe catcher is still NOT included in the model."
        "\nThis is intentional."
    )

    print(
        "\nNext:"
        "\nBuild the Bayesian hierarchical catcher model."
    )


if __name__ == "__main__":
    main()