# The model predicts call-strike probability using pitch location only
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = PROCESSED_DIR / "model_data_2025_2026.csv"
OUTPUT_FILE = PROCESSED_DIR / "baseline_predictions_2025_2026.csv"


# Load data
def load_data():
    data = pd.read_csv(INPUT_FILE)
    print(f"\nLoaded {len(data):,} pitches.")
    return data

# Create location features
def create_features(data):
    data = data.copy()

    # pitches must have plate location, strike call, date
    required = [
        "plate_x",
        "plate_z",
        "called_strike",
        "game_date",
    ]

    data = data.dropna(subset=required).copy()

    # Polynomial location terms allow the model to learn a curved
    # strike probability surface instead of assuming a straight line.
    X = data[["plate_x", "plate_z"]].copy()
    y = data["called_strike"].astype(int)

    return data, X, y

# Chronological split
def chronological_split(data, X, y):
    # Sort by date so future pitches are never used to predict earlier pitches
    order = data["game_date"].sort_values().index

    data = data.loc[order].reset_index(drop=True)
    X = X.loc[order].reset_index(drop=True)
    y = y.loc[order].reset_index(drop=True)

    split_index = int(len(data) * 0.80)

    # Split into train and test (80/20)
    train_data = data.iloc[:split_index].copy()
    test_data = data.iloc[split_index:].copy()

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return (
        train_data,
        test_data,
        X_train,
        X_test,
        y_train,
        y_test,
    )

# Train model (logistic regression)
def train_model(X_train, y_train):
    model = make_pipeline(
        PolynomialFeatures(
            degree=3,
            include_bias=False
        ),
        LogisticRegression(
            max_iter=2000
        )
    )
    model.fit(X_train, y_train)
    return model

# Evaluation
def evaluate_model(model, X_test, y_test):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print("\n")
    print("MODEL EVALUATION")

    print(f"\nAccuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(f"Log Loss:  {log_loss(y_test, probabilities):.4f}")
    print(f"ROC AUC:   {roc_auc_score(y_test, probabilities):.4f}")

    return probabilities
    
# Catcher residual analysis
def catcher_analysis(test_data, probabilities):
    results = test_data.copy()

    results["expected_strike_probability"] = probabilities
    # Residual = called_strike(1 or 0) - expected_strike_probability
    # A positive residual means a strike was called on a pitch that was expected to be a ball (good framing)
    # A negative residual means a ball was called on a pitch that was expected to be a strike
    results["residual"] = (
        results["called_strike"]
        - results["expected_strike_probability"]
    )

    catcher_summary = (
        results
        .groupby("catcher")
        .agg(
            pitches=("called_strike", "size"),
            actual_strike_rate=("called_strike", "mean"),
            expected_strike_rate=(
                "expected_strike_probability",
                "mean"
            ),
            average_residual=("residual", "mean"),
        )
        .reset_index()
    )

    catcher_summary = catcher_summary[
        catcher_summary["pitches"] >= 50
    ].copy()

    catcher_summary = catcher_summary.sort_values(
        "average_residual",
        ascending=False
    )

    print("\n")
    print("INITIAL CATCHER ANALYSIS")

    print(
        f"\nCatchers with >= 50 pitches: "
        f"{len(catcher_summary)}"
    )

    # Highest residuals
    print("\nHighest positive residuals:")
    print(
        catcher_summary.head(15)
        .round(4)
        .to_string(index=False)
    )

    # Lowest residuals
    print("\nLowest residuals:")
    print(
        catcher_summary.tail(15)
        .sort_values("average_residual")
        .round(4)
        .to_string(index=False)
    )

    return results, catcher_summary


# Comparison betweeen 2025 and 2026(ABS)
def season_analysis(results):
    print("\n")
    print("2025 vs 2026 BASELINE COMPARISON")

    summary = (
        results
        .groupby("season")
        .agg(
            pitches=("called_strike", "size"),
            actual_strike_rate=("called_strike", "mean"),
            expected_strike_rate=(
                "expected_strike_probability",
                "mean"
            ),
            average_residual=("residual", "mean"),
        )
        .reset_index()
    )

    print(
        summary.round(4)
        .to_string(index=False)
    )


# Plot probability surface
# Creates a heat map showing most likely area to be called strike (saved to baseline_probability_surface_2025_2026)
def create_probability_plot(model, data):
    x_min = data["plate_x"].quantile(0.01)
    x_max = data["plate_x"].quantile(0.99)
    z_min = data["plate_z"].quantile(0.01)
    z_max = data["plate_z"].quantile(0.99)

    x_values = np.linspace(x_min, x_max, 100)
    z_values = np.linspace(z_min, z_max, 100)

    xx, zz = np.meshgrid(x_values, z_values)

    grid = pd.DataFrame({
        "plate_x": xx.ravel(),
        "plate_z": zz.ravel(),
    })

    probability = model.predict_proba(grid)[:, 1]
    probability = probability.reshape(xx.shape)

    plt.figure(figsize=(8, 8))

    plt.contourf(
        xx,
        zz,
        probability,
        levels=20,
        alpha=0.75
    )

    plt.colorbar(
        label="Predicted Called-Strike Probability"
    )

    plt.xlabel("Horizontal Plate Location (feet)")
    plt.ylabel("Vertical Plate Location (feet)")
    plt.title("Baseline Called-Strike Probability Surface\n2025 + 2026")

    plt.xlim(-2, 2)
    plt.ylim(0, 5)

    plt.tight_layout()

    output = FIGURES_DIR / "baseline_probability_surface_2025_2026.png"
    plt.savefig(output, dpi=300)
    plt.close()

    print(f"\nSaved probability surface to:\n  {output}")

def main():
    data = load_data()
    data, X, y = create_features(data)
    (
        train_data,
        test_data,
        X_train,
        X_test,
        y_train,
        y_test,
    ) = chronological_split(data, X, y)

    print(f"Training pitches: {len(train_data):,}")
    print(f"Testing pitches:  {len(test_data):,}")

    print(
        f"\nTraining dates:\n"
        f"  {train_data['game_date'].min()} -> "
        f"{train_data['game_date'].max()}"
    )

    print(
        f"Testing dates:\n"
        f"  {test_data['game_date'].min()} -> "
        f"{test_data['game_date'].max()}"
    )

    print("\nTraining model...")
    model = train_model(X_train, y_train)
    print("Model training complete.")

    probabilities = evaluate_model(
        model,
        X_test,
        y_test
    )

    results, catcher_summary = catcher_analysis(
        test_data,
        probabilities
    )

    season_analysis(results)

    create_probability_plot(
        model,
        data
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"\nSaved predictions to:\n  {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
