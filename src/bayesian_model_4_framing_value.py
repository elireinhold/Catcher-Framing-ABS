from pathlib import Path
import pickle

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

# CONFIGURATION
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "model_data_2025_2026.csv"

RESULTS_FILE = PROCESSED_DIR / "bayesian_model_4_framing_value_2025_2026.csv"
TRACE_FILE = PROCESSED_DIR / "bayesian_model_4_framing_value_trace.nc"
FALLBACK_TRACE_FILE = PROCESSED_DIR / "bayesian_model_4_framing_value_trace.pkl"
PLOT_FILE = PROCESSED_DIR / "bayesian_model_4_framing_value.png"


# SAMPLING SETTINGS (Same as Model 3)
DRAWS = 1000
TUNE = 2000
CHAINS = 4
CORES = 12
RANDOM_SEED = 42
TARGET_ACCEPT = 0.995
MAX_TREEDEPTH = 15

MIN_CATCHER_PITCHES = 100
MAX_MODEL_PITCHES = 100_000

MAX_RHAT = 1.01
MIN_ESS = 400
MIN_CATCHER_ESS = 400

# MLB plate width = 17 inches -> half-width in feet
PLATE_HALF_WIDTH = 17 / 24

# Runs added per called strike vs. called ball.
RUNS_PER_STRIKE = 0.125


# LOAD DATA
def load_data():
    data = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(data):,} pitches.")
    return data
    
# PREPARE DATA
def prepare_data(data):
    data = data.copy()

    data["game_date"] = pd.to_datetime(data["game_date"], errors="coerce")
    data["season"] = pd.to_numeric(data["season"], errors="coerce")
    data = data[data["season"].isin([2025, 2026])].copy()

    print(f"After season filtering: {len(data):,} pitches")

    required = [
        "called_strike", "catcher", "season", "plate_x", "plate_z",
        "sz_top", "sz_bot", "release_speed", "pfx_x", "pfx_z",
        "balls", "strikes", "pitch_type", "stand", "p_throws",
    ]

    before = len(data)
    data = data.dropna(subset=required).copy()
    print(f"Removed {before - len(data):,} rows missing required variables.")

    numeric_columns = [
        "called_strike", "catcher", "plate_x", "plate_z",
        "sz_top", "sz_bot", "release_speed", "pfx_x", "pfx_z",
        "balls", "strikes",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=numeric_columns).copy()

    # TRUE automated strike zone (same construction as Model 2/3)
    horizontal_in_zone = data["plate_x"].abs() <= PLATE_HALF_WIDTH
    vertical_in_zone = (
        (data["plate_z"] >= data["sz_bot"])
        & (data["plate_z"] <= data["sz_top"])
    )

    data["automated_strike"] = (
        horizontal_in_zone & vertical_in_zone
    ).astype(int)

    automated_rate = data["automated_strike"].mean()
    umpire_rate = data["called_strike"].mean()

    print(f"  Umpire called-strike rate: {umpire_rate:.4f}")
    print(f"  Automated strike rate:     {automated_rate:.4f}")

    data = data.sort_values("game_date").reset_index(drop=True)

    # Catcher filter
    catcher_counts = data["catcher"].value_counts()
    valid_catchers = catcher_counts[
        catcher_counts >= MIN_CATCHER_PITCHES
    ].index
    data = data[data["catcher"].isin(valid_catchers)].copy()

    print(f"\nCatchers with >= {MIN_CATCHER_PITCHES} pitches: {len(valid_catchers)}")

    data["catcher"] = pd.to_numeric(data["catcher"], errors="coerce")
    data = data.dropna(subset=["catcher"]).copy()
    data["catcher"] = data["catcher"].astype(int)

    # Development sample
    if MAX_MODEL_PITCHES is not None and len(data) > MAX_MODEL_PITCHES:

        print("\nDataset is large.")
        print(f"Randomly sampling {MAX_MODEL_PITCHES:,} pitches...")

        sampled_parts = []
        for season in [2025, 2026]:
            season_data = data[data["season"] == season].copy()
            if len(season_data) == 0:
                continue
            proportion = len(season_data) / len(data)
            n_sample = min(int(MAX_MODEL_PITCHES * proportion), len(season_data))
            sampled_parts.append(
                season_data.sample(n=n_sample, random_state=RANDOM_SEED)
            )

        data = pd.concat(sampled_parts, ignore_index=True)
        data = data.sort_values("game_date").reset_index(drop=True)

        print(f"Development dataset: {len(data):,} pitches")

    # Encode catchers
    catcher_values = sorted(data["catcher"].unique())
    catcher_map = {c: i for i, c in enumerate(catcher_values)}
    data["catcher_idx"] = data["catcher"].map(catcher_map).astype(int)

    catcher_2026_counts = (
        data.loc[data["season"] == 2026, "catcher"]
        .value_counts()
        .reindex(catcher_values, fill_value=0)
    )
    thin_catchers = catcher_2026_counts[
        catcher_2026_counts < MIN_CATCHER_PITCHES
    ]
    if len(thin_catchers) > 0:
        print(
            f"\nNOTE: {len(thin_catchers)} catcher(s) have fewer than "
            f"{MIN_CATCHER_PITCHES} pitches in 2026. Their catcher x season "
            "effect will be partially pooled toward the prior."
        )

    # Categorical variables
    pitch_categories = sorted(data["pitch_type"].astype(str).unique())
    pitch_map = {v: i for i, v in enumerate(pitch_categories)}
    data["pitch_type_idx"] = data["pitch_type"].astype(str).map(pitch_map).astype(int)

    stand_categories = sorted(data["stand"].astype(str).unique())
    stand_map = {v: i for i, v in enumerate(stand_categories)}
    data["stand_idx"] = data["stand"].astype(str).map(stand_map).astype(int)

    throws_categories = sorted(data["p_throws"].astype(str).unique())
    throws_map = {v: i for i, v in enumerate(throws_categories)}
    data["throws_idx"] = data["p_throws"].astype(str).map(throws_map).astype(int)

    data["season_idx"] = (data["season"] == 2026).astype(int)

    # Standardize continuous predictors
    continuous = ["release_speed", "pfx_x", "pfx_z", "balls", "strikes"]

    for column in continuous:
        mean = data[column].mean()
        std = data[column].std()
        if not np.isfinite(std) or std == 0:
            std = 1.0
        data[column + "_std"] = (data[column] - mean) / std

    print("\nFinal Model 4 dataset:")
    print(f"  Pitches:  {len(data):,}")
    print(f"  Catchers: {data['catcher'].nunique()}")
    print(f"  2025:     {(data['season'] == 2025).sum():,}")
    print(f"  2026:     {(data['season'] == 2026).sum():,}")
    print(f"  Automated strikes: {data['automated_strike'].mean():.4f}")

    return (
        data,
        catcher_values,
        pitch_categories,
        stand_categories,
        throws_categories,
    )


# BUILD BAYESIAN MODEL
def build_model(
    data, catcher_values, pitch_categories, stand_categories, throws_categories
):

    print("\nBuilding Model 4 Bayesian hierarchical model...")

    y = data["called_strike"].astype(int).values

    catcher_idx = data["catcher_idx"].astype(int).values
    season = data["season_idx"].astype(int).values
    pitch_type = data["pitch_type_idx"].astype(int).values
    stand = data["stand_idx"].astype(int).values
    throws = data["throws_idx"].astype(int).values

    automated_strike = data["automated_strike"].astype(float).values

    velocity = data["release_speed_std"].astype(float).values
    pfx_x = data["pfx_x_std"].astype(float).values
    pfx_z = data["pfx_z_std"].astype(float).values
    balls = data["balls_std"].astype(float).values
    strikes = data["strikes_std"].astype(float).values

    n_catchers = len(catcher_values)
    n_pitch_types = len(pitch_categories)
    n_stands = len(stand_categories)
    n_throws = len(throws_categories)

    print("\nModel dimensions:")
    print(f"  Catchers:    {n_catchers}")
    print(f"  Pitch types: {n_pitch_types}")
    print(f"  Bat sides:   {n_stands}")
    print(f"  Pitch sides: {n_throws}")

    with pm.Model() as model:

        intercept = pm.Normal("intercept", mu=0, sigma=1)

        # TRUE ZONE STATUS
        beta_automated_strike = pm.Normal(
            "beta_automated_strike", mu=0, sigma=2
        )

        beta_velocity = pm.Normal("beta_velocity", mu=0, sigma=0.5)
        beta_pfx_x = pm.Normal("beta_pfx_x", mu=0, sigma=0.5)
        beta_pfx_z = pm.Normal("beta_pfx_z", mu=0, sigma=0.5)
        beta_balls = pm.Normal("beta_balls", mu=0, sigma=0.5)
        beta_strikes = pm.Normal("beta_strikes", mu=0, sigma=0.5)

        if n_pitch_types > 1:
            pitch_type_effect = pm.Normal(
                "pitch_type_effect", mu=0, sigma=0.30, shape=n_pitch_types - 1
            )

        if n_stands > 1:
            stand_effect = pm.Normal(
                "stand_effect", mu=0, sigma=0.30, shape=n_stands - 1
            )

        if n_throws > 1:
            throws_effect = pm.Normal(
                "throws_effect", mu=0, sigma=0.30, shape=n_throws - 1
            )

        season_effect = pm.Normal("season_effect", mu=0, sigma=0.20)

        catcher_sd = pm.HalfNormal("catcher_sd", sigma=0.10)
        catcher_raw = pm.Normal("catcher_raw", mu=0, sigma=1, shape=n_catchers)
        catcher_effect = pm.Deterministic(
            "catcher_effect", catcher_raw * catcher_sd
        )

        catcher_season_sd = pm.HalfNormal("catcher_season_sd", sigma=0.075)
        catcher_season_raw = pm.Normal(
            "catcher_season_raw", mu=0, sigma=1, shape=n_catchers
        )
        catcher_season_effect = pm.Deterministic(
            "catcher_season_effect", catcher_season_raw * catcher_season_sd
        )

        eta = (
            intercept
            + beta_automated_strike * automated_strike
            + beta_velocity * velocity
            + beta_pfx_x * pfx_x
            + beta_pfx_z * pfx_z
            + beta_balls * balls
            + beta_strikes * strikes
        )

        if n_pitch_types > 1:
            eta += pm.math.switch(
                pitch_type == 0, 0, pitch_type_effect[pitch_type - 1]
            )

        if n_stands > 1:
            eta += pm.math.switch(stand == 0, 0, stand_effect[stand - 1])

        if n_throws > 1:
            eta += pm.math.switch(throws == 0, 0, throws_effect[throws - 1])

        eta += season_effect * season
        eta += catcher_effect[catcher_idx]
        eta += catcher_season_effect[catcher_idx] * season

        probability = pm.math.sigmoid(eta)

        pm.Bernoulli("called_strike", p=probability, observed=y)

    return model


# SAMPLE MODEL
def sample_model(model):
    print(f"\nChains: {CHAINS}")
    print(f"Tuning draws: {TUNE}")
    print(f"Posterior draws: {DRAWS}")
    print(f"Target accept: {TARGET_ACCEPT}")
    print(f"Max treedepth: {MAX_TREEDEPTH}")
    print(f"Cores: {CORES}")
    print("\nStarting NUTS...")

    with model:
        trace = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            cores=CORES,
            random_seed=RANDOM_SEED,
            target_accept=TARGET_ACCEPT,
            max_treedepth=MAX_TREEDEPTH,
            nuts_sampler="nutpie",
            return_inferencedata=True,
            progressbar=True,
            compute_convergence_checks=True,
        )

    print("\nSampling complete.")

    return trace


# DIAGNOSTICS
def diagnostics(trace):
    divergences = int(trace.sample_stats["diverging"].sum().values)
    total_draws = CHAINS * DRAWS
    divergence_rate = divergences / total_draws

    print("\nDivergences:")
    print(f"  {divergences:,} of {total_draws:,}")
    print(f"  Rate: {divergence_rate:.2%}")
    print("  GOOD: No divergences." if divergences == 0 else "  WARNING: Divergences detected.")

    summary_vars = [
        "intercept", "beta_automated_strike", "season_effect",
        "catcher_sd", "catcher_season_sd",
    ]
    summary = az.summary(trace, var_names=summary_vars, round_to=4)
    print("\nKey parameter diagnostics:")
    print(summary)

    full_summary = az.summary(trace, round_to=4)
    max_rhat = full_summary["r_hat"].max()
    min_ess = full_summary["ess_bulk"].min()

    print(f"\nMaximum R-hat: {max_rhat:.4f}")
    print(f"Minimum bulk ESS: {min_ess:.1f}")

    catcher_summary = az.summary(
        trace, var_names=["catcher_effect", "catcher_season_effect"], round_to=4
    )
    catcher_max_rhat = catcher_summary["r_hat"].max()
    catcher_min_ess = catcher_summary["ess_bulk"].min()

    print("\nCatcher-specific diagnostics:")
    print(f"  Maximum catcher R-hat: {catcher_max_rhat:.4f}")
    print(f"  Minimum catcher bulk ESS: {catcher_min_ess:.1f}")

    catcher_converged = (
        divergences == 0
        and catcher_max_rhat <= MAX_RHAT
        and catcher_min_ess >= MIN_CATCHER_ESS
    )

    print("\n" + "-" * 60)
    print("MODEL 4 CATCHER STATUS: " + ("GOOD" if catcher_converged else "NOT READY"))
    print("-" * 60)

    return summary, catcher_summary, catcher_converged

# EXTRACT CATCHER EFFECTS
def extract_catcher_effects(trace, catcher_values):
    print("\nExtracting Model 4 catcher effects...")

    catcher_samples = (
        trace.posterior["catcher_effect"].stack(sample=("chain", "draw")).values
    )
    season_samples = (
        trace.posterior["catcher_season_effect"]
        .stack(sample=("chain", "draw"))
        .values
    )

    results = []
    for i, catcher in enumerate(catcher_values):

        effect_2025 = catcher_samples[i]
        effect_2026 = catcher_samples[i] + season_samples[i]
        change = effect_2026 - effect_2025

        results.append({
            "catcher": catcher,
            "effect_2025": np.mean(effect_2025),
            "effect_2025_lower": np.quantile(effect_2025, 0.025),
            "effect_2025_upper": np.quantile(effect_2025, 0.975),
            "effect_2026": np.mean(effect_2026),
            "effect_2026_lower": np.quantile(effect_2026, 0.025),
            "effect_2026_upper": np.quantile(effect_2026, 0.975),
            "change": np.mean(change),
            "change_lower": np.quantile(change, 0.025),
            "change_upper": np.quantile(change, 0.975),
            "probability_increased": np.mean(change > 0),
        })

    return pd.DataFrame(results)


# CONVERT CATCHER EFFECTS DIRECTLY TO FRAMING RUNS
def add_runs_estimate(results, data):
    pitch_counts_2025 = (
        data[data["season"] == 2025]["catcher"].value_counts()
    )
    pitch_counts_2026 = (
        data[data["season"] == 2026]["catcher"].value_counts()
    )

    results = results.copy()
    results["pitches_2025"] = results["catcher"].map(pitch_counts_2025).fillna(0)
    results["pitches_2026"] = results["catcher"].map(pitch_counts_2026).fillna(0)

    marginal_effect_factor = 0.25  # sigmoid'(0) upper bound

    results["extra_strikes_2025"] = (
        results["effect_2025"] * marginal_effect_factor * results["pitches_2025"]
    )
    results["extra_strikes_2026"] = (
        results["effect_2026"] * marginal_effect_factor * results["pitches_2026"]
    )

    results["framing_runs_2025"] = results["extra_strikes_2025"] * RUNS_PER_STRIKE
    results["framing_runs_2026"] = results["extra_strikes_2026"] * RUNS_PER_STRIKE

    return results


# PLOT
def plot_results(results):
    plot_data = results.sort_values("change").copy()
    plt.figure(figsize=(10, 12))
    y = np.arange(len(plot_data))
    lower_error = plot_data["change"] - plot_data["change_lower"]
    upper_error = plot_data["change_upper"] - plot_data["change"]
    plt.errorbar(plot_data["change"], y, xerr=[lower_error, upper_error], fmt="o")
    plt.axvline(0, linestyle="--")
    plt.yticks(y, plot_data["catcher"])
    plt.xlabel("Change in Catcher Effect (2026 - 2025, log-odds)")
    plt.ylabel("Catcher ID")
    plt.title("Model 4: Catcher Framing Value vs. True Automated Zone")
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=200)
    plt.close()

    print(f"\nSaved plot to:\n  {PLOT_FILE}")


def print_results(results):
    print("MODEL 4 RESULTS -- FRAMING VALUE ELIMINATED BY ABS")

    print(
        "\nPositive framing_runs_2026 = catcher currently GAINS strikes "
        "beyond the true zone -> LOSES this value under ABS."
    )
    print(
        "Negative framing_runs_2026 = catcher currently LOSES strikes "
        "relative to the true zone -> GAINS this value under ABS."
    )

    print(f"\nTotal 2026 framing runs (league-wide): "
          f"{results['framing_runs_2026'].sum():.1f}")

    print("\nBiggest LOSERS under ABS (best current framers, 2026):")
    print(
        results.sort_values("framing_runs_2026", ascending=False)
        .head(15)[["catcher", "pitches_2026", "framing_runs_2026"]]
        .round(3)
        .to_string(index=False)
    )

    print("\nBiggest GAINERS under ABS (worst current framers, 2026):")
    print(
        results.sort_values("framing_runs_2026", ascending=True)
        .head(15)[["catcher", "pitches_2026", "framing_runs_2026"]]
        .round(3)
        .to_string(index=False)
    )


def save_trace(trace):
    print("\nSaving Model 4 Bayesian trace...")
    try:
        trace.to_netcdf(TRACE_FILE)
        print(f"Saved trace to:\n  {TRACE_FILE}")
    except Exception as error:
        print(f"\nWARNING: Could not save NetCDF trace. Reason: {error}")
        with open(FALLBACK_TRACE_FILE, "wb") as file:
            pickle.dump(trace, file)
        print(f"Saved fallback trace to:\n  {FALLBACK_TRACE_FILE}")

def main():

    data = load_data()

    (
        data,
        catcher_values,
        pitch_categories,
        stand_categories,
        throws_categories,
    ) = prepare_data(data)

    model = build_model(
        data, catcher_values, pitch_categories, stand_categories, throws_categories
    )

    trace = sample_model(model)

    summary, catcher_summary, catcher_converged = diagnostics(trace)

    save_trace(trace)

    if not catcher_converged:
        print("\nMODEL 4 CATCHER EFFECTS DID NOT PASS CONVERGENCE CHECKS.")
        print("Trace was saved for debugging. No results will be produced.")
        return

    results = extract_catcher_effects(trace, catcher_values)
    results = add_runs_estimate(results, data)

    results.to_csv(RESULTS_FILE, index=False)
    print(f"\nSaved Model 4 catcher framing values to:\n  {RESULTS_FILE}")

    print_results(results)
    plot_results(results)

    print("\n")
    print("MODEL 4 COMPLETE")

if __name__ == "__main__":
    main()
