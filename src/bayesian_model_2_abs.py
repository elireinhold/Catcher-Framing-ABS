from pathlib import Path
import pickle

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_FILE = (
    PROCESSED_DIR / "model_data_2025_2026.csv"
)

RESULTS_FILE = (
    PROCESSED_DIR / "bayesian_abs_catcher_effects_2025_2026.csv"
)

TRACE_FILE = (
    PROCESSED_DIR / "bayesian_abs_trace.nc"
)

FALLBACK_TRACE_FILE = (
    PROCESSED_DIR / "bayesian_abs_trace.pkl"
)

PLOT_FILE = (
    PROCESSED_DIR / "bayesian_abs_catcher_effects.png"
)


# ============================================================
# SAMPLING SETTINGS
# ============================================================

DRAWS = 500
TUNE = 1500

CHAINS = 4
CORES = 1

RANDOM_SEED = 42

TARGET_ACCEPT = 0.99

MAX_TREEDEPTH = 14


# ============================================================
# CATCHER FILTER
# ============================================================

MIN_CATCHER_PITCHES = 100


# ============================================================
# DEVELOPMENT SAMPLE
# ============================================================

MAX_MODEL_PITCHES = 100_000


# ============================================================
# AUTOMATED STRIKE ZONE
# ============================================================

# MLB plate width is 17 inches.
#
# 17 inches / 12 = 1.4167 feet
#
# Therefore the horizontal half-width is:
#
# 1.4167 / 2 = 0.7083 feet

PLATE_HALF_WIDTH = 17 / 24


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 60)
    print("MODEL 2: COUNTERFACTUAL AUTOMATED STRIKE ZONE")
    print("=" * 60)

    print("\nLoading data...")

    data = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Loaded {len(data):,} pitches."
    )

    return data


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(data):

    print(
        "\nPreparing Model 2 data..."
    )

    data = data.copy()

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    data["game_date"] = pd.to_datetime(
        data["game_date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Season
    # --------------------------------------------------------

    data["season"] = pd.to_numeric(
        data["season"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Keep only 2025 and 2026
    # --------------------------------------------------------

    data = data[
        data["season"].isin([2025, 2026])
    ].copy()

    print(
        f"After season filtering: "
        f"{len(data):,} pitches"
    )

    # --------------------------------------------------------
    # Required variables
    #
    # NOTE:
    # plate_x, plate_z, sz_top, and sz_bot are required
    # because the counterfactual automated strike zone is
    # constructed from them.
    # --------------------------------------------------------

    required = [

        "called_strike",

        "catcher",

        "season",

        "plate_x",
        "plate_z",

        "sz_top",
        "sz_bot",

        "release_speed",

        "pfx_x",
        "pfx_z",

        "balls",
        "strikes",

        "pitch_type",

        "stand",
        "p_throws",

    ]

    before = len(data)

    data = data.dropna(
        subset=required
    ).copy()

    print(
        f"Removed {before - len(data):,} "
        "rows missing required variables."
    )

    # --------------------------------------------------------
    # Ensure numeric values
    # --------------------------------------------------------

    numeric_columns = [

        "called_strike",

        "catcher",

        "plate_x",
        "plate_z",

        "sz_top",
        "sz_bot",

        "release_speed",

        "pfx_x",
        "pfx_z",

        "balls",
        "strikes",

    ]

    for column in numeric_columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.dropna(
        subset=numeric_columns
    ).copy()

    # ========================================================
    # CONSTRUCT COUNTERFACTUAL AUTOMATED STRIKE ZONE
    # ========================================================

    print(
        "\nConstructing counterfactual automated strike zone..."
    )

    # --------------------------------------------------------
    # Horizontal zone
    #
    # A pitch is horizontally inside the plate if:
    #
    # -0.7083 <= plate_x <= 0.7083
    #
    # --------------------------------------------------------

    horizontal_in_zone = (
        data["plate_x"].abs()
        <= PLATE_HALF_WIDTH
    )

    # --------------------------------------------------------
    # Vertical zone
    #
    # Statcast supplies batter-specific:
    #
    # sz_bot = bottom of strike zone
    # sz_top = top of strike zone
    #
    # --------------------------------------------------------

    vertical_in_zone = (
        (data["plate_z"] >= data["sz_bot"])
        &
        (data["plate_z"] <= data["sz_top"])
    )

    # --------------------------------------------------------
    # Automated strike
    # --------------------------------------------------------

    data["automated_strike"] = (
        horizontal_in_zone
        &
        vertical_in_zone
    ).astype(int)

    # --------------------------------------------------------
    # Umpire disagreement
    #
    # 1 = umpire call differs from automated zone
    # 0 = umpire and automated zone agree
    # --------------------------------------------------------

    data["umpire_disagreement"] = (
        data["called_strike"].astype(int)
        != data["automated_strike"]
    ).astype(int)

    # --------------------------------------------------------
    # Print automated-zone statistics
    # --------------------------------------------------------

    automated_rate = (
        data["automated_strike"].mean()
    )

    umpire_rate = (
        data["called_strike"].mean()
    )

    disagreement_rate = (
        data["umpire_disagreement"].mean()
    )

    print(
        "\nCounterfactual automated-zone statistics:"
    )

    print(
        f"  Umpire called-strike rate: "
        f"{umpire_rate:.4f}"
    )

    print(
        f"  Automated strike rate:     "
        f"{automated_rate:.4f}"
    )

    print(
        f"  Umpire disagreement rate:  "
        f"{disagreement_rate:.4f}"
    )

    # ========================================================
    # CHRONOLOGICAL ORDER
    # ========================================================

    data = (
        data
        .sort_values("game_date")
        .reset_index(drop=True)
    )

    # ========================================================
    # CATCHER FILTER
    # ========================================================

    catcher_counts = (
        data["catcher"]
        .value_counts()
    )

    valid_catchers = catcher_counts[
        catcher_counts >= MIN_CATCHER_PITCHES
    ].index

    data = data[
        data["catcher"].isin(valid_catchers)
    ].copy()

    print(
        f"\nCatchers with >= "
        f"{MIN_CATCHER_PITCHES} pitches: "
        f"{len(valid_catchers)}"
    )

    # --------------------------------------------------------
    # Catcher IDs
    # --------------------------------------------------------

    data["catcher"] = pd.to_numeric(
        data["catcher"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["catcher"]
    ).copy()

    data["catcher"] = (
        data["catcher"]
        .astype(int)
    )

    # ========================================================
    # DEVELOPMENT SAMPLE
    # ========================================================

    if (
        MAX_MODEL_PITCHES is not None
        and len(data) > MAX_MODEL_PITCHES
    ):

        print("\nDataset is large.")

        print(
            f"Randomly sampling "
            f"{MAX_MODEL_PITCHES:,} pitches "
            "for Bayesian model development..."
        )

        sampled_parts = []

        for season in [2025, 2026]:

            season_data = data[
                data["season"] == season
            ].copy()

            if len(season_data) == 0:
                continue

            proportion = (
                len(season_data)
                / len(data)
            )

            n_sample = int(
                MAX_MODEL_PITCHES
                * proportion
            )

            n_sample = min(
                n_sample,
                len(season_data)
            )

            sampled = season_data.sample(
                n=n_sample,
                random_state=RANDOM_SEED
            )

            sampled_parts.append(
                sampled
            )

        data = pd.concat(
            sampled_parts,
            ignore_index=True
        )

        data = (
            data
            .sort_values("game_date")
            .reset_index(drop=True)
        )

        print(
            f"Development dataset: "
            f"{len(data):,} pitches"
        )

    # ========================================================
    # ENCODE CATCHERS
    # ========================================================

    catcher_values = sorted(
        data["catcher"].unique()
    )

    catcher_map = {
        catcher: i
        for i, catcher
        in enumerate(catcher_values)
    }

    data["catcher_idx"] = (
        data["catcher"]
        .map(catcher_map)
        .astype(int)
    )

    # --------------------------------------------------------
    # 2026 catcher sample sizes
    # --------------------------------------------------------

    catcher_2026_counts = (
        data.loc[
            data["season"] == 2026,
            "catcher"
        ]
        .value_counts()
        .reindex(
            catcher_values,
            fill_value=0
        )
    )

    thin_catchers = catcher_2026_counts[
        catcher_2026_counts < MIN_CATCHER_PITCHES
    ]

    if len(thin_catchers) > 0:

        print(
            f"\nNOTE: {len(thin_catchers)} catcher(s) "
            "have fewer than "
            f"{MIN_CATCHER_PITCHES} pitches in 2026."
        )

        print(
            "Their catcher × season effect will be "
            "pulled toward the prior."
        )

    # ========================================================
    # CATEGORICAL VARIABLES
    # ========================================================

    pitch_categories = sorted(
        data["pitch_type"]
        .astype(str)
        .unique()
    )

    pitch_map = {
        value: i
        for i, value
        in enumerate(pitch_categories)
    }

    data["pitch_type_idx"] = (
        data["pitch_type"]
        .astype(str)
        .map(pitch_map)
        .astype(int)
    )

    # --------------------------------------------------------
    # Batter handedness
    # --------------------------------------------------------

    stand_categories = sorted(
        data["stand"]
        .astype(str)
        .unique()
    )

    stand_map = {
        value: i
        for i, value
        in enumerate(stand_categories)
    }

    data["stand_idx"] = (
        data["stand"]
        .astype(str)
        .map(stand_map)
        .astype(int)
    )

    # --------------------------------------------------------
    # Pitcher handedness
    # --------------------------------------------------------

    throws_categories = sorted(
        data["p_throws"]
        .astype(str)
        .unique()
    )

    throws_map = {
        value: i
        for i, value
        in enumerate(throws_categories)
    }

    data["throws_idx"] = (
        data["p_throws"]
        .astype(str)
        .map(throws_map)
        .astype(int)
    )

    # ========================================================
    # SEASON
    # ========================================================

    data["season_idx"] = (
        data["season"] == 2026
    ).astype(int)

    # ========================================================
    # STANDARDIZE NON-ZONE CONTINUOUS PREDICTORS
    #
    # We intentionally DO NOT include plate_x / plate_z
    # because they directly define automated_strike.
    # ========================================================

    continuous = [

        "release_speed",

        "pfx_x",

        "pfx_z",

        "balls",

        "strikes",

    ]

    print(
        "\nStandardizing continuous predictors..."
    )

    for column in continuous:

        mean = data[column].mean()
        std = data[column].std()

        if (
            not np.isfinite(std)
            or std == 0
        ):
            std = 1.0

        data[column + "_std"] = (
            (data[column] - mean)
            / std
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\nFinal Model 2 dataset:"
    )

    print(
        f"  Pitches:  {len(data):,}"
    )

    print(
        f"  Catchers: {data['catcher'].nunique()}"
    )

    print(
        f"  2025:     "
        f"{(data['season'] == 2025).sum():,}"
    )

    print(
        f"  2026:     "
        f"{(data['season'] == 2026).sum():,}"
    )

    print(
        f"  Automated strikes: "
        f"{data['automated_strike'].mean():.4f}"
    )

    print(
        f"  Disagreements: "
        f"{data['umpire_disagreement'].mean():.4f}"
    )

    return (
        data,
        catcher_values,
        pitch_categories,
        stand_categories,
        throws_categories,
    )


# ============================================================
# BUILD BAYESIAN MODEL
# ============================================================

def build_model(
    data,
    catcher_values,
    pitch_categories,
    stand_categories,
    throws_categories,
):

    print(
        "\nBuilding Model 2 Bayesian hierarchical model..."
    )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    y = (
        data["umpire_disagreement"]
        .astype(int)
        .values
    )

    # --------------------------------------------------------
    # Indices
    # --------------------------------------------------------

    catcher_idx = (
        data["catcher_idx"]
        .astype(int)
        .values
    )

    season = (
        data["season_idx"]
        .astype(int)
        .values
    )

    pitch_type = (
        data["pitch_type_idx"]
        .astype(int)
        .values
    )

    stand = (
        data["stand_idx"]
        .astype(int)
        .values
    )

    throws = (
        data["throws_idx"]
        .astype(int)
        .values
    )

    # --------------------------------------------------------
    # Pitch characteristics
    # --------------------------------------------------------

    velocity = (
        data["release_speed_std"]
        .astype(float)
        .values
    )

    pfx_x = (
        data["pfx_x_std"]
        .astype(float)
        .values
    )

    pfx_z = (
        data["pfx_z_std"]
        .astype(float)
        .values
    )

    balls = (
        data["balls_std"]
        .astype(float)
        .values
    )

    strikes = (
        data["strikes_std"]
        .astype(float)
        .values
    )

    # ========================================================
    # DIMENSIONS
    # ========================================================

    n_catchers = len(
        catcher_values
    )

    n_pitch_types = len(
        pitch_categories
    )

    n_stands = len(
        stand_categories
    )

    n_throws = len(
        throws_categories
    )

    print(
        "\nModel dimensions:"
    )

    print(
        f"  Catchers:    {n_catchers}"
    )

    print(
        f"  Pitch types: {n_pitch_types}"
    )

    print(
        f"  Bat sides:   {n_stands}"
    )

    print(
        f"  Pitch sides: {n_throws}"
    )

    # ========================================================
    # MODEL
    # ========================================================

    with pm.Model() as model:

        # ====================================================
        # INTERCEPT
        # ====================================================

        intercept = pm.Normal(
            "intercept",
            mu=0,
            sigma=1
        )

        # ====================================================
        # PITCH CHARACTERISTICS
        # ====================================================

        beta_velocity = pm.Normal(
            "beta_velocity",
            mu=0,
            sigma=0.5
        )

        beta_pfx_x = pm.Normal(
            "beta_pfx_x",
            mu=0,
            sigma=0.5
        )

        beta_pfx_z = pm.Normal(
            "beta_pfx_z",
            mu=0,
            sigma=0.5
        )

        # ====================================================
        # COUNT
        # ====================================================

        beta_balls = pm.Normal(
            "beta_balls",
            mu=0,
            sigma=0.5
        )

        beta_strikes = pm.Normal(
            "beta_strikes",
            mu=0,
            sigma=0.5
        )

        # ====================================================
        # PITCH TYPE
        # ====================================================

        if n_pitch_types > 1:

            pitch_type_effect = pm.Normal(
                "pitch_type_effect",
                mu=0,
                sigma=0.35,
                shape=n_pitch_types - 1
            )

        # ====================================================
        # BATTER HANDEDNESS
        # ====================================================

        if n_stands > 1:

            stand_effect = pm.Normal(
                "stand_effect",
                mu=0,
                sigma=0.35,
                shape=n_stands - 1
            )

        # ====================================================
        # PITCHER HANDEDNESS
        # ====================================================

        if n_throws > 1:

            throws_effect = pm.Normal(
                "throws_effect",
                mu=0,
                sigma=0.35,
                shape=n_throws - 1
            )

        # ====================================================
        # SEASON
        # ====================================================

        season_effect = pm.Normal(
            "season_effect",
            mu=0,
            sigma=0.25
        )

        # ====================================================
        # CATCHER EFFECT
        #
        # Non-centered parameterization
        # ====================================================

        catcher_sd = pm.HalfNormal(
            "catcher_sd",
            sigma=0.15
        )

        catcher_raw = pm.Normal(
            "catcher_raw",
            mu=0,
            sigma=1,
            shape=n_catchers
        )

        catcher_effect = pm.Deterministic(
            "catcher_effect",
            catcher_raw * catcher_sd
        )

        # ====================================================
        # CATCHER × 2026
        #
        # This is the key Model 2 research quantity.
        # ====================================================

        catcher_season_sd = pm.HalfNormal(
            "catcher_season_sd",
            sigma=0.10
        )

        catcher_season_raw = pm.Normal(
            "catcher_season_raw",
            mu=0,
            sigma=1,
            shape=n_catchers
        )

        catcher_season_effect = pm.Deterministic(
            "catcher_season_effect",
            catcher_season_raw
            * catcher_season_sd
        )

        # ====================================================
        # LINEAR PREDICTOR
        # ====================================================

        eta = (

            intercept

            # Pitch characteristics
            + beta_velocity * velocity
            + beta_pfx_x * pfx_x
            + beta_pfx_z * pfx_z

            # Count
            + beta_balls * balls
            + beta_strikes * strikes
        )

        # ----------------------------------------------------
        # Pitch type
        # ----------------------------------------------------

        if n_pitch_types > 1:

            eta += pm.math.switch(
                pitch_type == 0,
                0,
                pitch_type_effect[
                    pitch_type - 1
                ]
            )

        # ----------------------------------------------------
        # Batter handedness
        # ----------------------------------------------------

        if n_stands > 1:

            eta += pm.math.switch(
                stand == 0,
                0,
                stand_effect[
                    stand - 1
                ]
            )

        # ----------------------------------------------------
        # Pitcher handedness
        # ----------------------------------------------------

        if n_throws > 1:

            eta += pm.math.switch(
                throws == 0,
                0,
                throws_effect[
                    throws - 1
                ]
            )

        # ----------------------------------------------------
        # Season
        # ----------------------------------------------------

        eta += (
            season_effect
            * season
        )

        # ----------------------------------------------------
        # Catcher baseline
        # ----------------------------------------------------

        eta += catcher_effect[
            catcher_idx
        ]

        # ----------------------------------------------------
        # Catcher × 2026
        # ----------------------------------------------------

        eta += (
            catcher_season_effect[
                catcher_idx
            ]
            * season
        )

        # ====================================================
        # LIKELIHOOD
        # ====================================================

        probability = pm.math.sigmoid(
            eta
        )

        pm.Bernoulli(
            "umpire_disagreement",
            p=probability,
            observed=y
        )

    return model


# ============================================================
# SAMPLE MODEL
# ============================================================

def sample_model(model):

    print(
        "\n" + "=" * 60
    )

    print(
        "STARTING MODEL 2 BAYESIAN SAMPLING"
    )

    print(
        "=" * 60
    )

    print(
        f"\nChains: {CHAINS}"
    )

    print(
        f"Tuning draws: {TUNE}"
    )

    print(
        f"Posterior draws: {DRAWS}"
    )

    print(
        f"Target accept: {TARGET_ACCEPT}"
    )

    print(
        f"Max treedepth: {MAX_TREEDEPTH}"
    )

    print(
        f"Cores: {CORES}"
    )

    print(
        f"Model sample: "
        f"{MAX_MODEL_PITCHES:,} pitches"
    )

    print(
        "\nStarting NUTS..."
    )

    with model:

        trace = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            random_seed=RANDOM_SEED,
            target_accept=TARGET_ACCEPT,
            max_treedepth=MAX_TREEDEPTH,
            nuts_sampler="nutpie",
            return_inferencedata=True,
            progressbar=True,
            compute_convergence_checks=True
        )

    print(
        "\nSampling complete."
    )

    return trace


# ============================================================
# DIAGNOSTICS
# ============================================================

def diagnostics(trace):

    print(
        "\n" + "=" * 60
    )

    print(
        "MODEL 2 BAYESIAN DIAGNOSTICS"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # Divergences
    # --------------------------------------------------------

    divergences = int(
        trace.sample_stats[
            "diverging"
        ]
        .sum()
        .values
    )

    total_draws = (
        CHAINS * DRAWS
    )

    divergence_rate = (
        divergences / total_draws
    )

    print(
        "\nDivergences:"
    )

    print(
        f"  {divergences:,} "
        f"of {total_draws:,}"
    )

    print(
        f"  Rate: "
        f"{divergence_rate:.2%}"
    )

    if divergences == 0:

        print(
            "  GOOD: No divergences."
        )

    else:

        print(
            "  WARNING: Divergences detected."
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_vars = [

        "intercept",

        "season_effect",

        "catcher_sd",

        "catcher_season_sd",

    ]

    summary = az.summary(
        trace,
        var_names=summary_vars,
        round_to=4
    )

    print(
        "\nKey parameter diagnostics:"
    )

    print(
        summary
    )

    # --------------------------------------------------------
    # R-hat
    # --------------------------------------------------------

    max_rhat = (
        summary["r_hat"]
        .max()
    )

    print(
        f"\nMaximum R-hat: "
        f"{max_rhat:.4f}"
    )

    if max_rhat <= 1.01:

        print(
            "  GOOD: R-hat is acceptable."
        )

    else:

        print(
            "  WARNING: "
            "R-hat indicates possible "
            "convergence problems."
        )

    # --------------------------------------------------------
    # ESS
    # --------------------------------------------------------

    min_ess = (
        summary["ess_bulk"]
        .min()
    )

    print(
        f"\nMinimum bulk ESS: "
        f"{min_ess:.1f}"
    )

    # --------------------------------------------------------
    # Overall status
    # --------------------------------------------------------

    converged = (
        divergences == 0
        and max_rhat <= 1.01
        and min_ess >= 100
    )

    print(
        "\n" + "-" * 60
    )

    if converged:

        print(
            "MODEL 2 STATUS: GOOD"
        )

        print(
            "The basic convergence checks passed."
        )

    else:

        print(
            "MODEL 2 STATUS: NOT READY"
        )

        print(
            "Do not interpret catcher effects yet."
        )

    print(
        "-" * 60
    )

    return summary, converged


# ============================================================
# EXTRACT CATCHER EFFECTS
# ============================================================

def extract_catcher_effects(
    trace,
    catcher_values
):

    print(
        "\nExtracting Model 2 catcher effects..."
    )

    catcher_samples = (
        trace.posterior[
            "catcher_effect"
        ]
        .stack(
            sample=("chain", "draw")
        )
        .values
    )

    season_samples = (
        trace.posterior[
            "catcher_season_effect"
        ]
        .stack(
            sample=("chain", "draw")
        )
        .values
    )

    results = []

    for i, catcher in enumerate(
        catcher_values
    ):

        # ----------------------------------------------------
        # Baseline catcher effect
        # ----------------------------------------------------

        effect_2025 = (
            catcher_samples[i]
        )

        # ----------------------------------------------------
        # 2026 catcher effect
        # ----------------------------------------------------

        effect_2026 = (
            catcher_samples[i]
            + season_samples[i]
        )

        # ----------------------------------------------------
        # 2025 → 2026 change
        # ----------------------------------------------------

        change = (
            effect_2026
            - effect_2025
        )

        results.append({

            "catcher": catcher,

            "effect_2025":
                np.mean(effect_2025),

            "effect_2025_lower":
                np.quantile(
                    effect_2025,
                    0.025
                ),

            "effect_2025_upper":
                np.quantile(
                    effect_2025,
                    0.975
                ),

            "effect_2026":
                np.mean(effect_2026),

            "effect_2026_lower":
                np.quantile(
                    effect_2026,
                    0.025
                ),

            "effect_2026_upper":
                np.quantile(
                    effect_2026,
                    0.975
                ),

            "change":
                np.mean(change),

            "change_lower":
                np.quantile(
                    change,
                    0.025
                ),

            "change_upper":
                np.quantile(
                    change,
                    0.975
                ),

            "probability_increased":
                np.mean(
                    change > 0
                ),

        })

    return pd.DataFrame(
        results
    )


# ============================================================
# PLOT RESULTS
# ============================================================

def plot_results(results):

    plot_data = (
        results
        .sort_values("change")
        .copy()
    )

    plt.figure(
        figsize=(10, 12)
    )

    y = np.arange(
        len(plot_data)
    )

    lower_error = (
        plot_data["change"]
        - plot_data["change_lower"]
    )

    upper_error = (
        plot_data["change_upper"]
        - plot_data["change"]
    )

    plt.errorbar(
        plot_data["change"],
        y,
        xerr=[
            lower_error,
            upper_error
        ],
        fmt="o"
    )

    plt.axvline(
        0,
        linestyle="--"
    )

    plt.yticks(
        y,
        plot_data["catcher"]
    )

    plt.xlabel(
        "Change in Catcher Effect "
        "(2026 - 2025, log-odds)"
    )

    plt.ylabel(
        "Catcher ID"
    )

    plt.title(
        "Model 2: Catcher Effect on "
        "Umpire–Automated Zone Disagreement"
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_FILE,
        dpi=200
    )

    plt.close()

    print(
        f"\nSaved plot to:\n"
        f"  {PLOT_FILE}"
    )


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(results):

    print(
        "\n" + "=" * 60
    )

    print(
        "MODEL 2 CATCHER RESULTS"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # Highest 2025 catcher effects
    # --------------------------------------------------------

    print(
        "\nHighest catcher effects in 2025:"
    )

    print(
        results
        .sort_values(
            "effect_2025",
            ascending=False
        )
        [[
            "catcher",
            "effect_2025",
            "effect_2025_lower",
            "effect_2025_upper"
        ]]
        .head(15)
        .round(4)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Highest 2026 catcher effects
    # --------------------------------------------------------

    print(
        "\nHighest catcher effects in 2026:"
    )

    print(
        results
        .sort_values(
            "effect_2026",
            ascending=False
        )
        [[
            "catcher",
            "effect_2026",
            "effect_2026_lower",
            "effect_2026_upper"
        ]]
        .head(15)
        .round(4)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Largest decreases
    # --------------------------------------------------------

    print(
        "\nLargest decreases from 2025 → 2026:"
    )

    print(
        results
        .sort_values(
            "change"
        )
        [[
            "catcher",
            "change",
            "change_lower",
            "change_upper",
            "probability_increased"
        ]]
        .head(15)
        .round(4)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Largest increases
    # --------------------------------------------------------

    print(
        "\nLargest increases from 2025 → 2026:"
    )

    print(
        results
        .sort_values(
            "change",
            ascending=False
        )
        [[
            "catcher",
            "change",
            "change_lower",
            "change_upper",
            "probability_increased"
        ]]
        .head(15)
        .round(4)
        .to_string(index=False)
    )


# ============================================================
# SAVE TRACE
# ============================================================

def save_trace(trace):

    print(
        "\nSaving Model 2 Bayesian trace..."
    )

    try:

        trace.to_netcdf(
            TRACE_FILE
        )

        print(
            f"Saved trace to:\n"
            f"  {TRACE_FILE}"
        )

    except Exception as error:

        print(
            "\nWARNING:"
        )

        print(
            "Could not save NetCDF trace."
        )

        print(
            f"Reason: {error}"
        )

        print(
            "\nSaving fallback pickle trace..."
        )

        with open(
            FALLBACK_TRACE_FILE,
            "wb"
        ) as file:

            pickle.dump(
                trace,
                file
            )

        print(
            f"Saved fallback trace to:\n"
            f"  {FALLBACK_TRACE_FILE}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    data = load_data()

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    (
        data,
        catcher_values,
        pitch_categories,
        stand_categories,
        throws_categories,
    ) = prepare_data(
        data
    )

    # --------------------------------------------------------
    # Build
    # --------------------------------------------------------

    model = build_model(
        data,
        catcher_values,
        pitch_categories,
        stand_categories,
        throws_categories,
    )

    # --------------------------------------------------------
    # Sample
    # --------------------------------------------------------

    trace = sample_model(
        model
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    summary, converged = diagnostics(
        trace
    )

    # --------------------------------------------------------
    # Stop if not converged
    # --------------------------------------------------------

    if not converged:

        print(
            "\n" + "=" * 60
        )

        print(
            "MODEL 2 DID NOT PASS "
            "CONVERGENCE CHECKS"
        )

        print(
            "=" * 60
        )

        print(
            "\nThe trace will still be saved "
            "for debugging."
        )

        save_trace(
            trace
        )

        print(
            "\nNo catcher rankings will be produced."
        )

        print(
            "Fix convergence before interpreting "
            "the model."
        )

        return

    # --------------------------------------------------------
    # Save trace
    # --------------------------------------------------------

    save_trace(
        trace
    )

    # --------------------------------------------------------
    # Extract catcher effects
    # --------------------------------------------------------

    results = extract_catcher_effects(
        trace,
        catcher_values
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results.to_csv(
        RESULTS_FILE,
        index=False
    )

    print(
        f"\nSaved Model 2 catcher effects to:\n"
        f"  {RESULTS_FILE}"
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print_results(
        results
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plot_results(
        results
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "MODEL 2 COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        "\nPrimary research quantity:"
    )

    print(
        "\nCatcher effect on "
        "umpire–automated-zone disagreement"
    )

    print(
        "\nSecondary research quantity:"
    )

    print(
        "\n2026 catcher effect"
        "\nminus"
        "\n2025 catcher effect"
    )

    print(
        "\nInterpretation:"
    )

    print(
        "\nA positive catcher effect means "
        "that catcher is associated with a "
        "higher probability that the umpire "
        "disagrees with the counterfactual "
        "automated strike zone."
    )

    print(
        "\nThe catcher × 2026 interaction "
        "measures how that relationship "
        "changed from 2025 to 2026."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()