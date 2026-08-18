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

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

INPUT_FILE = (
    PROCESSED_DIR
    / "model_data_2025_2026.csv"
)

RESULTS_FILE = (
    PROCESSED_DIR
    / "bayesian_model_3_abs_catcher_effects_2025_2026.csv"
)

TRACE_FILE = (
    PROCESSED_DIR
    / "bayesian_model_3_abs_trace.nc"
)

FALLBACK_TRACE_FILE = (
    PROCESSED_DIR
    / "bayesian_model_3_abs_trace.pkl"
)

PLOT_FILE = (
    PROCESSED_DIR
    / "bayesian_model_3_abs_catcher_effects.png"
)


# ============================================================
# SAMPLING SETTINGS
# ============================================================

DRAWS = 1000

TUNE = 2000

CHAINS = 4

CORES = 12

RANDOM_SEED = 42

TARGET_ACCEPT = 0.995

MAX_TREEDEPTH = 15


# ============================================================
# CATCHER FILTER
# ============================================================

MIN_CATCHER_PITCHES = 100


# ============================================================
# DEVELOPMENT SAMPLE
# ============================================================

MAX_MODEL_PITCHES = 100_000


# ============================================================
# CONVERGENCE THRESHOLDS
# ============================================================

MAX_RHAT = 1.01

MIN_ESS = 400

MIN_CATCHER_ESS = 400


# ============================================================
# AUTOMATED STRIKE ZONE
# ============================================================

# MLB plate width = 17 inches.
#
# 17 / 12 = 1.4167 feet
#
# Half-width:
#
# 1.4167 / 2 = 0.7083 feet

PLATE_HALF_WIDTH = 17 / 24


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 60)

    print(
        "MODEL 3: AUTOMATED STRIKE ZONE CATCHER EFFECT"
    )

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
        "\nPreparing Model 3 data..."
    )

    data = data.copy()

    # ========================================================
    # DATES
    # ========================================================

    data["game_date"] = pd.to_datetime(
        data["game_date"],
        errors="coerce"
    )

    # ========================================================
    # SEASON
    # ========================================================

    data["season"] = pd.to_numeric(
        data["season"],
        errors="coerce"
    )

    # ========================================================
    # KEEP 2025 + 2026
    # ========================================================

    data = data[
        data["season"].isin(
            [2025, 2026]
        )
    ].copy()

    print(
        f"After season filtering: "
        f"{len(data):,} pitches"
    )

    # ========================================================
    # REQUIRED VARIABLES
    # ========================================================

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

    # ========================================================
    # NUMERIC VARIABLES
    # ========================================================

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
    # COUNTERFACTUAL AUTOMATED STRIKE ZONE
    # ========================================================

    print(
        "\nConstructing counterfactual automated strike zone..."
    )

    # --------------------------------------------------------
    # Horizontal
    # --------------------------------------------------------

    horizontal_in_zone = (
        data["plate_x"].abs()
        <= PLATE_HALF_WIDTH
    )

    # --------------------------------------------------------
    # Vertical
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
    # 1 = umpire disagrees with automated zone
    # 0 = umpire agrees
    # --------------------------------------------------------

    data["umpire_disagreement"] = (
        data["called_strike"].astype(int)
        != data["automated_strike"]
    ).astype(int)

    # ========================================================
    # AUTOMATED-ZONE STATISTICS
    # ========================================================

    automated_rate = (
        data["automated_strike"].mean()
    )

    automated_strikes = int(
        data["automated_strike"].sum()
    )

    automated_balls = (
        len(data)
        - automated_strikes
    )

    print(
        "\nCounterfactual automated-zone statistics:"
    )

    print(
        f"  Automated strike rate: "
        f"{automated_rate:.4f}"
    )

    print(
        f"  Automated strikes: "
        f"{automated_strikes:,}"
    )

    print(
        f"  Automated balls: "
        f"{automated_balls:,}"
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
        catcher_counts
        >= MIN_CATCHER_PITCHES
    ].index

    data = data[
        data["catcher"].isin(
            valid_catchers
        )
    ].copy()

    print(
        f"\nCatchers with >= "
        f"{MIN_CATCHER_PITCHES} pitches: "
        f"{len(valid_catchers)}"
    )

    # ========================================================
    # CLEAN CATCHER IDs
    # ========================================================

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

        print(
            "\nDataset is large."
        )

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

    # ========================================================
    # 2026 CATCHER SAMPLE SIZES
    # ========================================================

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
        catcher_2026_counts
        < MIN_CATCHER_PITCHES
    ]

    if len(thin_catchers) > 0:

        print(
            f"\nNOTE: {len(thin_catchers)} "
            "catcher(s) have fewer than "
            f"{MIN_CATCHER_PITCHES} pitches in 2026."
        )

        print(
            "Their catcher × season effect "
            "will be partially pooled toward "
            "the prior."
        )

    # ========================================================
    # PITCH TYPE
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

    # ========================================================
    # BATTER HANDEDNESS
    # ========================================================

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

    # ========================================================
    # PITCHER HANDEDNESS
    # ========================================================

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
    # SEASON INDEX
    # ========================================================

    data["season_idx"] = (
        data["season"] == 2026
    ).astype(int)

    # ========================================================
    # STANDARDIZE CONTINUOUS PREDICTORS
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
        "\nFinal Model 3 dataset:"
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
        "\nBuilding Model 3 Bayesian hierarchical model..."
    )

    # ========================================================
    # RESPONSE
    # ========================================================

    y = (
        data["umpire_disagreement"]
        .astype(int)
        .values
    )

    # ========================================================
    # INDICES
    # ========================================================

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

    # ========================================================
    # PITCH CHARACTERISTICS
    # ========================================================

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
        #
        # Slightly stronger prior helps stabilize sparse
        # pitch-type categories.
        # ====================================================

        if n_pitch_types > 1:

            pitch_type_effect = pm.Normal(
                "pitch_type_effect",
                mu=0,
                sigma=0.30,
                shape=n_pitch_types - 1
            )

        # ====================================================
        # BATTER HANDEDNESS
        # ====================================================

        if n_stands > 1:

            stand_effect = pm.Normal(
                "stand_effect",
                mu=0,
                sigma=0.30,
                shape=n_stands - 1
            )

        # ====================================================
        # PITCHER HANDEDNESS
        # ====================================================

        if n_throws > 1:

            throws_effect = pm.Normal(
                "throws_effect",
                mu=0,
                sigma=0.30,
                shape=n_throws - 1
            )

        # ====================================================
        # SEASON
        # ====================================================

        season_effect = pm.Normal(
            "season_effect",
            mu=0,
            sigma=0.20
        )

        # ====================================================
        # CATCHER BASELINE
        #
        # Stronger partial pooling than previous version.
        # ====================================================

        catcher_sd = pm.HalfNormal(
            "catcher_sd",
            sigma=0.10
        )

        catcher_raw = pm.Normal(
            "catcher_raw",
            mu=0,
            sigma=1,
            shape=n_catchers
        )

        catcher_effect = pm.Deterministic(
            "catcher_effect",
            catcher_raw
            * catcher_sd
        )

        # ====================================================
        # CATCHER × 2026
        #
        # Stronger partial pooling.
        # ====================================================

        catcher_season_sd = pm.HalfNormal(
            "catcher_season_sd",
            sigma=0.075
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

        # ====================================================
        # PITCH TYPE
        # ====================================================

        if n_pitch_types > 1:

            eta += pm.math.switch(
                pitch_type == 0,
                0,
                pitch_type_effect[
                    pitch_type - 1
                ]
            )

        # ====================================================
        # BATTER HANDEDNESS
        # ====================================================

        if n_stands > 1:

            eta += pm.math.switch(
                stand == 0,
                0,
                stand_effect[
                    stand - 1
                ]
            )

        # ====================================================
        # PITCHER HANDEDNESS
        # ====================================================

        if n_throws > 1:

            eta += pm.math.switch(
                throws == 0,
                0,
                throws_effect[
                    throws - 1
                ]
            )

        # ====================================================
        # SEASON
        # ====================================================

        eta += (
            season_effect
            * season
        )

        # ====================================================
        # CATCHER
        # ====================================================

        eta += catcher_effect[
            catcher_idx
        ]

        # ====================================================
        # CATCHER × 2026
        # ====================================================

        eta += (
            catcher_season_effect[
                catcher_idx
            ]
            * season
        )

        # ====================================================
        # PROBABILITY
        # ====================================================

        probability = pm.math.sigmoid(
            eta
        )

        # ====================================================
        # LIKELIHOOD
        # ====================================================

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
        "STARTING MODEL 3 BAYESIAN SAMPLING"
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
            cores=CORES,
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
        "MODEL 3 BAYESIAN DIAGNOSTICS"
    )

    print(
        "=" * 60
    )

    # ========================================================
    # DIVERGENCES
    # ========================================================

    divergences = int(
        trace.sample_stats[
            "diverging"
        ]
        .sum()
        .values
    )

    total_draws = (
        CHAINS
        * DRAWS
    )

    divergence_rate = (
        divergences
        / total_draws
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

    # ========================================================
    # GENERAL MODEL PARAMETERS
    # ========================================================

    summary_vars = [

        "intercept",

        "beta_velocity",

        "beta_pfx_x",

        "beta_pfx_z",

        "beta_balls",

        "beta_strikes",

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

    # ========================================================
    # FULL MODEL R-HAT
    # ========================================================

    full_summary = az.summary(
        trace,
        round_to=4
    )

    max_rhat = (
        full_summary["r_hat"]
        .max()
    )

    worst_rhat_parameter = (
        full_summary["r_hat"]
        .idxmax()
    )

    print(
        f"\nMaximum R-hat: "
        f"{max_rhat:.4f}"
    )

    print(
        f"Worst parameter: "
        f"{worst_rhat_parameter}"
    )

    # ========================================================
    # FULL MODEL ESS
    # ========================================================

    min_ess = (
        full_summary["ess_bulk"]
        .min()
    )

    min_ess_parameter = (
        full_summary["ess_bulk"]
        .idxmin()
    )

    print(
        f"\nMinimum bulk ESS: "
        f"{min_ess:.1f}"
    )

    print(
        f"Lowest ESS parameter: "
        f"{min_ess_parameter}"
    )

    min_tail_ess = (
        full_summary["ess_tail"]
        .min()
    )

    print(
        f"\nMinimum tail ESS: "
        f"{min_tail_ess:.1f}"
    )

    # ========================================================
    # CATCHER-SPECIFIC DIAGNOSTICS
    # ========================================================

    catcher_summary = az.summary(
        trace,
        var_names=[
            "catcher_effect",
            "catcher_season_effect",
        ],
        round_to=4
    )

    catcher_max_rhat = (
        catcher_summary["r_hat"]
        .max()
    )

    catcher_min_ess = (
        catcher_summary["ess_bulk"]
        .min()
    )

    catcher_min_tail_ess = (
        catcher_summary["ess_tail"]
        .min()
    )

    print(
        "\nCatcher-specific diagnostics:"
    )

    print(
        f"  Maximum catcher R-hat: "
        f"{catcher_max_rhat:.4f}"
    )

    print(
        f"  Minimum catcher bulk ESS: "
        f"{catcher_min_ess:.1f}"
    )

    print(
        f"  Minimum catcher tail ESS: "
        f"{catcher_min_tail_ess:.1f}"
    )

    # ========================================================
    # IDENTIFY PROBLEMATIC CATCHER PARAMETERS
    # ========================================================

    bad_catcher_rhat = catcher_summary[
        catcher_summary["r_hat"]
        > MAX_RHAT
    ]

    bad_catcher_ess = catcher_summary[
        catcher_summary["ess_bulk"]
        < MIN_CATCHER_ESS
    ]

    if len(bad_catcher_rhat) > 0:

        print(
            "\nWARNING: Catcher parameters "
            "with R-hat > 1.01:"
        )

        print(
            bad_catcher_rhat[
                ["r_hat", "ess_bulk"]
            ].to_string()
        )

    else:

        print(
            "\nGOOD: All catcher parameters "
            "have R-hat <= 1.01."
        )

    if len(bad_catcher_ess) > 0:

        print(
            "\nWARNING: Catcher parameters "
            f"with bulk ESS < {MIN_CATCHER_ESS}:"
        )

        print(
            bad_catcher_ess[
                ["r_hat", "ess_bulk"]
            ].to_string()
        )

    else:

        print(
            f"GOOD: All catcher parameters "
            f"have bulk ESS >= {MIN_CATCHER_ESS}."
        )

    # ========================================================
    # CATCHER CONVERGENCE
    # ========================================================

    catcher_converged = (
        divergences == 0
        and catcher_max_rhat <= MAX_RHAT
        and catcher_min_ess >= MIN_CATCHER_ESS
    )

    # ========================================================
    # GENERAL MODEL CONVERGENCE
    # ========================================================

    model_converged = (
        divergences == 0
        and max_rhat <= MAX_RHAT
        and min_ess >= MIN_ESS
    )

    # ========================================================
    # STATUS
    # ========================================================

    print(
        "\n" + "-" * 60
    )

    if catcher_converged:

        print(
            "MODEL 3 CATCHER STATUS: GOOD"
        )

        print(
            "The catcher-specific posterior "
            "parameters passed the convergence checks."
        )

        if not model_converged:

            print(
                "\nNOTE:"
            )

            print(
                "Some non-catcher parameters may still "
                "have weaker convergence."
            )

            print(
                "These will be reported separately "
                "rather than automatically invalidating "
                "the catcher effects."
            )

    else:

        print(
            "MODEL 3 CATCHER STATUS: NOT READY"
        )

        print(
            "Do not interpret catcher effects yet."
        )

    print(
        "-" * 60
    )

    return (
        summary,
        catcher_summary,
        catcher_converged
    )


# ============================================================
# EXTRACT CATCHER EFFECTS
# ============================================================

def extract_catcher_effects(
    trace,
    catcher_values
):

    print(
        "\nExtracting Model 3 catcher effects..."
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

        # ====================================================
        # 2025
        # ====================================================

        effect_2025 = (
            catcher_samples[i]
        )

        # ====================================================
        # 2026
        # ====================================================

        effect_2026 = (
            catcher_samples[i]
            + season_samples[i]
        )

        # ====================================================
        # CHANGE
        # ====================================================

        change = (
            effect_2026
            - effect_2025
        )

        results.append({

            "catcher":
                catcher,

            "effect_2025":
                np.mean(
                    effect_2025
                ),

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
                np.mean(
                    effect_2026
                ),

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
                np.mean(
                    change
                ),

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

            "probability_decreased":
                np.mean(
                    change < 0
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
        "Model 3: Catcher Effect on "
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
        "MODEL 3 CATCHER RESULTS"
    )

    print(
        "=" * 60
    )

    # ========================================================
    # HIGHEST 2025
    # ========================================================

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
        .to_string(
            index=False
        )
    )

    # ========================================================
    # HIGHEST 2026
    # ========================================================

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
        .to_string(
            index=False
        )
    )

    # ========================================================
    # LARGEST DECREASES
    # ========================================================

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
        .to_string(
            index=False
        )
    )

    # ========================================================
    # LARGEST INCREASES
    # ========================================================

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
        .to_string(
            index=False
        )
    )


# ============================================================
# SAVE TRACE
# ============================================================

def save_trace(trace):

    print(
        "\nSaving Model 3 Bayesian trace..."
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

    # ========================================================
    # LOAD
    # ========================================================

    data = load_data()

    # ========================================================
    # PREPARE
    # ========================================================

    (
        data,
        catcher_values,
        pitch_categories,
        stand_categories,
        throws_categories,
    ) = prepare_data(
        data
    )

    # ========================================================
    # BUILD
    # ========================================================

    model = build_model(
        data,
        catcher_values,
        pitch_categories,
        stand_categories,
        throws_categories,
    )

    # ========================================================
    # SAMPLE
    # ========================================================

    trace = sample_model(
        model
    )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    (
        summary,
        catcher_summary,
        catcher_converged
    ) = diagnostics(
        trace
    )

    # ========================================================
    # ALWAYS SAVE TRACE
    # ========================================================

    save_trace(
        trace
    )

    # ========================================================
    # STOP IF CATCHER PARAMETERS DID NOT CONVERGE
    # ========================================================

    if not catcher_converged:

        print(
            "\n" + "=" * 60
        )

        print(
            "MODEL 3 CATCHER EFFECTS "
            "DID NOT PASS CONVERGENCE CHECKS"
        )

        print(
            "=" * 60
        )

        print(
            "\nThe trace was saved for debugging."
        )

        print(
            "\nNo catcher rankings will be produced."
        )

        print(
            "Increase sampling or further adjust "
            "the model before interpreting "
            "the catcher effects."
        )

        return

    # ========================================================
    # EXTRACT
    # ========================================================

    results = extract_catcher_effects(
        trace,
        catcher_values
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results.to_csv(
        RESULTS_FILE,
        index=False
    )

    print(
        f"\nSaved Model 3 catcher effects to:\n"
        f"  {RESULTS_FILE}"
    )

    # ========================================================
    # PRINT
    # ========================================================

    print_results(
        results
    )

    # ========================================================
    # PLOT
    # ========================================================

    plot_results(
        results
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "MODEL 3 COMPLETE"
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