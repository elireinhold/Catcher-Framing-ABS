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

INPUT_FILE = (
    PROCESSED_DIR / "model_data_2025_2026.csv"
)

RESULTS_FILE = (
    PROCESSED_DIR / "bayesian_catcher_effects_2025_2026.csv"
)

TRACE_FILE = (
    PROCESSED_DIR / "bayesian_trace.nc"
)

FALLBACK_TRACE_FILE = (
    PROCESSED_DIR / "bayesian_trace.pkl"
)

PLOT_FILE = (
    PROCESSED_DIR / "bayesian_catcher_effects.png"
)

# DEVELOPMENT SETTINGS
DRAWS = 500
TUNE = 1500
CHAINS = 4
CORES = 12
RANDOM_SEED = 42
TARGET_ACCEPT = 0.99
MAX_TREEDEPTH = 14

# Limit number of pitches to ensure there is enough data
MIN_CATCHER_PITCHES = 100
MAX_MODEL_PITCHES = 100_000


# LOAD DATA
def load_data():
    data = pd.read_csv(INPUT_FILE)
    print(
        f"Loaded {len(data):,} pitches."
    )
    return data

# PREPARE DATA
def prepare_data(data):

    print("\nPreparing Bayesian modeling data...")

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
    # --------------------------------------------------------

    required = [
        "called_strike",
        "catcher",
        "season",
        "plate_x",
        "plate_z",
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
    # Chronological order
    # --------------------------------------------------------

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

    # ADDED: track how many 2026 pitches each catcher has, so we
    # can flag weak identification of the catcher x season term.
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
            f"\nNOTE: {len(thin_catchers)} catcher(s) have fewer "
            f"than {MIN_CATCHER_PITCHES} pitches in 2026. Their "
            "catcher_season_effect will be pulled strongly toward "
            "the prior and is a likely source of any remaining "
            "divergences/low ESS."
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
    # STANDARDIZE CONTINUOUS VARIABLES
    # ========================================================

    continuous = [
        "plate_x",
        "plate_z",
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

        if not np.isfinite(std) or std == 0:
            std = 1.0

        data[column + "_std"] = (
            (data[column] - mean)
            / std
        )

    # ========================================================
    # CENTER QUADRATIC / INTERACTION TERMS
    #
    # CHANGED: plate_x_std**2 and plate_z_std**2 have a mean
    # near 1, not 0, because they're squares of standardized
    # variables. Left uncentered, beta_x2/beta_z2 end up highly
    # correlated with the intercept and with beta_x/beta_z,
    # which warps the posterior geometry and is a common cause
    # of divergences. Centering removes that correlation without
    # changing what the model represents (it just shifts the
    # intercept, which is already a free parameter).
    # ========================================================

    data["plate_x2_c"] = (
        data["plate_x_std"] ** 2
        - (data["plate_x_std"] ** 2).mean()
    )

    data["plate_z2_c"] = (
        data["plate_z_std"] ** 2
        - (data["plate_z_std"] ** 2).mean()
    )

    data["plate_xz_c"] = (
        data["plate_x_std"] * data["plate_z_std"]
        - (
            data["plate_x_std"] * data["plate_z_std"]
        ).mean()
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\nFinal Bayesian dataset:")

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
        "\nCalled-strike rate:"
    )

    print(
        f"  {data['called_strike'].mean():.4f}"
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
        "\nBuilding Bayesian hierarchical model..."
    )

    # --------------------------------------------------------
    # Arrays
    # --------------------------------------------------------

    y = (
        data["called_strike"]
        .astype(int)
        .values
    )

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

    plate_x = (
        data["plate_x_std"]
        .astype(float)
        .values
    )

    plate_z = (
        data["plate_z_std"]
        .astype(float)
        .values
    )

    # CHANGED: use the centered quadratic/interaction terms
    # built in prepare_data() instead of raw squares/products.
    plate_x2 = (
        data["plate_x2_c"]
        .astype(float)
        .values
    )

    plate_z2 = (
        data["plate_z2_c"]
        .astype(float)
        .values
    )

    plate_xz = (
        data["plate_xz_c"]
        .astype(float)
        .values
    )

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
    # Dimensions
    # --------------------------------------------------------

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

    print("\nModel dimensions:")

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
        # LOCATION
        # ====================================================

        beta_x = pm.Normal(
            "beta_x",
            mu=0,
            sigma=1
        )

        beta_z = pm.Normal(
            "beta_z",
            mu=0,
            sigma=1
        )

        beta_x2 = pm.Normal(
            "beta_x2",
            mu=0,
            sigma=0.5
        )

        beta_z2 = pm.Normal(
            "beta_z2",
            mu=0,
            sigma=0.5
        )

        beta_xz = pm.Normal(
            "beta_xz",
            mu=0,
            sigma=0.5
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
        # NON-CENTERED PARAMETERIZATION
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
        # CATCHER × SEASON EFFECT
        #
        # NON-CENTERED PARAMETERIZATION
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

            # Location
            + beta_x * plate_x
            + beta_z * plate_z
            + beta_x2 * plate_x2
            + beta_z2 * plate_z2
            + beta_xz * plate_xz

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
            "called_strike",
            p=probability,
            observed=y
        )

    return model


# ============================================================
# SAMPLE MODEL
# ============================================================

def sample_model(model):

    print("\n" + "=" * 60)
    print("STARTING BAYESIAN SAMPLING")
    print("=" * 60)

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

        # CHANGED: use the nutpie sampler instead of PyMC's
        # default. nutpie ships prebuilt Windows wheels (Rust,
        # not C++), so it works without g++/MSVC installed and
        # is typically faster than the default sampler even when
        # a compiler IS available. Requires: pip install nutpie
        #
        # nutpie also runs chains in parallel by default
        # regardless of CORES, so tuning/sampling won't be
        # sequential the way it was with cores=1 above.
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
        "BAYESIAN DIAGNOSTICS"
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
        f"\nDivergences:"
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
            "MODEL STATUS: GOOD"
        )

        print(
            "The basic convergence checks passed."
        )

    else:

        print(
            "MODEL STATUS: NOT READY"
        )

        print(
            "Do not interpret catcher rankings yet."
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
        "\nExtracting posterior catcher effects..."
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

        effect_2025 = (
            catcher_samples[i]
        )

        effect_2026 = (
            catcher_samples[i]
            + season_samples[i]
        )

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

            "probability_improved":
                np.mean(change > 0),

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
        "Bayesian Catcher Effect Change: "
        "2025 → 2026"
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
        "BAYESIAN CATCHER RESULTS"
    )

    print(
        "=" * 60
    )

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
            "probability_improved"
        ]]
        .head(15)
        .round(4)
        .to_string(index=False)
    )

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
            "probability_improved"
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
        "\nSaving Bayesian trace..."
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
    # STOP IF MODEL IS NOT CONVERGED
    # --------------------------------------------------------

    if not converged:

        print(
            "\n" + "=" * 60
        )

        print(
            "MODEL DID NOT PASS CONVERGENCE CHECKS"
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
        f"\nSaved catcher effects to:\n"
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
        "BAYESIAN MODEL COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        "\nPrimary research quantity:"
    )

    print(
        "\n2026 catcher effect"
        "\nminus"
        "\n2025 catcher effect"
    )

    print(
        "\nThis is represented by the "
        "catcher × 2026 interaction."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
