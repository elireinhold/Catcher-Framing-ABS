from pathlib import Path
import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = PROCESSED_DIR / "model_data_2025_2026.csv"

# The data collection script creates files with names like:
#   statcast_2025-03-01_2025-09-10.csv
#   statcast_2026-03-01_2026-08-16.csv
#
# We intentionally discover the files instead of hard-coding the
# 2025/2026 end dates.


# ============================================================
# Find raw season files
# ============================================================

def find_season_file(season):
    files = sorted(RAW_DIR.glob(f"statcast_{season}-*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No Statcast file found for {season} in:\n{RAW_DIR}\n\n"
            f"Expected a filename beginning with "
            f"'statcast_{season}-'."
        )

    if len(files) > 1:
        print(f"\nWARNING: Found multiple {season} raw files:")
        for file in files:
            print(f"  {file.name}")
        print(f"Using the most recently modified file: {files[-1].name}")

    return files[-1]


# ============================================================
# Load and label a season
# ============================================================

def load_season(season):
    file = find_season_file(season)

    print(f"\nLoading {season} data:")
    print(f"  {file}")

    data = pd.read_csv(file)

    # Preserve the season explicitly. This is important because
    # game_date alone is not enough for later season comparisons.
    data["season"] = season

    print(f"  Rows: {len(data):,}")
    print(f"  Columns: {len(data.columns):,}")

    return data


# ============================================================
# Create taken-pitch dataset
# ============================================================

def prepare_data(data):
    print("\n" + "=" * 60)
    print("PREPARING CATCHER-FRAMING MODEL DATA")
    print("=" * 60)

    required = [
        "description",
        "plate_x",
        "plate_z",
        "pitcher",
        "batter",
        "fielder_2",
        "game_date",
    ]

    missing_columns = [
        column for column in required if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(f"  - {c}" for c in missing_columns)
        )

    # Only pitches where the batter did not swing and the call was
    # a normal ball or called strike are framing opportunities.
    #
    # automatic_ball / automatic_strike are intentionally excluded:
    # these are automated calls and therefore should not be attributed
    # to catcher framing.
    taken_descriptions = [
        "ball",
        "called_strike",
    ]

    taken = data[
        data["description"].isin(taken_descriptions)
    ].copy()

    print(f"\nAll pitches:       {len(data):,}")
    print(f"Taken pitches:     {len(taken):,}")

    # Create binary outcome.
    taken["called_strike"] = (
        taken["description"] == "called_strike"
    ).astype(int)

    # fielder_2 is the Statcast defensive position identifier for
    # the catcher.
    taken["catcher"] = taken["fielder_2"]

    # Ensure important numeric fields are numeric.
    numeric_columns = [
        "plate_x",
        "plate_z",
        "release_speed",
        "pfx_x",
        "pfx_z",
        "balls",
        "strikes",
    ]

    for column in numeric_columns:
        if column in taken.columns:
            taken[column] = pd.to_numeric(
                taken[column],
                errors="coerce"
            )

    # Remove rows that cannot be used for the core framing analysis.
    critical = [
        "game_date",
        "pitcher",
        "batter",
        "catcher",
        "plate_x",
        "plate_z",
        "called_strike",
    ]

    before = len(taken)

    taken = taken.dropna(subset=critical).copy()

    print(
        f"Removed {before - len(taken):,} rows "
        f"missing critical modeling data."
    )

    # Normalize dates.
    taken["game_date"] = pd.to_datetime(
        taken["game_date"],
        errors="coerce"
    )

    taken = taken.dropna(subset=["game_date"]).copy()

    # Reset index after filtering.
    taken = taken.reset_index(drop=True)

    # A simple exploratory zone. This is NOT the official ABS zone
    # and is not used as the final framing definition.
    ZONE_LEFT = -0.83
    ZONE_RIGHT = 0.83
    ZONE_BOTTOM = 1.50
    ZONE_TOP = 3.50

    taken["in_simple_zone"] = (
        taken["plate_x"].between(ZONE_LEFT, ZONE_RIGHT)
        & taken["plate_z"].between(ZONE_BOTTOM, ZONE_TOP)
    )

    # Keep a useful, stable set of columns while retaining the
    # original Statcast fields that may be useful later.
    print("\nSeason distribution:")
    print(taken["season"].value_counts().sort_index().to_string())

    print("\nOutcome distribution by season:")
    print(
        pd.crosstab(
            taken["season"],
            taken["called_strike"],
            normalize="index"
        ).mul(100).round(2).to_string()
    )

    return taken


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("CATCHER FRAMING + ABS")
    print("PHASE 2A: PREPARE 2025 + 2026 MODELING DATA")
    print("=" * 60)

    data_2025 = load_season(2025)
    data_2026 = load_season(2026)

    # Combine the two seasons before filtering so that all downstream
    # models operate on one consistent dataset.
    data = pd.concat(
        [data_2025, data_2026],
        ignore_index=True
    )

    print("\nCombined raw dataset:")
    print(f"  Rows:    {len(data):,}")
    print(f"  Columns: {len(data.columns):,}")

    modeling_data = prepare_data(data)

    modeling_data.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 60)
    print("FINAL MODELING DATASET")
    print("=" * 60)
    print(f"Rows:    {len(modeling_data):,}")
    print(f"Columns: {len(modeling_data.columns):,}")

    print("\nSeason counts:")
    print(
        modeling_data["season"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nSaved modeling dataset to:")
    print(f"  {OUTPUT_FILE}")

    print("\nSample:")
    sample_columns = [
        "season",
        "game_date",
        "called_strike",
        "plate_x",
        "plate_z",
        "pitch_type",
        "release_speed",
        "balls",
        "strikes",
        "pitcher",
        "batter",
        "catcher",
        "stand",
        "p_throws",
    ]

    available = [
        c for c in sample_columns
        if c in modeling_data.columns
    ]

    print(
        modeling_data[available]
        .head(10)
        .to_string(index=False)
    )

    print("\n" + "=" * 60)
    print("PHASE 2A COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()