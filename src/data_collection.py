from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pybaseball import statcast


# ============================================================
# Configuration
# ============================================================

DATE_RANGES = [
    {
        "label": "2025",
        "start": "2025-03-01",
        "end": "2025-09-10",
    },
    {
        "label": "2026",
        "start": "2026-03-01",
        "end": "2026-08-16",
    },
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Step 1: Download Statcast data
# ============================================================

def download_statcast(start_date, end_date):
    print("=" * 60)
    print("DOWNLOADING STATCAST DATA")
    print("=" * 60)

    raw_file = (
        RAW_DIR
        / f"statcast_{start_date}_{end_date}.csv"
    )

    if raw_file.exists():

        print("Raw data already exists.")
        print(f"  {raw_file}")
        print("Skipping download.\n")

        return pd.read_csv(raw_file)

    print(
        f"Date range: {start_date} -> {end_date}"
    )

    print(
        "Downloading from Baseball Savant...\n"
    )

    data = statcast(
        start_dt=start_date,
        end_dt=end_date
    )

    print("\nDownload complete.")

    print(
        f"Rows: {len(data):,}"
    )

    print(
        f"Columns: {len(data.columns):,}"
    )

    data.to_csv(
        raw_file,
        index=False
    )

    print("\nSaved raw data to:")
    print(f"  {raw_file}\n")

    return data


# ============================================================
# Step 2: Inspect the dataset
# ============================================================

def inspect_data(data):
    print("=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)

    print("\nShape:")
    print(data.shape)

    print("\nFirst 5 rows:")
    print(data.head())

    print("\nNumber of columns:")
    print(len(data.columns))

    print("\nMissing values for important fields:")

    important_columns = [
        "game_date",
        "pitcher",
        "batter",
        "pitch_type",
        "release_speed",
        "plate_x",
        "plate_z",
        "balls",
        "strikes",
        "description",
        "fielder_2",
    ]

    for column in important_columns:

        if column in data.columns:

            missing = data[column].isna().sum()

            percentage = (
                missing / len(data) * 100
            )

            print(
                f"  {column:20s}"
                f"{missing:10,} missing"
                f" ({percentage:.2f}%)"
            )

        else:

            print(
                f"  {column:20s}"
                f"NOT FOUND"
            )

    print()


# ============================================================
# Step 3: Look for catcher-related fields
# ============================================================

def find_catcher_columns(data):

    print("=" * 60)
    print("CATCHER-RELATED COLUMNS")
    print("=" * 60)

    catcher_columns = [
        column
        for column in data.columns
        if "catch" in column.lower()
    ]

    fielder_columns = [
        column
        for column in data.columns
        if "fielder" in column.lower()
    ]

    print("\nColumns containing 'catch':")

    if catcher_columns:

        for column in catcher_columns:
            print(f"  {column}")

    else:

        print("  None found.")

    print("\nColumns containing 'fielder':")

    if fielder_columns:

        for column in fielder_columns:
            print(f"  {column}")

    else:

        print("  None found.")

    print(
        "\nUsing fielder_2 as the catcher identifier."
    )

    print()


# ============================================================
# Step 4: Examine pitch descriptions
# ============================================================

def examine_descriptions(data):

    print("=" * 60)
    print("PITCH DESCRIPTIONS")
    print("=" * 60)

    if "description" not in data.columns:

        print(
            "ERROR: 'description' column not found."
        )

        return

    print(
        data["description"]
        .value_counts(dropna=False)
        .to_string()
    )

    print()


# ============================================================
# Step 5: Create taken-pitch dataset
# ============================================================

def create_taken_pitch_dataset(
    data,
    start_date,
    end_date
):

    print("=" * 60)
    print("CREATING TAKEN-PITCH DATASET")
    print("=" * 60)

    if "description" not in data.columns:

        raise ValueError(
            "Statcast data does not contain "
            "a 'description' column."
        )

    # --------------------------------------------------------
    # For catcher framing, we only care about pitches where
    # the batter did not swing.
    #
    # For now, we define these as:
    #   - called_strike
    #   - ball
    # --------------------------------------------------------

    taken_descriptions = [
        "called_strike",
        "ball"
    ]

    taken = data[
        data["description"].isin(
            taken_descriptions
        )
    ].copy()

    print(
        f"Total pitches:       {len(data):,}"
    )

    print(
        f"Taken pitches:       {len(taken):,}"
    )

    print(
        f"Percentage taken:    "
        f"{len(taken) / len(data) * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Create catcher column.
    #
    # Statcast uses defensive position numbering:
    #   2 = catcher
    #
    # Therefore fielder_2 represents the catcher.
    # --------------------------------------------------------

    if "fielder_2" in taken.columns:

        taken["catcher"] = taken["fielder_2"]

        print(
            "\nCreated 'catcher' column "
            "from 'fielder_2'."
        )

    else:

        print(
            "\nWARNING: 'fielder_2' was not found."
        )

        taken["catcher"] = pd.NA

    # --------------------------------------------------------
    # Display called strikes vs. balls
    # --------------------------------------------------------

    print("\nCalled strikes vs. balls:")

    print(
        taken["description"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Remove rows missing important pitch-location data.
    # --------------------------------------------------------

    before = len(taken)

    taken = taken.dropna(
        subset=[
            "plate_x",
            "plate_z"
        ]
    )

    after = len(taken)

    print(
        f"\nRemoved {before - after:,} pitches "
        f"missing plate location."
    )

    # --------------------------------------------------------
    # Reset the index.
    # --------------------------------------------------------

    taken = taken.reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Create a simple initial strike zone.
    #
    # IMPORTANT:
    # This is NOT our final ABS zone.
    # It is only for initial exploration.
    # --------------------------------------------------------

    ZONE_LEFT = -0.83
    ZONE_RIGHT = 0.83

    ZONE_BOTTOM = 1.50
    ZONE_TOP = 3.50

    taken["in_simple_zone"] = (
        taken["plate_x"].between(
            ZONE_LEFT,
            ZONE_RIGHT
        )
        &
        taken["plate_z"].between(
            ZONE_BOTTOM,
            ZONE_TOP
        )
    )

    # --------------------------------------------------------
    # Save processed data.
    # --------------------------------------------------------

    taken_file = (
        PROCESSED_DIR
        / f"taken_pitches_{start_date}_{end_date}.csv"
    )

    taken.to_csv(
        taken_file,
        index=False
    )

    print("\nSaved taken-pitch dataset to:")
    print(f"  {taken_file}")

    return taken


# ============================================================
# Step 6: Verify catcher information
# ============================================================

def verify_catcher_data(taken):

    print("=" * 60)
    print("CATCHER DATA VERIFICATION")
    print("=" * 60)

    if "catcher" not in taken.columns:

        print(
            "ERROR: 'catcher' column was not created."
        )

        return

    missing = taken["catcher"].isna().sum()

    print(
        f"\nMissing catcher values: "
        f"{missing:,} "
        f"({missing / len(taken) * 100:.2f}%)"
    )

    print(
        "\nMost common catcher IDs:"
    )

    print(
        taken["catcher"]
        .value_counts()
        .head(20)
        .to_string()
    )

    # --------------------------------------------------------
    # Verify that catcher matches fielder_2.
    # --------------------------------------------------------

    if "fielder_2" in taken.columns:

        matches = (
            taken["catcher"]
            == taken["fielder_2"]
        ).all()

        print(
            f"\nCatcher matches fielder_2: "
            f"{matches}"
        )

    print()


# ============================================================
# Step 7: Analyze simple strike zone
# ============================================================

def analyze_simple_zone(taken):

    print("=" * 60)
    print("SIMPLE STRIKE-ZONE ANALYSIS")
    print("=" * 60)

    taken = taken.reset_index(
        drop=True
    )

    zone_status = (
        taken["in_simple_zone"]
        .astype(str)
    )

    table = pd.crosstab(
        zone_status,
        taken["description"],
        normalize="index"
    ) * 100

    table = table.rename(
        index={
            "True": "Inside simple zone",
            "False": "Outside simple zone"
        }
    )

    print(
        "\nPercentage of calls by location:"
    )

    print(
        table.round(2)
        .to_string()
    )

    print()


# ============================================================
# Step 8: Plot pitch locations
# ============================================================

def create_location_plot(
    taken,
    start_date,
    end_date
):

    print("=" * 60)
    print("CREATING PITCH LOCATION PLOT")
    print("=" * 60)

    strikes = taken[
        taken["description"]
        == "called_strike"
    ]

    balls = taken[
        taken["description"]
        == "ball"
    ]

    plt.figure(
        figsize=(8, 8)
    )

    # --------------------------------------------------------
    # Called strikes
    # --------------------------------------------------------

    plt.scatter(
        strikes["plate_x"],
        strikes["plate_z"],
        alpha=0.15,
        label="Called Strike"
    )

    # --------------------------------------------------------
    # Balls
    # --------------------------------------------------------

    plt.scatter(
        balls["plate_x"],
        balls["plate_z"],
        alpha=0.15,
        label="Ball"
    )

    # --------------------------------------------------------
    # Initial approximate strike zone
    # --------------------------------------------------------

    ZONE_LEFT = -0.83
    ZONE_RIGHT = 0.83

    ZONE_BOTTOM = 1.50
    ZONE_TOP = 3.50

    plt.plot(
        [
            ZONE_LEFT,
            ZONE_RIGHT,
            ZONE_RIGHT,
            ZONE_LEFT,
            ZONE_LEFT
        ],
        [
            ZONE_BOTTOM,
            ZONE_BOTTOM,
            ZONE_TOP,
            ZONE_TOP,
            ZONE_BOTTOM
        ],
        linewidth=2,
        label="Initial Approximate Zone"
    )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    plt.xlabel(
        "Horizontal Plate Location (feet)"
    )

    plt.ylabel(
        "Vertical Plate Location (feet)"
    )

    plt.title(
        "Called Strikes vs. Balls\n"
        f"{start_date} to {end_date}"
    )

    plt.xlim(
        -2,
        2
    )

    plt.ylim(
        0,
        5
    )

    plt.legend()

    plt.grid(
        alpha=0.2
    )

    plt.tight_layout()

    plot_file = (
        FIGURES_DIR
        / f"called_strikes_vs_balls_"
        f"{start_date}_{end_date}.png"
    )

    plt.savefig(
        plot_file,
        dpi=300
    )

    plt.close()

    print(
        "\nSaved plot to:"
    )

    print(
        f"  {plot_file}\n"
    )


# ============================================================
# Step 9: Display sample
# ============================================================

def display_sample(taken):

    print("=" * 60)
    print("SAMPLE TAKEN PITCHES")
    print("=" * 60)

    columns = [
        "game_date",
        "pitcher",
        "batter",
        "catcher",
        "pitch_type",
        "release_speed",
        "plate_x",
        "plate_z",
        "balls",
        "strikes",
        "description",
    ]

    available_columns = [
        column
        for column in columns
        if column in taken.columns
    ]

    print(
        taken[
            available_columns
        ]
        .head(20)
        .to_string(index=False)
    )

    print()


# ============================================================
# Step 10: Process one date range
# ============================================================

def process_date_range(start_date, end_date):

    print("\n")
    print("#" * 60)
    print(
        f"PROCESSING {start_date} -> {end_date}"
    )
    print("#" * 60)
    print()

    # --------------------------------------------------------
    # 1. Download Statcast data
    # --------------------------------------------------------

    data = download_statcast(
        start_date,
        end_date
    )

    print("\nUmpire-related columns:")

    print([
        c
        for c in data.columns
        if "ump" in c.lower()
    ])

    print(
        "\nBatter/pitcher handedness columns:"
    )

    print([
        c
        for c in data.columns
        if "stand" in c.lower()
        or "throw" in c.lower()
    ])

    print(
        "\nPotential context columns:"
    )

    print([
        c
        for c in data.columns
        if c in [
            "p_throws",
            "stand",
            "inning",
            "outs_when_up",
            "on_1b",
            "on_2b",
            "on_3b"
        ]
    ])

    # --------------------------------------------------------
    # 2. Inspect data
    # --------------------------------------------------------

    inspect_data(data)

    # --------------------------------------------------------
    # 3. Find catcher-related columns
    # --------------------------------------------------------

    find_catcher_columns(data)

    # --------------------------------------------------------
    # 4. Examine pitch descriptions
    # --------------------------------------------------------

    examine_descriptions(data)

    # --------------------------------------------------------
    # 5. Create taken-pitch dataset
    # --------------------------------------------------------

    taken = create_taken_pitch_dataset(
        data,
        start_date,
        end_date
    )

    # --------------------------------------------------------
    # 6. Verify catcher information
    # --------------------------------------------------------

    verify_catcher_data(
        taken
    )

    # --------------------------------------------------------
    # 7. Analyze simple zone
    # --------------------------------------------------------

    analyze_simple_zone(
        taken
    )

    # --------------------------------------------------------
    # 8. Create visualization
    # --------------------------------------------------------

    create_location_plot(
        taken,
        start_date,
        end_date
    )

    # --------------------------------------------------------
    # 9. Display sample
    # --------------------------------------------------------

    display_sample(
        taken
    )

    print("=" * 60)
    print(
        f"DATE RANGE COMPLETE: "
        f"{start_date} -> {end_date}"
    )
    print("=" * 60)
    print()


# ============================================================
# Main
# ============================================================

def main():

    print("\n")

    print("=" * 60)
    print("CATCHER FRAMING + ABS PROJECT")
    print("PHASE 1: STATCAST DATA COLLECTION")
    print("=" * 60)

    print(
        "\nCollecting:"
    )

    for date_range in DATE_RANGES:

        print(
            f"  {date_range['label']}: "
            f"{date_range['start']} -> "
            f"{date_range['end']}"
        )

    print()

    # --------------------------------------------------------
    # Process each season
    # --------------------------------------------------------

    for date_range in DATE_RANGES:

        process_date_range(
            date_range["start"],
            date_range["end"]
        )

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------

    print("=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)

    print(
        "\nProcessed date ranges:"
    )

    for date_range in DATE_RANGES:

        print(
            f"  {date_range['start']} -> "
            f"{date_range['end']}"
        )

    print(
        "\nRaw data is stored in:"
    )

    print(
        f"  {RAW_DIR}"
    )

    print(
        "\nProcessed data is stored in:"
    )

    print(
        f"  {PROCESSED_DIR}"
    )

    print(
        "\nFigures are stored in:"
    )

    print(
        f"  {FIGURES_DIR}"
    )

    print("\nNext step:")

    print(
        "Verify the catcher IDs and Statcast fields "
        "before building the framing model."
    )

    print()


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()