"""
preliminary_renewable_only_balance.py

PRELIMINARY, PRE-LOPF SCREENING ONLY -- not the final LOPF analysis.

Calculates the raw hourly mismatch between available renewable generation
(wind + tidal + wave) and demand, BEFORE any storage, gas generation,
mainland import/export, curtailment optimisation, or dispatch optimisation
is introduced:

    Balance_t = Wind_t + Wave_t + Tidal_t - Demand_t

    positive balance = renewable excess/surplus
    negative balance = renewable shortfall

This uses ONLY the existing files in the DRES project input folder -- no
external data, literature values, or additional assumptions are introduced.

Scenarios (wind and tidal held fixed; wave varied):
    EB1: wind (55.4 MW existing DRES fleet), tidal 7.2 MW, wave 0 MW
    EB2: wind (55.4 MW existing DRES fleet), tidal 7.2 MW, wave 5 MW
    EB3: wind (55.4 MW existing DRES fleet), tidal 7.2 MW, wave 7 MW
    EB4: wind (55.4 MW existing DRES fleet), tidal 7.2 MW, wave 15 MW
    EB5: wind (55.4 MW existing DRES fleet), tidal 7.2 MW, wave 30 MW

Explicitly EXCLUDED from this screening (see the LOPF scripts for these):
    battery storage, gas turbine generation, Slack/mainland import/export,
    curtailment optimisation, PyPSA network.optimize(), PyPSA dispatch
    results, operating cost, capital costs, voltage/line-loading analysis.

This is a simple hourly arithmetic calculation only.
"""

import glob
import os
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# CONSTANTS
# ==============================================================================

INPUT_DIR = "DATA/inputs"
OUTPUT_DIR = "DATA/outputs"
FIGURES_DIR = f"{OUTPUT_DIR}/figures"

LOAD_PROFILE_FILE = f"{INPUT_DIR}/Load_Profile.csv"
TIDAL_PROFILE_FILE = f"{INPUT_DIR}/tidal_orkney_2019.csv"
WAVE_PROFILE_FILE = f"{INPUT_DIR}/wave_orkney_2019.csv"
WIND_FILE_PATTERN = f"{INPUT_DIR}/WindTurbine_*.csv"

SUMMARY_CSV = f"{OUTPUT_DIR}/preliminary_renewable_only_energy_balance_5_scenarios.csv"
HOURLY_CSV = f"{OUTPUT_DIR}/preliminary_renewable_only_hourly_balance_5_scenarios.csv"

EXPECTED_HOURS = 8760
EXPECTED_ANNUAL_DEMAND_MWH = 195014  # per project record -- warn, do not force-correct, if materially different

TIDAL_CAPACITY_MW = 7.2  # fixed, matching the final LOPF analysis

# Scenario ID -> wave capacity (MW). Wind and tidal are identical across all five.
SCENARIOS = {
    "EB1": 0,
    "EB2": 5,
    "EB3": 7,
    "EB4": 15,
    "EB5": 30,
}


# ==============================================================================
# DATA LOADING -- defensive, with explicit validation checks throughout
# ==============================================================================

def find_datetime_column(df, filename=""):
    """Best-effort detection of a DateTime-like column, since exact column
    naming may vary slightly between input files."""
    candidates = [c for c in df.columns if str(c).strip().lower() in
                  ("datetime", "date_time", "date", "timestamp", "time")]
    if candidates:
        return candidates[0]
    # Fall back to the first column if nothing obviously named matches
    warnings.warn(f"No obviously-named DateTime column found in {filename}; "
                   f"falling back to first column '{df.columns[0]}'. Verify this is correct.")
    return df.columns[0]


def parse_datetime_robust(series, filename=""):
    """
    Tries dayfirst=True first (per the original request), then falls back
    to dayfirst=False (month-first) if that raises -- different input files
    in this project have been found to use different date conventions
    (e.g. wind turbine files use month-first: "1/13/2019" = 13 Jan, which
    cannot be day-first since there is no 13th month). Reports which
    convention was actually used so this is never silently guessed.
    """
    try:
        parsed = pd.to_datetime(series, dayfirst=True, errors="raise")
        print(f"  [{filename}] Parsed as day-first.")
        return parsed
    except (ValueError, TypeError):
        pass

    try:
        parsed = pd.to_datetime(series, dayfirst=False, errors="raise")
        print(f"  [{filename}] Parsed as MONTH-first (day-first failed).")
        return parsed
    except (ValueError, TypeError):
        pass

    # Last resort: let pandas infer per-element format
    parsed = pd.to_datetime(series, format="mixed", dayfirst=True, errors="raise")
    warnings.warn(f"[{filename}] Neither pure day-first nor month-first parsing worked cleanly; "
                   f"used format='mixed' as a last resort. VERIFY this file's dates parsed correctly.")
    return parsed


def load_wind_generation():
    """
    Loads all DATA/inputs/WindTurbine_*.csv files and aggregates total wind
    generation, correctly matching base_model.py's replication logic.

    IMPORTANT: each WindTurbine_N.csv file represents ONE representative
    turbine's profile for that site -- NOT the site's full aggregate output.
    base_model.py replicates this single profile across n_turbines separate
    generator objects per site (add_wind_turbines_to_network(): the same
    file is loaded once, then network.add("Generator", ...) is called
    n_turbines times in a loop, each with identical p_nom and p_max_pu).

    An earlier version of this script summed each file ONCE, silently
    undercounting every site with n_turbines > 1 -- confirmed against the
    base network file, which has 32 wind generator components from these
    same 20 site files (12 "extra" generators exactly matching the sites
    below with n_turbines > 1). This version applies the same per-site
    turbine count multiplier used in base_model.py, so results match.

    Per-site turbine counts below were derived directly from inspecting the
    base network's generator list (network_wave5MW_tidal7.2MW.nc) -- NOT
    from an assumption -- since this screening does not have direct access
    to the underlying WindFarms asset configuration file base_model.py
    reads n_turbines from.
    """
    # Site file (basename, without extension) -> number of physical turbines
    # at that site. Confirmed against the base network's actual generator
    # list -- any site not listed here defaults to 1 turbine.
    TURBINE_COUNT_PER_SITE = {
        "WindTurbine_3": 2,
        "WindTurbine_4": 2,
        "WindTurbine_5": 5,    # Hammers Hill
        "WindTurbine_12": 5,   # Spurness
        "WindTurbine_16": 3,
    }

    wind_files = sorted(glob.glob(WIND_FILE_PATTERN))
    if not wind_files:
        raise FileNotFoundError(f"No wind turbine files found matching {WIND_FILE_PATTERN}")

    print(f"Found {len(wind_files)} wind turbine SITE file(s) -- each representing one turbine's profile, "
          f"replicated per-site by turbine count to match base_model.py's logic.")

    aggregated = None
    total_turbines = 0
    for path in wind_files:
        df = pd.read_csv(path)
        dt_col = find_datetime_column(df, filename=os.path.basename(path))
        df[dt_col] = parse_datetime_robust(df[dt_col], filename=os.path.basename(path))
        df = df.set_index(dt_col).sort_index()

        if len(df) != EXPECTED_HOURS:
            warnings.warn(f"{os.path.basename(path)} has {len(df)} rows, expected {EXPECTED_HOURS}.")

        if "ActivePower" not in df.columns:
            raise KeyError(f"'ActivePower' column not found in {path}. Columns present: {list(df.columns)}")

        site_key = os.path.splitext(os.path.basename(path))[0]
        n_turbines = TURBINE_COUNT_PER_SITE.get(site_key, 1)
        total_turbines += n_turbines

        # Multiply this single turbine's profile by the site's turbine count
        # -- matching base_model.py's replication of n_turbines identical
        # generators per site, each with the same p_nom and p_max_pu shape.
        series = (df["ActivePower"] * n_turbines).rename(os.path.basename(path))
        print(f"  {os.path.basename(path)}: n_turbines={n_turbines}, "
              f"single-turbine annual sum={df['ActivePower'].sum():,.1f} MWh, "
              f"site total={series.sum():,.1f} MWh")

        aggregated = series if aggregated is None else aggregated.add(series, fill_value=0.0)

    aggregated.name = "wind_mw"
    print(f"\nAggregated wind generation across {len(wind_files)} site file(s) / {total_turbines} turbines: "
          f"{aggregated.sum():,.1f} MWh/year (raw sum, pre-alignment).")
    return aggregated, len(wind_files)


def load_demand():
    """
    Loads Load_Profile.csv and sums all columns whose name ends in '[MW]'
    to get total Orkney demand.
    """
    df = pd.read_csv(LOAD_PROFILE_FILE)
    dt_col = find_datetime_column(df, filename="Load_Profile.csv")
    df[dt_col] = parse_datetime_robust(df[dt_col], filename="Load_Profile.csv")
    df = df.set_index(dt_col).sort_index()

    mw_cols = [c for c in df.columns if str(c).strip().endswith("[MW]")]
    if not mw_cols:
        raise KeyError(f"No columns ending in '[MW]' found in {LOAD_PROFILE_FILE}. "
                        f"Columns present: {list(df.columns)}")

    demand = df[mw_cols].sum(axis=1)
    demand.name = "demand_mw"
    print(f"Demand loaded from {len(mw_cols)} column(s) ending in '[MW]': {mw_cols}")
    return demand


def load_normalised_profile(path, label):
    """
    Loads a normalised (0-1) availability profile (wave or tidal), clipping
    defensively to [0, 1] and warning if clipping was actually needed.
    """
    df = pd.read_csv(path)
    dt_col = find_datetime_column(df, filename=os.path.basename(path))
    df[dt_col] = parse_datetime_robust(df[dt_col], filename=os.path.basename(path))
    df = df.set_index(dt_col).sort_index()

    value_cols = [c for c in df.columns if c != dt_col]
    if not value_cols:
        raise KeyError(f"No value column found in {path} besides the datetime column.")
    series = df[value_cols[0]].astype(float)

    below_zero = (series < 0).sum()
    above_one = (series > 1).sum()
    if below_zero or above_one:
        warnings.warn(f"{label} profile ({path}) had {below_zero} value(s) < 0 and {above_one} value(s) > 1 "
                       f"before clipping -- check the source file if this is unexpected.")
    series = series.clip(lower=0.0, upper=1.0)
    series.name = f"{label}_pu"

    print(f"{label} profile loaded: {len(series)} rows.")
    return series


# ==============================================================================
# ALIGNMENT AND VALIDATION
# ==============================================================================

def align_all_series(wind_mw, demand_mw, wave_pu, tidal_pu):
    """
    Aligns all four series to their common timestamp intersection, confirms
    no missing values remain, and reports the aligned length.
    """
    common_index = wind_mw.index.intersection(demand_mw.index).intersection(
        wave_pu.index).intersection(tidal_pu.index)

    wind_aligned = wind_mw.reindex(common_index)
    demand_aligned = demand_mw.reindex(common_index)
    wave_aligned = wave_pu.reindex(common_index)
    tidal_aligned = tidal_pu.reindex(common_index)

    n_common = len(common_index)
    print(f"\nCommon aligned timestamps across all four series: {n_common}")
    if n_common != EXPECTED_HOURS:
        warnings.warn(f"Aligned series length ({n_common}) does not equal the expected {EXPECTED_HOURS} hours.")

    missing_report = {
        "wind": int(wind_aligned.isna().sum()),
        "demand": int(demand_aligned.isna().sum()),
        "wave": int(wave_aligned.isna().sum()),
        "tidal": int(tidal_aligned.isna().sum()),
    }
    any_missing = any(v > 0 for v in missing_report.values())
    if any_missing:
        raise ValueError(f"Missing values found after alignment: {missing_report}. "
                          f"Resolve before proceeding -- this screening does not fill gaps.")
    print("No missing values found after alignment.")

    return wind_aligned, demand_aligned, wave_aligned, tidal_aligned, n_common, any_missing


def validate_annual_demand(demand_mw):
    annual_demand_mwh = float(demand_mw.sum())
    diff = annual_demand_mwh - EXPECTED_ANNUAL_DEMAND_MWH
    pct_diff = diff / EXPECTED_ANNUAL_DEMAND_MWH * 100
    print(f"\nAnnual demand: {annual_demand_mwh:,.1f} MWh (expected approximately "
          f"{EXPECTED_ANNUAL_DEMAND_MWH:,} MWh, difference {pct_diff:+.2f}%)")
    if abs(pct_diff) > 1.0:
        warnings.warn(f"Annual demand differs from the expected value by {pct_diff:+.2f}% "
                       f"(more than 1%). NOT force-corrected -- investigate the source data if unexpected.")
    return annual_demand_mwh


# ==============================================================================
# SCENARIO CALCULATION
# ==============================================================================

def calculate_scenario(scenario_id, wave_capacity_mw, wind_mw, demand_mw, wave_pu, tidal_pu):
    """
    Calculates the hourly renewable-only balance and annual summary metrics
    for one scenario. Wind is used as-is (not rescaled); tidal is fixed at
    TIDAL_CAPACITY_MW; wave is scaled by wave_capacity_mw.
    """
    tidal_mw = tidal_pu * TIDAL_CAPACITY_MW
    wave_mw = wave_pu * wave_capacity_mw

    total_renewable_mw = wind_mw + tidal_mw + wave_mw
    balance_mw = total_renewable_mw - demand_mw

    shortfall_mw = (-balance_mw).clip(lower=0.0)
    excess_mw = balance_mw.clip(lower=0.0)

    wind_generation_mwh = float(wind_mw.sum())
    wave_generation_mwh = float(wave_mw.sum())
    tidal_generation_mwh = float(tidal_mw.sum())
    total_renewable_generation_mwh = float(total_renewable_mw.sum())
    total_demand_mwh = float(demand_mw.sum())
    annual_net_balance_mwh = total_renewable_generation_mwh - total_demand_mwh

    shortfall_energy_mwh = float(shortfall_mw.sum())
    excess_energy_mwh = float(excess_mw.sum())
    shortfall_hours = int((balance_mw < 0).sum())
    excess_hours = int((balance_mw > 0).sum())
    max_shortfall_mw = float(balance_mw.min())   # most negative hourly balance
    max_excess_mw = float(balance_mw.max())
    n_hours = len(balance_mw)
    pct_shortfall_hours = shortfall_hours / n_hours * 100
    pct_excess_hours = excess_hours / n_hours * 100

    summary = {
        "scenario_id": scenario_id,
        "wave_capacity_mw": wave_capacity_mw,
        "tidal_capacity_mw": TIDAL_CAPACITY_MW,
        "wind_generation_mwh": wind_generation_mwh,
        "wave_generation_mwh": wave_generation_mwh,
        "tidal_generation_mwh": tidal_generation_mwh,
        "total_renewable_generation_mwh": total_renewable_generation_mwh,
        "total_demand_mwh": total_demand_mwh,
        "annual_net_balance_mwh": annual_net_balance_mwh,
        "shortfall_energy_mwh": shortfall_energy_mwh,
        "excess_energy_mwh": excess_energy_mwh,
        "shortfall_hours": shortfall_hours,
        "excess_hours": excess_hours,
        "max_shortfall_mw": max_shortfall_mw,
        "max_excess_mw": max_excess_mw,
        "pct_hours_shortfall": pct_shortfall_hours,
        "pct_hours_excess": pct_excess_hours,
    }

    hourly = pd.DataFrame({
        f"wave_mw_{scenario_id}": wave_mw,
        f"tidal_mw_{scenario_id}": tidal_mw,
        f"total_renewable_mw_{scenario_id}": total_renewable_mw,
        f"balance_mw_{scenario_id}": balance_mw,
        f"shortfall_mw_{scenario_id}": shortfall_mw,
        f"excess_mw_{scenario_id}": excess_mw,
    })

    return summary, hourly


def make_scenario_dashboard(scenario_id, wave_capacity_mw, summary_row, hourly_df, wind_mw, demand_mw,
                             wave_mw, tidal_mw, figures_dir=FIGURES_DIR):
    """
    One combined dashboard figure per scenario, matching the required layout:
      Panel 1: Wind generation vs demand (full year)
      Panel 2: Wave and tidal generation, own scale (full year)
      Panel 3: Net balance -- excess (green, above zero) / shortfall (red, below zero)
      Panel 4: Monthly excess / shortfall bar chart
    (No text summary panel -- charts only. Summary statistics are available
    separately in the summary CSV.)
    """
    os.makedirs(figures_dir, exist_ok=True)

    balance_mw = hourly_df[f"balance_mw_{scenario_id}"]
    shortfall_mw = hourly_df[f"shortfall_mw_{scenario_id}"]
    excess_mw = hourly_df[f"excess_mw_{scenario_id}"]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(11, 9))
    fig.subplots_adjust(hspace=0.55)

    fig.suptitle(f"MODEL ENERGY BALANCE: {wave_capacity_mw}MW WAVE + {TIDAL_CAPACITY_MW}MW TIDAL",
                 fontsize=18, fontweight="normal", y=0.98)

    # --- Panel 1: Wind generation vs demand ------------------------------------
    ax1.fill_between(wind_mw.index, wind_mw.values, color="#7fa8d9", alpha=0.85, label="Wind")
    ax1.plot(demand_mw.index, demand_mw.values, color="black", linewidth=0.8, linestyle="--", label="Demand")
    ax1.set_title("Orkney Baseline: Wind Generation vs Demand (2019)", fontsize=10)
    ax1.set_ylabel("Power (MW)")
    ax1.legend(loc="upper right", fontsize=8)

    # --- Panel 2: Wave and tidal generation, own scale --------------------------
    ax2.fill_between(tidal_mw.index, tidal_mw.values, color="#4a7c3f", alpha=0.75,
                      label=f"Tidal (Westray-South {TIDAL_CAPACITY_MW}MW)")
    ax2.fill_between(wave_mw.index, wave_mw.values, color="#e8a33d", alpha=0.75,
                      label=f"Wave (CorPower {wave_capacity_mw}MW)")
    ax2.set_title("Wave and Tidal Generation (own scale)", fontsize=10)
    ax2.set_ylabel("Power (MW)")
    ax2.legend(loc="upper right", fontsize=8)

    # --- Panel 3: Net balance (excess / shortfall) -------------------------------
    ax3.fill_between(balance_mw.index, balance_mw.clip(lower=0), color="#4a9c4a", alpha=0.85, label="Excess")
    ax3.fill_between(balance_mw.index, balance_mw.clip(upper=0), color="#d9534f", alpha=0.85, label="Shortfall")
    ax3.axhline(0, color="black", linewidth=0.6)
    ax3.set_title("Net Balance (Renewable - Demand)", fontsize=10)
    ax3.set_ylabel("Power (MW)")
    ax3.legend(loc="upper right", fontsize=8)

    # --- Panel 4: Monthly excess / shortfall -------------------------------------
    monthly_excess = excess_mw.groupby(excess_mw.index.to_period("M")).sum()
    monthly_shortfall = (-shortfall_mw).groupby(shortfall_mw.index.to_period("M")).sum()
    months = monthly_excess.index.to_timestamp()
    ax4.bar(months, monthly_excess.values, width=20, color="#4a9c4a", alpha=0.85, label="Excess")
    ax4.bar(months, monthly_shortfall.values, width=20, color="#d9534f", alpha=0.85, label="Shortfall")
    ax4.axhline(0, color="black", linewidth=0.6)
    ax4.set_title("Monthly Excess / Shortfall", fontsize=10)
    ax4.set_ylabel("Energy (MWh)")
    ax4.legend(loc="upper right", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = f"{figures_dir}/{scenario_id}_energy_balance_dashboard.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def make_all_dashboards(summary_df, hourly_df, wind_mw, demand_mw):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"\nGenerating {len(summary_df)} scenario dashboard(s)...")
    for _, row in summary_df.iterrows():
        scenario_id = row["scenario_id"]
        wave_capacity_mw = row["wave_capacity_mw"]
        wave_mw = hourly_df[f"wave_mw_{scenario_id}"]
        tidal_mw = hourly_df[f"tidal_mw_{scenario_id}"]
        make_scenario_dashboard(scenario_id, wave_capacity_mw, row, hourly_df, wind_mw, demand_mw,
                                 wave_mw, tidal_mw)
    print(f"All dashboards saved to {FIGURES_DIR}/")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 100)
    print("PRELIMINARY RENEWABLE-ONLY HOURLY ENERGY BALANCE SCREENING (PRE-LOPF)")
    print("=" * 100)

    wind_mw, n_wind_files = load_wind_generation()
    demand_mw = load_demand()
    wave_pu = load_normalised_profile(WAVE_PROFILE_FILE, "wave")
    tidal_pu = load_normalised_profile(TIDAL_PROFILE_FILE, "tidal")

    wind_mw, demand_mw, wave_pu, tidal_pu, n_common, any_missing = align_all_series(
        wind_mw, demand_mw, wave_pu, tidal_pu
    )

    annual_demand_mwh = validate_annual_demand(demand_mw)

    print("\n" + "=" * 100)
    print("CALCULATING 5 SCENARIOS")
    print("=" * 100)

    summaries = []
    hourly_frames = []
    for scenario_id, wave_capacity_mw in SCENARIOS.items():
        print(f"\n[{scenario_id}] wave={wave_capacity_mw} MW, tidal={TIDAL_CAPACITY_MW} MW (fixed), "
              f"wind=existing DRES fleet")
        summary, hourly = calculate_scenario(scenario_id, wave_capacity_mw, wind_mw, demand_mw, wave_pu, tidal_pu)
        summaries.append(summary)
        hourly_frames.append(hourly)
        print(f"  Total renewable: {summary['total_renewable_generation_mwh']:,.1f} MWh | "
              f"Shortfall: {summary['shortfall_energy_mwh']:,.1f} MWh ({summary['shortfall_hours']} hrs) | "
              f"Excess: {summary['excess_energy_mwh']:,.1f} MWh ({summary['excess_hours']} hrs)")

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    hourly_df = pd.DataFrame({"demand_mw": demand_mw, "wind_mw": wind_mw, "tidal_pu": tidal_pu, "wave_pu": wave_pu})
    for h in hourly_frames:
        hourly_df = hourly_df.join(h)
    hourly_df.index.name = "DateTime"
    hourly_df.to_csv(HOURLY_CSV)

    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print(summary_df.to_string(index=False))

    make_all_dashboards(summary_df, hourly_df, wind_mw, demand_mw)

    print("\n" + "=" * 100)
    print("VALIDATION SUMMARY")
    print("=" * 100)
    print(f"Wind files loaded:              {n_wind_files}")
    print(f"Total wind annual generation:   {wind_mw.sum():,.1f} MWh")
    print(f"Demand annual total:            {annual_demand_mwh:,.1f} MWh")
    print(f"Wave profile row count:         {len(wave_pu)}")
    print(f"Tidal profile row count:        {len(tidal_pu)}")
    print(f"Common aligned timestamps:      {n_common}")
    print(f"Missing values found:           {any_missing}")
    print(f"Scenarios generated:            {len(summary_df)} / 5")
    print(f"Summary CSV:                    {SUMMARY_CSV}")
    print(f"Hourly CSV:                     {HOURLY_CSV}")
    print(f"Figures directory:              {FIGURES_DIR}/")


if __name__ == "__main__":
    main()