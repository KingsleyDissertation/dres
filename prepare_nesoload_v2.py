"""
Converts the NESO-sourced demanddata_2019.csv (a single half-hourly total
Orkney demand figure) into the per-bus Load_Profile.csv format that
base_model.py's add_loads_to_network() expects.

WHAT THIS DOES:
  1. Loads demanddata_2019.csv, parses SETTLEMENT_DATE + SETTLEMENT_PERIOD
     into real timestamps, and resamples the half-hourly "ORKNEY LOAD (MW)"
     column to hourly (mean of each pair of half-hour readings).
  2. Loads the EXISTING Load_Profile.csv (old/uncorrected) purely to extract
     each bus's PROPORTION of total demand -- i.e. its relative share of the
     whole, not its absolute magnitude. This assumes the old file's relative
     shape between buses is still a reasonable distribution key, even though
     its overall total was wrong.
  3. Redistributes the new NESO hourly total across all 17 buses using those
     proportions, so each bus's new load = bus_share(t) * new_total(t).
  4. Reactive power: since NESO has no reactive data, each bus's existing
     reactive/active ratio (from the old file) is preserved and applied to
     the new active power series.
  5. Writes the result in the EXACT same column structure and DD/MM/YYYY
     HH:MM date format as the original, so base_model.py can read it with
     no changes.

IMPORTANT ASSUMPTION TO STATE IN YOUR METHODOLOGY:
  The per-bus SPATIAL distribution (which bus gets what share of total
  demand) still comes from the old, uncorrected file -- only the TOTAL
  Orkney-wide magnitude and its temporal shape are corrected using NESO data.
  This is a reasonable approach if the old per-bus proportions were derived
  from real network/customer data and only the aggregate calibration was
  wrong, but it is an assumption, not something this script can verify on
  its own. State this explicitly.
"""

import pandas as pd
import numpy as np
import os

OLD_LOAD_PROFILE = "DATA/inputs/Load_Profile_OLD_BACKUP.csv"  # the GENUINE original -- read shares from here, not from the (already overwritten) current file
NESO_RAW_FILE = "DATA/inputs/demanddata_2019.csv"
OUTPUT_FILE = "DATA/inputs/Load_Profile.csv"  # this gets overwritten -- bus shares always come from the backup instead
BACKUP_FILE = "DATA/inputs/Load_Profile_OLD_BACKUP.csv"


def load_neso_hourly_total():
    print(f"Loading NESO raw data: {NESO_RAW_FILE}")
    df = pd.read_csv(NESO_RAW_FILE, sep=";")
    print(f"  Raw shape: {df.shape}")
    print(f"  Columns  : {df.columns.tolist()}")

    # UK electricity settlement periods are defined in LOCAL (Europe/London)
    # wall-clock time, not a fixed UTC offset. Two days per year have a
    # non-standard number of periods:
    #   - Spring-forward (clocks go forward, e.g. 31 Mar 2019): only 46
    #     periods, since the 01:00-02:00 hour does not exist that day.
    #   - Autumn fall-back (clocks go back, e.g. 27 Oct 2019): 50 periods,
    #     since the 01:00-02:00 hour occurs twice.
    # A naive (period-1)*30-minute offset from midnight gets this wrong on
    # both days, silently shifting/dropping an hour. Instead, we localize to
    # Europe/London (which correctly handles the skipped/repeated hour) and
    # then convert to a plain continuous UTC-style clock, matching the
    # DST-free hourly convention used by the wind/wave/tidal resource files.
    df["SETTLEMENT_DATE"] = pd.to_datetime(df["SETTLEMENT_DATE"], dayfirst=True)
    minutes_offset = (df["SETTLEMENT_PERIOD"] - 1) * 30
    naive_local = df["SETTLEMENT_DATE"] + pd.to_timedelta(minutes_offset, unit="m")

    localized = naive_local.dt.tz_localize(
        "Europe/London",
        ambiguous="NaT",       # autumn: mark the repeated hour's second occurrence as NaT, drop it
        nonexistent="NaT",     # spring: mark the skipped hour as NaT, drop it
    )
    n_dropped = localized.isna().sum()
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} row(s) falling in a DST-ambiguous/nonexistent local time "
              f"(expected: a few rows around the spring/autumn clock changes)")

    df["DateTime"] = localized.dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.dropna(subset=["DateTime"]).set_index("DateTime").sort_index()

    orkney_col = "ORKNEY LOAD (MW)"
    if orkney_col not in df.columns:
        raise KeyError(f"Expected column '{orkney_col}' not found. Columns are: {df.columns.tolist()}")

    half_hourly = df[orkney_col]
    print(f"  Half-hourly rows after DST correction: {len(half_hourly)}")
    print(f"  Date range: {half_hourly.index.min()} to {half_hourly.index.max()}")

    # Resample to hourly by averaging each pair of half-hour readings
    hourly = half_hourly.resample("h").mean()
    n_missing_hourly = hourly.isna().sum()
    if n_missing_hourly > 0:
        print(f"  WARNING: {n_missing_hourly} hourly slot(s) still have no data after resampling -- "
              f"forward-filling from the previous hour as a fallback.")
        hourly = hourly.ffill()
    print(f"  Resampled to hourly: {len(hourly)} rows")
    print(f"  Hourly min/mean/max: {hourly.min():.2f} / {hourly.mean():.2f} / {hourly.max():.2f} MW")

    return hourly


def load_old_profile_shares():
    print(f"\nLoading existing (old) profile for bus proportions: {OLD_LOAD_PROFILE}")
    df = pd.read_csv(OLD_LOAD_PROFILE)
    df["DateTime"] = pd.to_datetime(df["DateTime"], dayfirst=True, format="mixed")
    df = df.set_index("DateTime").sort_index()

    active_cols = [c for c in df.columns if "[MW]" in c]
    reactive_cols = [c for c in df.columns if "[MVar]" in c]
    print(f"  Active columns  : {len(active_cols)}")
    print(f"  Reactive columns: {len(reactive_cols)}")

    old_total = df[active_cols].sum(axis=1)
    print(f"  Old total demand -- min/mean/max: {old_total.min():.2f} / {old_total.mean():.2f} / {old_total.max():.2f} MW")

    # Per-bus share of total active power, at every hour. Guard against
    # division by zero at any hour where old_total happens to be 0.
    bus_shares = df[active_cols].div(old_total.replace(0, np.nan), axis=0)
    bus_shares = bus_shares.fillna(bus_shares.mean())  # fallback to average share if any hour was NaN

    # Per-bus reactive/active ratio (kept constant using each bus's own annual ratio,
    # since NESO gives no reactive information at all)
    reactive_ratio = {}
    for a_col in active_cols:
        bus_name = a_col.replace(" [MW]", "")
        r_col = f"{bus_name} [MVar]"
        if r_col in reactive_cols:
            total_active_bus = df[a_col].sum()
            total_reactive_bus = df[r_col].sum()
            reactive_ratio[bus_name] = (total_reactive_bus / total_active_bus) if total_active_bus != 0 else 0.0
        else:
            reactive_ratio[bus_name] = 0.0

    return bus_shares, reactive_ratio, active_cols, df.index


def main():
    # ── Safety: back up the current file before overwriting, if not already done ──
    if os.path.exists(OUTPUT_FILE) and not os.path.exists(BACKUP_FILE):
        import shutil
        shutil.copy(OUTPUT_FILE, BACKUP_FILE)
        print(f"Backed up current profile to: {BACKUP_FILE}")
    elif os.path.exists(BACKUP_FILE):
        print(f"Backup already exists at {BACKUP_FILE} -- using it as the source of bus proportions "
              f"(not the possibly-already-converted current Load_Profile.csv).")

    hourly_total = load_neso_hourly_total()
    bus_shares, reactive_ratio, active_cols, old_index = load_old_profile_shares()

    # ── Align the two on a common hourly index ──────────────────────────────
    common_index = old_index.intersection(hourly_total.index)
    print(f"\nCommon hourly timestamps between NESO data and old profile: {len(common_index)} "
          f"(8760 expected for a full year match)")
    if len(common_index) == 0:
        raise ValueError("No overlapping timestamps found between NESO data and old Load_Profile.csv -- "
                          "check date ranges/years match.")

    hourly_total = hourly_total.loc[common_index]
    bus_shares = bus_shares.loc[common_index]

    # ── Redistribute new total across buses using old proportions ───────────
    new_active = bus_shares.multiply(hourly_total, axis=0)
    new_active.columns = active_cols  # restore original " [MW]" column names

    # ── Reactive power: apply each bus's own reactive/active ratio ──────────
    new_reactive = pd.DataFrame(index=common_index)
    for a_col in active_cols:
        bus_name = a_col.replace(" [MW]", "")
        r_col = f"{bus_name} [MVar]"
        new_reactive[r_col] = new_active[a_col] * reactive_ratio[bus_name]

    # ── Combine, format, and save ────────────────────────────────────────────
    result = pd.concat([new_active, new_reactive], axis=1)
    result = result.sort_index()  # guarantee chronological order regardless of upstream ordering
    result.index.name = "DateTime"
    result = result.reset_index()
    result["DateTime"] = result["DateTime"].dt.strftime("%d/%m/%Y %H:%M")

    result.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved corrected demand profile: {OUTPUT_FILE}")
    print(f"Shape: {result.shape} (8760 rows x 1 DateTime + 17 active + 17 reactive expected)")

    # ── Final sanity check: confirm the written file is genuinely monotonic ──
    verify_df = pd.read_csv(OUTPUT_FILE)
    verify_df["DateTime"] = pd.to_datetime(verify_df["DateTime"], dayfirst=True)
    is_monotonic = verify_df["DateTime"].is_monotonic_increasing
    print(f"Final file DateTime is monotonic increasing: {is_monotonic}")
    if not is_monotonic:
        print("  WARNING: output is still not sorted correctly -- investigate further before using this file.")

    check_active = [c for c in result.columns if "[MW]" in c]
    new_total_check = result[check_active].sum(axis=1)
    print(f"\nNew total demand -- min/mean/max: {new_total_check.min():.2f} / "
          f"{new_total_check.mean():.2f} / {new_total_check.max():.2f} MW")
    print(f"New total annual demand: {new_total_check.sum():.1f} MWh")


if __name__ == "__main__":
    main()