"""
R01_wave0MW_batt2x.py

Runs R01 only (wave=0MW, battery=2x) and saves the result.

Includes:
  - Collapsed storage: single combined unit (Orkney Storage Park), with
    KIRKWA3A_Storage removed.
  - Renewable dispatch priority via marginal cost: Wave (29.90) < Tidal
    (29.99) < Wind (30.00).
  - Interconnector constraint: 21.05 MVA on the two real subsea cables only.

This is the production run for R01 -- it actually saves the solved network.
For a detailed diagnostic/validation pass instead, use validate_optimize_r01.py.
"""

import pypsa
import pandas as pd
import numpy as np
import os

BASE_NETWORK_FILE = "DATA/outputs/network_wave5MW_tidal7.2MW.nc"
OUTPUT_DIR = "DATA/outputs"

S_NOM = 21.05
INTERCONNECTOR_LINES = [
    "19330 - 86018", "86018 - 86170",
    "19330 - 86019", "86019 - 86171",
]

WAVE_GEN = "Wave Generator"
TIDAL_GEN = "EDAY Tidal Generator"

WAVE_MARGINAL_COST = 29.90
TIDAL_MARGINAL_COST = 29.99
WIND_MARGINAL_COST = 30.00

STORAGE_UNIT = "Orkney Storage Park"
STORAGE_UNIT_TO_REMOVE = "KIRKWA3A_Storage"
STORAGE_BASE_PNOM = 2.0 + 2.585   # = 4.585 MW combined (1x)
STORAGE_MAX_HOURS = 8
STORAGE_MARGINAL_COST = 5
STORAGE_CYCLIC = False

# R01 settings
WAVE_MW = 0
BATTERY_MULT = 2
OUTPUT_FILE = f"{OUTPUT_DIR}/R01_wave{WAVE_MW}MW_batt{BATTERY_MULT}x.nc"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Running R01: wave={WAVE_MW}MW, battery={BATTERY_MULT}x")

    network = pypsa.Network()
    network.import_from_netcdf(BASE_NETWORK_FILE)

    # CRITICAL FIX: clear generator p_set before optimize(). p_set was
    # deliberately populated in base_model.py to fix pf(), but optimize()
    # treats any non-NaN p_set as a hard equality constraint, fixing
    # dispatch rigidly (including Slack, pinned at exactly 0.0) and making
    # the model infeasible whenever fixed generation != demand (almost
    # always). Confirmed via direct testing. loads_t.p_set is untouched.
    network.generators.loc[:, "p_set"] = np.nan
    network.generators_t.p_set = pd.DataFrame(index=network.snapshots)

    # Wave capacity
    network.generators.loc[WAVE_GEN, "p_nom"] = WAVE_MW

    # Renewable dispatch priority via marginal cost
    network.generators.loc[WAVE_GEN, "marginal_cost"] = WAVE_MARGINAL_COST
    network.generators.loc[TIDAL_GEN, "marginal_cost"] = TIDAL_MARGINAL_COST
    wind_gens = [g for g in network.generators.index if "WindFarm" in g]
    network.generators.loc[wind_gens, "marginal_cost"] = WIND_MARGINAL_COST

    # Collapse storage into one combined unit.
    # IMPORTANT: do NOT remove KIRKWA3A_Storage entirely via network.remove().
    # Doing so leaves its bus (KIRKWA3A) structurally unchanged but strips out
    # a component that bus may have depended on to be feasible at all (e.g.
    # if combined with the tight interconnector constraint and the zero-
    # reactance lines PyPSA warned about). Instead, shrink it to a negligible
    # epsilon (same technique already used for "no battery" scenarios) and
    # put ALL combined capacity into Orkney Storage Park. This preserves the
    # network's structure exactly -- same buses, same components present --
    # while still functionally representing "one combined battery".
    network.storage_units.loc[STORAGE_UNIT, "max_hours"] = STORAGE_MAX_HOURS
    network.storage_units.loc[STORAGE_UNIT, "marginal_cost"] = STORAGE_MARGINAL_COST
    network.storage_units.loc[STORAGE_UNIT, "cyclic_state_of_charge"] = STORAGE_CYCLIC
    network.storage_units.loc[STORAGE_UNIT, "p_nom"] = STORAGE_BASE_PNOM * BATTERY_MULT

    if STORAGE_UNIT_TO_REMOVE in network.storage_units.index:
        # Epsilon of 0.01 MW (10 kW) rather than 0.000001 MW: still fully
        # negligible relative to a multi-MW battery or the 55 MW wind fleet,
        # but avoids an extreme numerical range against the Slack generator's
        # 1,000,000 MW p_nom, which triggered a solver presolve warning
        # ("excessively small row bounds") and a false INFEASIBLE result on
        # the first attempt at this fix.
        network.storage_units.loc[STORAGE_UNIT_TO_REMOVE, "p_nom"] = 0.01

    # Grid constraint
    network.lines.loc[INTERCONNECTOR_LINES, "s_nom"] = S_NOM

    # Reset storage time series and initial SOC.
    # IMPORTANT FIX (round 2): reindex(..., fill_value=0.0) only fills
    # genuinely NEW rows/columns -- it does NOT overwrite NaN already
    # present in EXISTING cells. Since the dataframe already had entries for
    # every snapshot (just NaN, left over from the pf() rebuild), reindexing
    # onto the same index was a silent no-op and every NaN remained exactly
    # as it was. Building a brand new all-zero DataFrame from scratch (not
    # modifying the old one at all) guarantees no NaN can survive.
    for attr in list(network.storage_units_t.keys()):
        df = network.storage_units_t[attr]
        network.storage_units_t[attr] = pd.DataFrame(
            0.0, index=network.snapshots, columns=df.columns
        )
    network.storage_units.loc[STORAGE_UNIT, "state_of_charge_initial"] = 0.0

    # Solve
    status, condition = network.optimize(solver_name="highs")
    print(f"Status: {status} | Condition: {condition}")

    if status != "ok":
        print("Optimization did not solve -- not saving.")
        return

    print(f"Objective (system cost): {network.objective:.2f}")

    # Save
    network.export_to_netcdf(OUTPUT_FILE)
    print(f"Saved: {OUTPUT_FILE}")

    # Brief summary
    total_demand = network.loads_t.p_set.sum().sum()
    total_gen = network.generators_t.p.sum().sum()
    print(f"\nTotal generation: {total_gen:.1f} MWh | Total demand: {total_demand:.1f} MWh")
    print(f"Wave dispatch: {network.generators_t.p[WAVE_GEN].sum():.1f} MWh")
    print(f"Tidal dispatch: {network.generators_t.p[TIDAL_GEN].sum():.1f} MWh")
    print(f"Wind dispatch: {network.generators_t.p[wind_gens].sum().sum():.1f} MWh")
    print(f"Storage discharge: {network.storage_units_t.p[STORAGE_UNIT].clip(lower=0).sum():.1f} MWh")
    print(f"Storage charge: {network.storage_units_t.p[STORAGE_UNIT].clip(upper=0).sum():.1f} MWh")


if __name__ == "__main__":
    main()