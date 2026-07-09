"""
extract_pure_optimized_balance.py (corrected)

Extracts a full HOURLY energy balance and dispatched energy cost using ONLY
the final optimized outputs stored inside the solved scenario .nc files.
No external generation profiles or availability tracking curves are used.

Corrections applied:
  1. The demand fallback ("if not network.loads_t.p.empty else 0.0") could
     silently zero out an entire scenario's demand if loads_t.p ever came
     back empty (e.g. an unsolved or corrupted file), making the balance
     check LOOK clean (near-zero) for the wrong reason. Replaced with an
     explicit check that raises a clear error instead of silently
     proceeding with fabricated zero demand.
  2. Storage discharge cost was missing entirely -- Orkney Storage Park has
     marginal_cost=5 (non-zero), so the previous "hourly_dispatched_energy_cost"
     could never reconcile with the network's actual solved objective. Added
     cost_energy_battery_main / cost_energy_battery_secondary (discharge-only,
     matching PyPSA's actual costing convention -- marginal_cost applies to
     p_dispatch, not to charging), plus an explicit reconciliation check
     against network.objective so any remaining gap is caught, not assumed.
"""

import os
import pypsa
import pandas as pd
import numpy as np

OUTPUT_DIR = "DATA/outputs/wave_storage_core_v2"
HOURLY_DIR = f"{OUTPUT_DIR}/hourly_balance"

FILES = {
    "R01": "R01_wave0MW_batt2x.nc",
    "R02": "R02_wave5MW_batt2x.nc",
    "R06": "R06_wave7MW_batt2x.nc",
    "R03": "R03_wave0MW_batt3x.nc",
    "R04": "R04_wave5MW_batt3x.nc",
    "R07": "R07_wave7MW_batt3x.nc",
}

STORAGE_UNIT = "Orkney Storage Park"
STORAGE_UNIT_SECONDARY = "KIRKWA3A_Storage"
GAS_GEN = "gas_turbine(Oil_station)"
SLACK_GEN = "Slack_generator"
WAVE_GEN = "Wave Generator"
TIDAL_GEN = "EDAY Tidal Generator"


def discharge_series(network, su):
    """Discharge-only portion of a storage unit's dispatch, for costing --
    PyPSA's objective charges marginal_cost against DISCHARGE (p_dispatch),
    not net power, so charging is not separately costed."""
    if hasattr(network.storage_units_t, "p_dispatch") and su in network.storage_units_t.p_dispatch.columns:
        return network.storage_units_t.p_dispatch[su]
    return network.storage_units_t.p[su].clip(lower=0)


def extract_hourly_from_nc(network, scenario_id):
    weights = network.snapshot_weightings["generators"]
    wind_gens = [g for g in network.generators.index if "WindFarm" in g]

    df = pd.DataFrame(index=network.snapshots)
    df.index.name = "snapshot"

    # ── 1. Pure Optimized Dispatches (Directly from the .nc File) ─────────────
    # System Demand
    # FIX #1: do NOT silently fall back to 0.0 if loads_t.p is empty -- an
    # empty loads_t.p means this network was never solved (or the file is
    # corrupted), and silently treating demand as zero would make the
    # balance_residual_mw check look falsely clean. Raise instead.
    if network.loads_t.p.empty:
        raise ValueError(
            f"[{scenario_id}] network.loads_t.p is EMPTY -- this network does not contain solved "
            f"dispatch results (was it actually optimized before saving?). Refusing to silently "
            f"treat demand as zero."
        )
    df["demand_mw"] = network.loads_t.p.sum(axis=1)

    # Generator Outputs
    df["wind_mw"] = network.generators_t.p[wind_gens].sum(axis=1) if wind_gens else 0.0
    df["wave_mw"] = network.generators_t.p[WAVE_GEN] if WAVE_GEN in network.generators.index else 0.0
    df["tidal_mw"] = network.generators_t.p[TIDAL_GEN] if TIDAL_GEN in network.generators.index else 0.0
    df["gas_mw"] = network.generators_t.p[GAS_GEN] if GAS_GEN in network.generators.index else 0.0

    # Interconnector Flows (Imports vs Exports based on optimized slack sign)
    df["mainland_import_mw"] = network.generators_t.p[SLACK_GEN].clip(lower=0)
    df["mainland_export_mw"] = -network.generators_t.p[SLACK_GEN].clip(upper=0)

    # Battery Performance Outputs
    df["battery_main_net_mw"] = network.storage_units_t.p[STORAGE_UNIT]
    df["battery_secondary_net_mw"] = network.storage_units_t.p[STORAGE_UNIT_SECONDARY]

    # ── 2. Hourly Physics Equilibrium Verification ───────────────────────────
    total_generation = (df["wind_mw"] + df["wave_mw"] + df["tidal_mw"] + df["gas_mw"]
                         + df["mainland_import_mw"] - df["mainland_export_mw"])
    total_storage_net = df["battery_main_net_mw"] + df["battery_secondary_net_mw"]

    # This should be ~0 because it uses the actual math solution inside the file
    df["balance_residual_mw"] = total_generation + total_storage_net - df["demand_mw"]

    # ── 3. Dispatched Energy Cost (Using Node Matrix Multiplication) ──────────
    if wind_gens:
        df["cost_energy_wind"] = (network.generators_t.p[wind_gens]
                                  .mul(network.generators.loc[wind_gens, "marginal_cost"], axis=1)
                                  .sum(axis=1))
    else:
        df["cost_energy_wind"] = 0.0

    df["cost_energy_wave"] = df["wave_mw"] * network.generators.loc[WAVE_GEN, "marginal_cost"] if WAVE_GEN in network.generators.index else 0.0
    df["cost_energy_tidal"] = df["tidal_mw"] * network.generators.loc[TIDAL_GEN, "marginal_cost"] if TIDAL_GEN in network.generators.index else 0.0
    df["cost_energy_gas"] = df["gas_mw"] * network.generators.loc[GAS_GEN, "marginal_cost"] if GAS_GEN in network.generators.index else 0.0

    # Cost applied strictly to what we imported
    df["cost_energy_mainland"] = df["mainland_import_mw"] * network.generators.loc[SLACK_GEN, "marginal_cost"]

    # FIX #2: storage discharge cost -- previously missing entirely, even
    # though Orkney Storage Park has marginal_cost=5 (non-zero). Discharge-
    # only, matching PyPSA's actual objective formulation.
    df["cost_energy_battery_main"] = (
        discharge_series(network, STORAGE_UNIT) * network.storage_units.loc[STORAGE_UNIT, "marginal_cost"]
    )
    df["cost_energy_battery_secondary"] = (
        discharge_series(network, STORAGE_UNIT_SECONDARY) * network.storage_units.loc[STORAGE_UNIT_SECONDARY, "marginal_cost"]
    )

    # ── 4. Total Aggregation ──────────────────────────────────────────────────
    cost_cols = ["cost_energy_wind", "cost_energy_wave", "cost_energy_tidal", "cost_energy_gas",
                 "cost_energy_mainland", "cost_energy_battery_main", "cost_energy_battery_secondary"]
    df["hourly_dispatched_energy_cost"] = df[cost_cols].sum(axis=1)
    df["weighted_hourly_energy_cost"] = df["hourly_dispatched_energy_cost"] * weights

    df["scenario_id"] = scenario_id
    return df


def main():
    os.makedirs(HOURLY_DIR, exist_ok=True)
    summary_rows = []

    for scenario_id, filename in FILES.items():
        path = f"{OUTPUT_DIR}/{filename}"
        if not os.path.exists(path):
            print(f"[{scenario_id}] File not found at: {path} -- skipping.")
            continue

        # Load network purely from the saved scenario file
        network = pypsa.Network()
        network.import_from_netcdf(path)

        df = extract_hourly_from_nc(network, scenario_id)

        max_residual = df["balance_residual_mw"].abs().max()
        total_energy_cost = df["weighted_hourly_energy_cost"].sum()

        # Reconciliation check against the network's actual solved objective --
        # now that storage discharge cost is included, this should close to
        # (near) zero. If it doesn't, something is still uncosted or the
        # discharge-only assumption doesn't match this PyPSA version.
        objective = network.objective
        cost_gap = total_energy_cost - objective

        out_path = f"{HOURLY_DIR}/{scenario_id}_pure_optimized_balance.csv"
        df.to_csv(out_path)

        summary_rows.append({
            "scenario": scenario_id,
            "max_balance_residual_mw": max_residual,
            "total_energy_dispatched_cost": total_energy_cost,
            "network_objective": objective,
            "cost_gap": cost_gap,
            "output_file": out_path,
        })

        print(f"[{scenario_id}] Processed {len(df)} snapshots from .nc")
        print(f"    Worst balance error: {max_residual:.6e} MW")
        print(f"    Dispatched Energy Cost: £{total_energy_cost:,.2f}  |  "
              f"Network objective: £{objective:,.2f}  |  Gap: £{cost_gap:,.4f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{HOURLY_DIR}/pure_nc_balance_summary.csv", index=False)
    print("\n" + "=" * 100)
    print("PURE .NC OPTIMIZED BALANCE SUMMARY")
    print("=" * 100)
    print(summary_df.to_string(index=False))

    if (summary_df["cost_gap"].abs() > 1.0).any():
        print("\n*** WARNING: one or more scenarios show a cost gap > £1.00 -- the cost breakdown does ***")
        print("*** NOT fully reconcile with the solved objective. Investigate before trusting these    ***")
        print("*** figures as a precise decomposition of system cost.                                   ***")
    else:
        print("\nAll scenarios reconcile cleanly (cost gap < £1.00) -- the cost breakdown is a valid")
        print("decomposition of each scenario's solved objective.")


if __name__ == "__main__":
    main()