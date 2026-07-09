"""
run_nine_scaling_scenarios.py

Batch runner for the 9 NEW scenarios completing the wave-battery scaling
study's full 3x5 factorial grid (battery: 2x/3x/4x; wave: 0/5/7/15/30 MW).

This script is a structural derivative of the already validated
run_six_core_scenarios.py -- every constant, every step in run_scenario(),
every validation call, and every metric definition that OVERLAPS with the
six-scenario script is preserved unchanged. This revision ADDS: a scenario-
grid integrity check, explicit non-extendable capacity guarantees, direct
use of PyPSA's own p_store/p_dispatch variables (rather than clipped net
power), gas-turbine metrics, full renewable availability/dispatch/
curtailment splits, per-scenario incremental CSV saves, and an isolated
test-mode output file. None of these additions change any of the metrics
that already existed in the six-scenario script -- see the accompanying
change log for why each addition is a genuine no-op against the already-
validated six-scenario results (all six showed zero extendable components,
zero simultaneous charge+discharge hours, and wind marginal cost already
matching baseline).

Research question:
    How does increasing wave generation affect battery storage operation,
    and how does this effect change as battery capacity increases from 2x
    to 3x and 4x?

EFC is used as a normalised cycling-intensity metric alongside absolute
charging, discharge, throughput, operating hours, SOC behaviour, renewable
curtailment, gas generation and mainland imports. Increased or decreased
EFC is NOT automatically "better" -- the final interpretation must
distinguish absolute battery utilisation, normalised cycling intensity, SOC
behaviour, renewable displacement, wave curtailment, wind curtailment, gas
displacement, import displacement, and operating-cost effects as separate,
non-interchangeable findings.

Full 3x5 grid this study now covers (6 already validated + 9 new):
    Battery   0 MW     5 MW     7 MW     15 MW    30 MW
    2x        R01 *    R02 *    R06 *    R08      R09
    3x        R03 *    R04 *    R07 *    R10      R11
    4x        R12      R13      R14      R15      R16
    (* = already run and validated in run_six_core_scenarios.py -- NOT
    rerun here; reuse those existing results directly when building the
    combined analysis.)

*** IMPORTANT ID-COLLISION WARNING ***
The ORIGINAL run_six_core_scenarios.py also defines an
OPTIONAL_SENSITIVITY_SCENARIOS list, which reuses the IDs "R05" and
"R08"-"R13" for a COMPLETELY DIFFERENT, unrelated set of scenarios (10x
battery and no-battery/high-wave sensitivity tests from an earlier, broader
13-scenario design). Do NOT run OPTIONAL_SENSITIVITY_SCENARIOS from
run_six_core_scenarios.py alongside this script. This script's R08-R16
refer ONLY to the wave-battery scaling grid defined above.

*** NOT INTRODUCED HERE ***
This script does NOT impose the 7 MW gas cap or 20 MW mainland import cap
used in the separate import-cap sensitivity experiment
(run_importcap_scenarios.py). These nine scenarios retain the SAME grid
assumptions as the original six core scenarios -- gas and Slack (mainland
import) are left at their original, unconstrained base-network capacities
throughout.
"""

import os
import sys
import pypsa
import pandas as pd
import numpy as np

# ── Constants (IDENTICAL to run_six_core_scenarios.py) ──────────────────────
BASE_NETWORK_FILE = "DATA/outputs/network_wave5MW_tidal7.2MW.nc"

OUTPUT_DIR = "DATA/outputs/wave_storage_scaling_v1"
SUMMARY_FILE = f"{OUTPUT_DIR}/nine_scaling_scenarios_summary.csv"
TEST_SUMMARY_FILE = f"{OUTPUT_DIR}/test_r08_summary.csv"

INTERCONNECTOR_LINES = [
    "19330 - 86018", "86018 - 86170",
    "19330 - 86019", "86019 - 86171",
]
INTERCONNECTOR_S_NOM = 21.05

WAVE_GEN = "Wave Generator"
TIDAL_GEN = "EDAY Tidal Generator"
GAS_GEN = "gas_turbine(Oil_station)"

WAVE_MARGINAL_COST = 29.90
TIDAL_MARGINAL_COST = 29.99
WIND_MARGINAL_COST = 30.00

STORAGE_UNIT = "Orkney Storage Park"
STORAGE_UNIT_SECONDARY = "KIRKWA3A_Storage"
STORAGE_SECONDARY_EPSILON = 0.01
STORAGE_MAX_HOURS = 8
STORAGE_MARGINAL_COST = 5
STORAGE_CYCLIC = False

INCLUDE_OBJECTIVE_CONSTANT = False

SCENARIOS = [
    {"id": "R08", "wave_mw": 15, "battery_mult": 2},
    {"id": "R09", "wave_mw": 30, "battery_mult": 2},
    {"id": "R10", "wave_mw": 15, "battery_mult": 3},
    {"id": "R11", "wave_mw": 30, "battery_mult": 3},
    {"id": "R12", "wave_mw": 0,  "battery_mult": 4},
    {"id": "R13", "wave_mw": 5,  "battery_mult": 4},
    {"id": "R14", "wave_mw": 7,  "battery_mult": 4},
    {"id": "R15", "wave_mw": 15, "battery_mult": 4},
    {"id": "R16", "wave_mw": 30, "battery_mult": 4},
]

EXPECTED_NEW_COMBINATIONS = {
    (2, 15), (2, 30),
    (3, 15), (3, 30),
    (4, 0), (4, 5), (4, 7), (4, 15), (4, 30),
}


def validate_scaling_grid(scenarios):
    combinations = {(s["battery_mult"], s["wave_mw"]) for s in scenarios}

    if combinations != EXPECTED_NEW_COMBINATIONS:
        missing = EXPECTED_NEW_COMBINATIONS - combinations
        unexpected = combinations - EXPECTED_NEW_COMBINATIONS
        raise ValueError(
            f"Scaling scenario grid is incorrect. Missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )

    ids = [s["id"] for s in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("Scenario IDs must be unique.")


def build_output_filename(scenario, output_dir=OUTPUT_DIR):
    wave_tag = f"wave{scenario['wave_mw']}MW"
    if scenario["battery_mult"] is None:
        batt_tag = "noBatt"
    else:
        batt_tag = f"batt{scenario['battery_mult']}x"
    return f"{output_dir}/{scenario['id']}_{wave_tag}_{batt_tag}.nc"


def clear_stale_storage_data(network):
    result_attrs = [
        "p", "p_dispatch", "p_store", "q", "state_of_charge", "spill",
        "mu_upper", "mu_lower", "mu_state_of_charge_set", "mu_energy_balance",
    ]
    for attr in result_attrs:
        if attr in network.storage_units_t.keys():
            network.storage_units_t[attr] = pd.DataFrame(index=network.snapshots)

    fixed_schedule_attrs = ["p_set", "p_dispatch_set", "p_store_set", "state_of_charge_set"]
    for attr in fixed_schedule_attrs:
        if attr in network.storage_units.columns:
            network.storage_units.loc[:, attr] = np.nan
        if attr in network.storage_units_t.keys():
            network.storage_units_t[attr] = pd.DataFrame(index=network.snapshots)


def verify_baseline_storage_capacity(baseline_network, expected_mw=4.585, tolerance=1e-6):
    baseline_combined_storage = float(
        baseline_network.storage_units.loc[[STORAGE_UNIT, STORAGE_UNIT_SECONDARY], "p_nom"].sum()
    )
    if abs(baseline_combined_storage - expected_mw) > tolerance:
        raise ValueError(
            f"Baseline combined storage capacity ({baseline_combined_storage:.6f} MW) does not "
            f"match the expected value ({expected_mw} MW) -- the base network file may have "
            f"changed. Investigate before trusting any scaled scenario results."
        )
    return baseline_combined_storage


def run_scenario(scenario, output_dir=OUTPUT_DIR):
    network = pypsa.Network()
    network.import_from_netcdf(BASE_NETWORK_FILE)

    baseline_network = pypsa.Network()
    baseline_network.import_from_netcdf(BASE_NETWORK_FILE)
    baseline_combined_storage = verify_baseline_storage_capacity(baseline_network)

    network.generators.loc[:, "p_set"] = np.nan
    network.generators_t.p_set = pd.DataFrame(index=network.snapshots)

    network.generators.loc[WAVE_GEN, "p_nom"] = scenario["wave_mw"]
    network.generators.loc[WAVE_GEN, "p_nom_extendable"] = False

    network.generators.loc[WAVE_GEN, "marginal_cost"] = WAVE_MARGINAL_COST
    network.generators.loc[TIDAL_GEN, "marginal_cost"] = TIDAL_MARGINAL_COST
    wind_gens = [g for g in network.generators.index if "WindFarm" in g]
    network.generators.loc[wind_gens, "marginal_cost"] = WIND_MARGINAL_COST

    network.storage_units.loc[STORAGE_UNIT, "max_hours"] = STORAGE_MAX_HOURS
    network.storage_units.loc[STORAGE_UNIT, "marginal_cost"] = STORAGE_MARGINAL_COST
    network.storage_units.loc[STORAGE_UNIT, "cyclic_state_of_charge"] = STORAGE_CYCLIC
    network.storage_units.loc[STORAGE_UNIT_SECONDARY, "p_nom"] = STORAGE_SECONDARY_EPSILON

    if scenario["battery_mult"] is None:
        network.storage_units.loc[STORAGE_UNIT, "p_nom"] = 0.000001
    else:
        mult = scenario["battery_mult"]
        network.storage_units.loc[STORAGE_UNIT, "p_nom"] = baseline_combined_storage * mult

    network.storage_units.loc[STORAGE_UNIT, "p_nom_extendable"] = False
    network.storage_units.loc[STORAGE_UNIT_SECONDARY, "p_nom_extendable"] = False

    missing_lines = [ln for ln in INTERCONNECTOR_LINES if ln not in network.lines.index]
    if missing_lines:
        raise KeyError(
            f"Interconnector line(s) not found in network.lines: {missing_lines}. "
            "Line names may differ between network files — re-check with "
            "network.lines[['bus0','bus1']] before proceeding."
        )
    network.lines.loc[INTERCONNECTOR_LINES, "s_nom"] = INTERCONNECTOR_S_NOM
    network.lines.loc[INTERCONNECTOR_LINES, "s_nom_extendable"] = False

    clear_stale_storage_data(network)
    network.storage_units.loc[STORAGE_UNIT, "state_of_charge_initial"] = 0.0

    status, condition = network.optimize(
        solver_name="highs", include_objective_constant=INCLUDE_OBJECTIVE_CONSTANT
    )
    solve_ok = (str(status) == "ok") and (str(condition) == "optimal")

    result = {
        "id": scenario["id"],
        "wave_mw": scenario["wave_mw"],
        "battery_mult": scenario["battery_mult"] if scenario["battery_mult"] else 0,
        "status": status,
        "condition": condition,
        "optimisation_validation_pass": None,
        "critical_hour_pf_pass": None,
        "pf_slack_review_needed": None,
        "objective": None,
        "mainland_import_mwh": None,
        "mainland_export_mwh": None,
        "slack_active_hours": None,
        "slack_max_mw": None,
        "gas_generation_mwh": None,
        "gas_operating_hours": None,
        "gas_peak_dispatch_mw": None,
        "combined_gas_import_mwh": None,
        "wind_available_mwh": None, "wind_dispatch_mwh": None,
        "wind_curtail_mwh": None, "wind_curtail_pct": None,
        "wave_available_mwh": None, "wave_dispatch_mwh": None,
        "wave_curtail_mwh": None, "wave_curtail_pct": None,
        "tidal_available_mwh": None, "tidal_dispatch_mwh": None,
        "tidal_curtail_mwh": None, "tidal_curtail_pct": None,
        "total_renewable_curtail_mwh": None,
        "interconnector_max_loading_pct": None,
        "battery_power_capacity_mw": None,
        "battery_energy_capacity_mwh": None,
        "battery_charge_mwh": None,
        "battery_discharge_mwh": None,
        "battery_throughput_mwh": None,
        "battery_equivalent_full_cycles": None,
        "battery_charging_hours": None,
        "battery_discharging_hours": None,
        "battery_peak_charge_mw": None,
        "battery_peak_discharge_mw": None,
        "battery_mean_soc_mwh": None,
        "battery_max_soc_mwh": None,
        "battery_mean_soc_pct": None,
        "battery_hours_below_10pct": None,
        "battery_hours_above_80pct": None,
    }

    if not solve_ok:
        print(f"  [{scenario['id']}] Solve did not reach optimal: status={status}, condition={condition}")
        return result, network

    network.meta.update({
        "solver_status": str(status),
        "termination_condition": str(condition),
        "scenario_id": scenario["id"],
        "wave_mw": float(scenario["wave_mw"]),
        "battery_mult": float(scenario["battery_mult"]) if scenario["battery_mult"] is not None else None,
        "main_storage_cyclic": STORAGE_CYCLIC,
        "main_storage_initial_soc": 0.0,
        "wave_marginal_cost": WAVE_MARGINAL_COST,
        "tidal_marginal_cost": TIDAL_MARGINAL_COST,
        "wind_marginal_cost": WIND_MARGINAL_COST,
        "include_objective_constant": INCLUDE_OBJECTIVE_CONSTANT,
    })

    from validate_scenario import validate_scenario
    scenario_for_validation = dict(scenario)
    scenario_for_validation["status"] = status
    scenario_for_validation["condition"] = condition
    scenario_for_validation["allow_slack"] = True
    scenario_for_validation["allowed_marginal_cost_changes"] = {
        WAVE_GEN: WAVE_MARGINAL_COST,
        TIDAL_GEN: TIDAL_MARGINAL_COST,
        **{gen: WIND_MARGINAL_COST for gen in wind_gens},
    }
    scenario_for_validation["main_storage_cyclic"] = STORAGE_CYCLIC
    scenario_for_validation["main_storage_initial_soc"] = 0.0
    scenario_for_validation["include_objective_constant"] = INCLUDE_OBJECTIVE_CONSTANT
    validation_report = validate_scenario(network, baseline_network, scenario_for_validation, tolerance_mw=1e-4)
    result["optimisation_validation_pass"] = validation_report["overall_pass"]
    if not validation_report["overall_pass"]:
        print(f"  [{scenario['id']}] *** OPTIMISATION VALIDATION FAILED -- see [FAIL] lines above. "
              f"Result saved anyway but should NOT be trusted without investigation. ***")

    generator_weights = network.snapshot_weightings["generators"].reindex(network.snapshots).astype(float)
    storage_weights = (
        network.snapshot_weightings["stores"]
        if "stores" in network.snapshot_weightings.columns
        else network.snapshot_weightings["generators"]
    )
    storage_weights = storage_weights.reindex(network.snapshots).astype(float)

    def weighted_energy(series):
        return float(series.mul(generator_weights).sum())

    def weighted_storage_energy(series):
        return float(series.mul(storage_weights).sum())

    result["objective"] = network.objective

    slack_col = [g for g in network.generators.index if "Slack" in g]
    if slack_col:
        slack_p = network.generators_t.p[slack_col].sum(axis=1)
        result["mainland_import_mwh"] = weighted_energy(slack_p.clip(lower=0))
        result["mainland_export_mwh"] = weighted_energy(-slack_p.clip(upper=0))
        result["slack_active_hours"] = float((slack_p.abs() > 1e-4).mul(generator_weights).sum())
        result["slack_max_mw"] = float(slack_p.abs().max())

    gas_p = network.generators_t.p[GAS_GEN].reindex(network.snapshots).fillna(0.0).astype(float)
    result["gas_generation_mwh"] = weighted_energy(gas_p)
    result["gas_operating_hours"] = float((gas_p > 1e-4).mul(generator_weights).sum())
    result["gas_peak_dispatch_mw"] = float(gas_p.max())
    result["combined_gas_import_mwh"] = result["gas_generation_mwh"] + (result["mainland_import_mwh"] or 0.0)

    generator_p_max_pu = network.get_switchable_as_dense("Generator", "p_max_pu")

    def renewable_metrics(gen_list):
        if not gen_list:
            return {"available_mwh": 0.0, "dispatch_mwh": 0.0, "curtail_mwh": 0.0, "curtail_pct": np.nan}

        availability = pd.DataFrame({
            gen: (generator_p_max_pu[gen].clip(lower=0.0, upper=1.0) * float(network.generators.loc[gen, "p_nom"]))
            for gen in gen_list
        }).sum(axis=1)

        dispatch = network.generators_t.p[gen_list].sum(axis=1)

        maximum_violation = float((dispatch - availability).max())
        if maximum_violation > 1e-4:
            raise ValueError(
                f"Renewable dispatch exceeds availability by {maximum_violation:.6f} MW for {gen_list}."
            )

        curtailed = (availability - dispatch).clip(lower=0.0)

        available_mwh = weighted_energy(availability)
        dispatch_mwh = weighted_energy(dispatch)
        curtail_mwh = weighted_energy(curtailed)

        return {
            "available_mwh": available_mwh,
            "dispatch_mwh": dispatch_mwh,
            "curtail_mwh": curtail_mwh,
            "curtail_pct": (curtail_mwh / available_mwh * 100 if available_mwh > 0 else np.nan),
        }

    wave_gens = [g for g in network.generators.index if "Wave" in g]
    tidal_gens = [g for g in network.generators.index if "Tidal" in g]

    wind_metrics = renewable_metrics(wind_gens)
    wave_metrics = renewable_metrics(wave_gens)
    tidal_metrics = renewable_metrics(tidal_gens)

    result["wind_available_mwh"] = wind_metrics["available_mwh"]
    result["wind_dispatch_mwh"] = wind_metrics["dispatch_mwh"]
    result["wind_curtail_mwh"] = wind_metrics["curtail_mwh"]
    result["wind_curtail_pct"] = wind_metrics["curtail_pct"]

    result["wave_available_mwh"] = wave_metrics["available_mwh"]
    result["wave_dispatch_mwh"] = wave_metrics["dispatch_mwh"]
    result["wave_curtail_mwh"] = wave_metrics["curtail_mwh"]
    result["wave_curtail_pct"] = wave_metrics["curtail_pct"]

    result["tidal_available_mwh"] = tidal_metrics["available_mwh"]
    result["tidal_dispatch_mwh"] = tidal_metrics["dispatch_mwh"]
    result["tidal_curtail_mwh"] = tidal_metrics["curtail_mwh"]
    result["tidal_curtail_pct"] = tidal_metrics["curtail_pct"]

    result["total_renewable_curtail_mwh"] = (
        wind_metrics["curtail_mwh"] + wave_metrics["curtail_mwh"] + tidal_metrics["curtail_mwh"]
    )

    line_capacity = network.lines["s_nom"].astype(float).copy()
    if "s_nom_extendable" in network.lines.columns:
        extendable = network.lines["s_nom_extendable"].fillna(False).astype(bool)
        if extendable.any():
            if "s_nom_opt" in network.lines.columns:
                optimised_capacity = network.lines["s_nom_opt"].reindex(network.lines.index)
                valid_optimised_capacity = (
                    optimised_capacity.notna()
                    & np.isfinite(optimised_capacity)
                    & (optimised_capacity > 0)
                )
                use_optimised_capacity = extendable & valid_optimised_capacity
                line_capacity.loc[use_optimised_capacity] = optimised_capacity.loc[use_optimised_capacity]

                fallback_lines = extendable & ~valid_optimised_capacity
                if fallback_lines.any():
                    print(f"  [WARNING] Extendable line(s) without a valid s_nom_opt; falling back "
                          f"to s_nom: {fallback_lines[fallback_lines].index.tolist()}")

    line_s_max_pu_dense = network.get_switchable_as_dense("Line", "s_max_pu")
    line_s_max_pu = line_s_max_pu_dense.reindex(index=network.snapshots, columns=INTERCONNECTOR_LINES)

    missing_s_max_columns = [line for line in INTERCONNECTOR_LINES if line not in line_s_max_pu_dense.columns]
    if missing_s_max_columns:
        print(f"  [WARNING] Missing effective s_max_pu data for {missing_s_max_columns}; using 1.0.")

    line_s_max_pu = line_s_max_pu.fillna(1.0).astype(float)

    interconnector_limit = line_s_max_pu.mul(line_capacity[INTERCONNECTOR_LINES], axis=1)

    invalid_limit = ~np.isfinite(interconnector_limit) | (interconnector_limit <= 0)
    if invalid_limit.any().any():
        bad_lines = invalid_limit.any(axis=0)
        raise ValueError(f"Invalid effective interconnector capacity for: "
                          f"{bad_lines[bad_lines].index.tolist()}")

    interconnector_flow = pd.DataFrame(
        np.maximum(
            network.lines_t.p0[INTERCONNECTOR_LINES].abs(),
            network.lines_t.p1[INTERCONNECTOR_LINES].abs(),
        ),
        index=network.snapshots,
        columns=INTERCONNECTOR_LINES,
    )

    interconnector_loading_pct = (
        interconnector_flow.div(interconnector_limit.replace(0, np.nan)) * 100
    )
    result["interconnector_max_loading_pct"] = float(interconnector_loading_pct.max().max())

    battery_charge = (
        network.storage_units_t.p_store[STORAGE_UNIT].reindex(network.snapshots).fillna(0.0).astype(float)
    )
    battery_discharge = (
        network.storage_units_t.p_dispatch[STORAGE_UNIT].reindex(network.snapshots).fillna(0.0).astype(float)
    )
    battery_net_power = (
        network.storage_units_t.p[STORAGE_UNIT].reindex(network.snapshots).fillna(0.0).astype(float)
    )
    battery_soc = network.storage_units_t.state_of_charge[STORAGE_UNIT]

    battery_power_capacity = float(network.storage_units.loc[STORAGE_UNIT, "p_nom"])
    battery_energy_capacity = battery_power_capacity * float(
        network.storage_units.loc[STORAGE_UNIT, "max_hours"]
    )

    charge_mwh = weighted_storage_energy(battery_charge)
    discharge_mwh = weighted_storage_energy(battery_discharge)
    throughput_mwh = charge_mwh + discharge_mwh
    equivalent_full_cycles = (
        throughput_mwh / (2 * battery_energy_capacity) if battery_energy_capacity > 0 else np.nan
    )

    total_storage_weight = float(storage_weights.sum())
    if total_storage_weight <= 0:
        raise ValueError("Storage snapshot weights must have a positive total.")
    weighted_mean_soc = float(battery_soc.mul(storage_weights).sum() / total_storage_weight)

    result.update({
        "battery_power_capacity_mw": battery_power_capacity,
        "battery_energy_capacity_mwh": battery_energy_capacity,
        "battery_charge_mwh": charge_mwh,
        "battery_discharge_mwh": discharge_mwh,
        "battery_throughput_mwh": throughput_mwh,
        "battery_equivalent_full_cycles": equivalent_full_cycles,
        "battery_charging_hours": float((battery_charge > 1e-4).mul(storage_weights).sum()),
        "battery_discharging_hours": float((battery_discharge > 1e-4).mul(storage_weights).sum()),
        "battery_peak_charge_mw": float(battery_charge.max()),
        "battery_peak_discharge_mw": float(battery_discharge.max()),
        "battery_mean_soc_mwh": weighted_mean_soc,
        "battery_max_soc_mwh": float(battery_soc.max()),
        "battery_mean_soc_pct": (
            weighted_mean_soc / battery_energy_capacity * 100
            if battery_energy_capacity > 0 else np.nan
        ),
        "battery_hours_below_10pct": float(
            (battery_soc < 0.10 * battery_energy_capacity).mul(storage_weights).sum()
        ),
        "battery_hours_above_80pct": float(
            (battery_soc > 0.80 * battery_energy_capacity).mul(storage_weights).sum()
        ),
    })

    os.makedirs(output_dir, exist_ok=True)
    out_path = build_output_filename(scenario, output_dir=output_dir)
    network.export_to_netcdf(out_path)
    result["output_file"] = out_path

    return result, network


CONSOLE_SUMMARY_COLUMNS = [
    "id", "wave_mw", "battery_mult", "status", "condition",
    "optimisation_validation_pass", "objective",
    "battery_power_capacity_mw", "battery_energy_capacity_mwh",
    "battery_charge_mwh", "battery_discharge_mwh", "battery_throughput_mwh",
    "battery_equivalent_full_cycles", "battery_charging_hours",
    "battery_discharging_hours", "battery_mean_soc_pct",
    "mainland_import_mwh", "mainland_export_mwh",
    "gas_generation_mwh", "combined_gas_import_mwh",
    "wind_curtail_mwh", "wave_curtail_mwh", "tidal_curtail_mwh",
    "total_renewable_curtail_mwh",
    "interconnector_max_loading_pct",
]


def run_batch(scenarios=SCENARIOS, output_dir=OUTPUT_DIR, summary_file=SUMMARY_FILE):
    if scenarios is SCENARIOS:
        validate_scaling_grid(scenarios)

    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    total_scenarios = len(scenarios)

    print("=" * 100)
    print(f"RUNNING {total_scenarios} WAVE-BATTERY SCALING SCENARIO(S)")
    print("=" * 100)

    for i, scenario in enumerate(scenarios, start=1):
        print(f"\n[{i}/{total_scenarios}] {scenario['id']}: wave={scenario['wave_mw']}MW, "
              f"battery_mult={scenario['battery_mult']}")

        result, _ = run_scenario(scenario, output_dir=output_dir)
        all_results.append(result)

        running_df = pd.DataFrame(all_results)
        running_df.to_csv(summary_file, index=False)

        solve_ok = (str(result["status"]) == "ok") and (str(result["condition"]) == "optimal")
        if solve_ok:
            print(f"  -> Solved optimally. Objective={result['objective']:.2f} | "
                  f"Battery throughput={result['battery_throughput_mwh']:.1f} MWh | "
                  f"Battery EFC={result['battery_equivalent_full_cycles']:.1f} | "
                  f"Mainland import={result['mainland_import_mwh']:.1f} MWh | "
                  f"Gas={result['gas_generation_mwh']:.1f} MWh | "
                  f"Curtailed W/Wv/T={result['wind_curtail_mwh']:.1f}/"
                  f"{result['wave_curtail_mwh']:.1f}/{result['tidal_curtail_mwh']:.1f} MWh | "
                  f"Interconnector max loading={result['interconnector_max_loading_pct']:.1f}%")
            print(f"  -> Saved: {result['output_file']}")
        else:
            print(f"  -> Scenario did not solve optimally: "
                  f"status={result['status']}, condition={result['condition']}")

        print("\n  --- Running summary so far ---")
        print(running_df[CONSOLE_SUMMARY_COLUMNS].to_string(index=False))

    final_df = pd.DataFrame(all_results)
    final_df.to_csv(summary_file, index=False)

    print("\n" + "=" * 100)
    print(f"ALL {total_scenarios} SCENARIO(S) COMPLETE")
    print("=" * 100)
    print(final_df[CONSOLE_SUMMARY_COLUMNS].to_string(index=False))
    print(f"\nSaved FULL summary (all fields): {summary_file}")
    print("\nNOTE: critical_hour_pf_pass / pf_slack_review_needed are NOT populated by this batch --")
    print("run validate_critical_hours_pf() separately on each saved .nc file afterward if a fully")
    print("cross-checked summary is needed.")
    print("\nREMINDER: this study's full 3x5 grid also includes R01, R02, R03, R04, R06, R07 from")
    print("run_six_core_scenarios.py (already validated -- not rerun here). Combine both summary CSVs")
    print("for the complete 15-cell analysis.")

    return final_df


def main():
    """
    Usage:
        python run_nine_scaling_scenarios.py            -> run all 9 new scenarios
                                                             -> saves to SUMMARY_FILE
        python run_nine_scaling_scenarios.py --test-r08  -> run ONLY R08 (15 MW, 2x
                                                             battery)
                                                             -> saves to TEST_SUMMARY_FILE
                                                             (does NOT overwrite the
                                                             production summary)

    RECOMMENDED TESTING ORDER before running the full batch:
        1. Test R08 (--test-r08 above) -- the new 15 MW wave level at 2x
           battery (an EXISTING battery size, so only wave capacity is new).
        2. Test R12 manually (SCENARIOS[4]) -- the new 4x battery level at
           0 MW wave (an EXISTING wave level, so only battery capacity is
           new).
        Both must: solve with status="ok"/condition="optimal"; pass
        validate_scenario(); maintain hourly energy balance; respect
        generator/storage limits; retain the correct fixed capacities;
        show zero unexplained baseline differences. Only after BOTH pass
        should the remaining seven scenarios be run.
    """
    if "--test-r08" in sys.argv:
        print("*** TEST MODE: running R08 only ***\n")
        run_batch(scenarios=[SCENARIOS[0]], output_dir=OUTPUT_DIR, summary_file=TEST_SUMMARY_FILE)
    else:
        run_batch(SCENARIOS, OUTPUT_DIR, SUMMARY_FILE)


if __name__ == "__main__":
    main()