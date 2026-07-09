"""
run_six_core_scenarios_harmonised.py

Reruns the original six core scenarios with the SAME expanded metric set,
column names, and calculation methods as the final validated
run_nine_scaling_scenarios.py -- producing a harmonised six-scenario
summary that is column-compatible with (and directly appendable to) the
nine-scenario summary, for a complete 15-scenario analysis.

Scenario definitions, all modelling assumptions (base network file, wave
capacities, battery multipliers, demand, renewable profiles, gas capacity,
Slack/mainland-import treatment, storage duration/cost/efficiency, storage
boundary conditions, interconnector rating, marginal-cost ordering, solver,
objective-constant setting, validation tolerance) are UNCHANGED from the
original validated run_six_core_scenarios.py. The 7 MW gas cap / 20 MW
import cap used in the separate import-cap sensitivity experiment are NOT
introduced here.

What changes vs. the original run_six_core_scenarios.py: ONLY metric
extraction, output fields, explicit fixed-capacity safeguards, output-file
handling, and summary compatibility with the nine-scenario file.

Outputs go to a NEW directory -- the original validated six-scenario
results in DATA/outputs/wave_storage_core_v2/ are NEVER touched:
    DATA/outputs/wave_storage_core_harmonised_v1/
        R01_wave0MW_batt2x.nc ... R07_wave7MW_batt3x.nc
        six_core_scenarios_harmonised_summary.csv
        six_core_scenarios_harmonisation_audit.csv

This script does NOT combine the six and nine summaries -- it only
confirms column-compatibility (see check_column_compatibility()).
"""

import os
import pandas as pd
import numpy as np
import pypsa

# ── Constants (UNCHANGED modelling assumptions from the original six-scenario study) ─
BASE_NETWORK_FILE = "DATA/outputs/network_wave5MW_tidal7.2MW.nc"

# NEW output location -- does NOT touch the original validated six-scenario results
OUTPUT_DIR = "DATA/outputs/wave_storage_core_harmonised_v1"
SUMMARY_FILE = f"{OUTPUT_DIR}/six_core_scenarios_harmonised_summary.csv"
AUDIT_FILE = f"{OUTPUT_DIR}/six_core_scenarios_harmonisation_audit.csv"

# Reference files used ONLY for the audit / compatibility checks below
ORIGINAL_SUMMARY_FILE = "DATA/outputs/wave_storage_core_v2/six_core_wave_storage_scenarios_summary.csv"
NINE_SUMMARY_FILE = "DATA/outputs/wave_storage_scaling_v1/nine_scaling_scenarios_summary.csv"

INTERCONNECTOR_LINES = [
    "19330 - 86018", "86018 - 86170",
    "19330 - 86019", "86019 - 86171",
]
INTERCONNECTOR_S_NOM = 21.05  # MVA per cable -- UNCHANGED

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

# ── The original six scenario definitions -- UNCHANGED ──────────────────────
SCENARIOS = [
    {"id": "R01", "wave_mw": 0, "battery_mult": 2},
    {"id": "R02", "wave_mw": 5, "battery_mult": 2},
    {"id": "R06", "wave_mw": 7, "battery_mult": 2},
    {"id": "R03", "wave_mw": 0, "battery_mult": 3},
    {"id": "R04", "wave_mw": 5, "battery_mult": 3},
    {"id": "R07", "wave_mw": 7, "battery_mult": 3},
]

# ── Required final column order (must match the nine-scenario summary exactly) ─
FINAL_COLUMN_ORDER = [
    "id", "wave_mw", "battery_mult", "status", "condition",
    "optimisation_validation_pass", "critical_hour_pf_pass", "pf_slack_review_needed",
    "objective",
    "mainland_import_mwh", "mainland_export_mwh", "slack_active_hours", "slack_max_mw",
    "gas_generation_mwh", "gas_operating_hours", "gas_peak_dispatch_mw", "combined_gas_import_mwh",
    "wind_available_mwh", "wind_dispatch_mwh", "wind_curtail_mwh", "wind_curtail_pct",
    "wave_available_mwh", "wave_dispatch_mwh", "wave_curtail_mwh", "wave_curtail_pct",
    "tidal_available_mwh", "tidal_dispatch_mwh", "tidal_curtail_mwh", "tidal_curtail_pct",
    "total_renewable_curtail_mwh", "interconnector_max_loading_pct",
    "battery_power_capacity_mw", "battery_energy_capacity_mwh",
    "battery_charge_mwh", "battery_discharge_mwh", "battery_throughput_mwh",
    "battery_equivalent_full_cycles", "battery_charging_hours", "battery_discharging_hours",
    "battery_peak_charge_mw", "battery_peak_discharge_mw",
    "battery_mean_soc_mwh", "battery_max_soc_mwh", "battery_mean_soc_pct",
    "battery_hours_below_10pct", "battery_hours_above_80pct",
    "output_file",
]

CONSOLE_SUMMARY_COLUMNS = [
    "id", "wave_mw", "battery_mult", "status", "condition",
    "optimisation_validation_pass", "objective",
    "battery_power_capacity_mw", "battery_energy_capacity_mwh",
    "battery_charge_mwh", "battery_discharge_mwh", "battery_throughput_mwh",
    "battery_equivalent_full_cycles", "battery_charging_hours",
    "battery_discharging_hours", "battery_mean_soc_pct",
    "mainland_import_mwh", "mainland_export_mwh", "gas_generation_mwh", "combined_gas_import_mwh",
    "wind_curtail_mwh", "wave_curtail_mwh", "tidal_curtail_mwh", "total_renewable_curtail_mwh",
    "interconnector_max_loading_pct",
]

# Audited metrics: name in the ORIGINAL six-scenario summary -> name in this
# harmonised summary (identical in every case, listed explicitly for clarity)
AUDITED_METRICS = [
    "objective", "mainland_import_mwh", "mainland_export_mwh",
    "battery_charge_mwh", "battery_discharge_mwh", "battery_throughput_mwh",
    "battery_equivalent_full_cycles", "battery_charging_hours", "battery_discharging_hours",
    "battery_mean_soc_pct", "wind_curtail_mwh", "wave_curtail_mwh", "tidal_curtail_mwh",
    "interconnector_max_loading_pct",
]


def build_output_filename(scenario, output_dir=OUTPUT_DIR):
    wave_tag = f"wave{scenario['wave_mw']}MW"
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
            f"match the expected value ({expected_mw} MW). Investigate before trusting scaled results."
        )
    return baseline_combined_storage


def run_scenario(scenario, output_dir=OUTPUT_DIR):
    # ── Overwrite safeguard: refuse to silently clobber an existing file ────
    out_path = build_output_filename(scenario, output_dir=output_dir)
    if os.path.exists(out_path):
        raise FileExistsError(
            f"Scenario output already exists: {out_path}. "
            "Use a new output directory or remove the existing file deliberately."
        )

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

    # Original storage-scaling convention -- UNCHANGED: main unit scaled from
    # the original combined baseline capacity; secondary retained separately
    # at its epsilon (NOT subtracted from the main unit).
    mult = scenario["battery_mult"]
    network.storage_units.loc[STORAGE_UNIT, "p_nom"] = baseline_combined_storage * mult

    network.storage_units.loc[STORAGE_UNIT, "p_nom_extendable"] = False
    network.storage_units.loc[STORAGE_UNIT_SECONDARY, "p_nom_extendable"] = False

    missing_lines = [ln for ln in INTERCONNECTOR_LINES if ln not in network.lines.index]
    if missing_lines:
        raise KeyError(f"Interconnector line(s) not found in network.lines: {missing_lines}.")
    network.lines.loc[INTERCONNECTOR_LINES, "s_nom"] = INTERCONNECTOR_S_NOM
    network.lines.loc[INTERCONNECTOR_LINES, "s_nom_extendable"] = False

    clear_stale_storage_data(network)
    network.storage_units.loc[STORAGE_UNIT, "state_of_charge_initial"] = 0.0

    status, condition = network.optimize(
        solver_name="highs", include_objective_constant=INCLUDE_OBJECTIVE_CONSTANT
    )
    solve_ok = (str(status) == "ok") and (str(condition) == "optimal")

    result = {col: None for col in FINAL_COLUMN_ORDER}
    result.update({
        "id": scenario["id"], "wave_mw": scenario["wave_mw"], "battery_mult": scenario["battery_mult"],
        "status": status, "condition": condition,
        "optimisation_validation_pass": None,
        "critical_hour_pf_pass": None,      # pending -- filled in by a SEPARATE post-batch PF pass
        "pf_slack_review_needed": None,     # pending -- filled in by a SEPARATE post-batch PF pass
    })

    if not solve_ok:
        print(f"  [{scenario['id']}] Solve did not reach optimal: status={status}, condition={condition}")
        return result, network

    network.meta.update({
        "solver_status": str(status), "termination_condition": str(condition),
        "scenario_id": scenario["id"], "wave_mw": float(scenario["wave_mw"]),
        "battery_mult": float(scenario["battery_mult"]),
        "main_storage_cyclic": STORAGE_CYCLIC, "main_storage_initial_soc": 0.0,
        "wave_marginal_cost": WAVE_MARGINAL_COST, "tidal_marginal_cost": TIDAL_MARGINAL_COST,
        "wind_marginal_cost": WIND_MARGINAL_COST, "include_objective_constant": INCLUDE_OBJECTIVE_CONSTANT,
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
        print(f"  [{scenario['id']}] *** OPTIMISATION VALIDATION FAILED -- see [FAIL] lines above. ***")

    # ── Snapshot weights, aligned exactly as in the nine-scenario runner ────
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

    # ── Slack (mainland) metrics ──────────────────────────────────────────────
    slack_col = [gen for gen in network.generators.index if "Slack" in gen]
    slack_p = network.generators_t.p[slack_col].sum(axis=1)
    result["mainland_import_mwh"] = weighted_energy(slack_p.clip(lower=0.0))
    result["mainland_export_mwh"] = weighted_energy(-slack_p.clip(upper=0.0))
    result["slack_active_hours"] = float((slack_p.abs() > 1e-4).mul(generator_weights).sum())
    result["slack_max_mw"] = float(slack_p.abs().max())

    # ── Gas-turbine metrics -- calculated directly from solved dispatch ──────
    gas_p = network.generators_t.p[GAS_GEN].reindex(network.snapshots).fillna(0.0).astype(float)
    result["gas_generation_mwh"] = weighted_energy(gas_p)
    result["gas_operating_hours"] = float((gas_p > 1e-4).mul(generator_weights).sum())
    result["gas_peak_dispatch_mw"] = float(gas_p.max())
    result["combined_gas_import_mwh"] = result["gas_generation_mwh"] + result["mainland_import_mwh"]

    # ── Full renewable availability / dispatch / curtailment ─────────────────
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

    # ── Interconnector loading (effective capacity, max of |p0|/|p1|) ─────────
    line_capacity = network.lines["s_nom"].astype(float).copy()
    if "s_nom_extendable" in network.lines.columns:
        extendable = network.lines["s_nom_extendable"].fillna(False).astype(bool)
        if extendable.any():
            if "s_nom_opt" in network.lines.columns:
                optimised_capacity = network.lines["s_nom_opt"].reindex(network.lines.index)
                valid_optimised_capacity = (
                    optimised_capacity.notna() & np.isfinite(optimised_capacity) & (optimised_capacity > 0)
                )
                use_optimised_capacity = extendable & valid_optimised_capacity
                line_capacity.loc[use_optimised_capacity] = optimised_capacity.loc[use_optimised_capacity]
                fallback_lines = extendable & ~valid_optimised_capacity
                if fallback_lines.any():
                    print(f"  [WARNING] Extendable line(s) without a valid s_nom_opt; falling back to s_nom: "
                          f"{fallback_lines[fallback_lines].index.tolist()}")

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
        raise ValueError(f"Invalid effective interconnector capacity for: {bad_lines[bad_lines].index.tolist()}")

    interconnector_flow = pd.DataFrame(
        np.maximum(network.lines_t.p0[INTERCONNECTOR_LINES].abs(), network.lines_t.p1[INTERCONNECTOR_LINES].abs()),
        index=network.snapshots, columns=INTERCONNECTOR_LINES,
    )
    result["interconnector_max_loading_pct"] = float(
        (interconnector_flow.div(interconnector_limit.replace(0, np.nan)) * 100).max().max()
    )

    # ── Battery metrics -- p_store/p_dispatch used DIRECTLY, not clipped net p ─
    battery_charge = (
        network.storage_units_t.p_store[STORAGE_UNIT].reindex(network.snapshots).fillna(0.0).astype(float)
    )
    battery_discharge = (
        network.storage_units_t.p_dispatch[STORAGE_UNIT].reindex(network.snapshots).fillna(0.0).astype(float)
    )
    battery_net_power = (
        network.storage_units_t.p[STORAGE_UNIT].reindex(network.snapshots).fillna(0.0).astype(float)
    )
    battery_soc = network.storage_units_t.state_of_charge[STORAGE_UNIT].reindex(network.snapshots).fillna(0.0).astype(float)

    battery_power_capacity = float(network.storage_units.loc[STORAGE_UNIT, "p_nom"])
    battery_energy_capacity = battery_power_capacity * float(network.storage_units.loc[STORAGE_UNIT, "max_hours"])

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
            weighted_mean_soc / battery_energy_capacity * 100 if battery_energy_capacity > 0 else np.nan
        ),
        "battery_hours_below_10pct": float((battery_soc < 0.10 * battery_energy_capacity).mul(storage_weights).sum()),
        "battery_hours_above_80pct": float((battery_soc > 0.80 * battery_energy_capacity).mul(storage_weights).sum()),
    })

    os.makedirs(output_dir, exist_ok=True)
    network.export_to_netcdf(out_path)
    result["output_file"] = out_path

    return result, network


def build_audit(final_df, original_summary_file=ORIGINAL_SUMMARY_FILE, audit_file=AUDIT_FILE, tolerance=1e-6):
    """
    Compares overlapping metrics between the original validated six-scenario
    summary and this harmonised rerun -- honestly reports any differences,
    does not force agreement.
    """
    if not os.path.exists(original_summary_file):
        print(f"  [AUDIT] Original summary not found at {original_summary_file} -- skipping audit.")
        return None

    original_df = pd.read_csv(original_summary_file).set_index("id")
    harmonised_df = final_df.set_index("id")

    audit_rows = []
    for scenario_id in harmonised_df.index:
        if scenario_id not in original_df.index:
            continue
        for metric in AUDITED_METRICS:
            if metric not in original_df.columns or metric not in harmonised_df.columns:
                continue
            original_value = original_df.loc[scenario_id, metric]
            harmonised_value = harmonised_df.loc[scenario_id, metric]
            try:
                abs_diff = float(harmonised_value) - float(original_value)
                rel_diff_pct = (abs_diff / float(original_value) * 100) if float(original_value) != 0 else np.nan
                matches = abs(abs_diff) <= tolerance
            except (TypeError, ValueError):
                abs_diff, rel_diff_pct, matches = np.nan, np.nan, False

            audit_rows.append({
                "id": scenario_id, "metric": metric,
                "original_value": original_value, "harmonised_value": harmonised_value,
                "absolute_difference": abs_diff, "relative_difference_pct": rel_diff_pct,
                "matches_within_tolerance": matches,
            })

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(audit_file, index=False)

    n_mismatches = int((~audit_df["matches_within_tolerance"]).sum())
    print(f"\n  [AUDIT] Saved: {audit_file}")
    print(f"  [AUDIT] {len(audit_df) - n_mismatches} / {len(audit_df)} metric comparisons match within "
          f"tolerance ({tolerance}).")
    if n_mismatches > 0:
        print(f"  [AUDIT] *** {n_mismatches} comparison(s) did NOT match -- see audit file for details. ***")
        print(audit_df[~audit_df["matches_within_tolerance"]].to_string(index=False))

    return audit_df


def check_column_compatibility(final_df, nine_summary_file=NINE_SUMMARY_FILE):
    """
    Confirms (does NOT combine) that this harmonised six-scenario summary's
    columns exactly match the nine-scenario summary's columns, in the same
    order.
    """
    if not os.path.exists(nine_summary_file):
        print(f"  [COMPATIBILITY] Nine-scenario summary not found at {nine_summary_file} -- skipping check.")
        return

    nine_columns = list(pd.read_csv(nine_summary_file, nrows=0).columns)
    six_columns = list(final_df.columns)

    if six_columns != nine_columns:
        missing = [col for col in nine_columns if col not in six_columns]
        unexpected = [col for col in six_columns if col not in nine_columns]
        raise ValueError(
            "The harmonised six-scenario summary is not column-compatible with the nine-scenario summary. "
            f"Missing={missing}, unexpected={unexpected}, order_matches={six_columns == nine_columns}"
        )

    print(f"  [COMPATIBILITY] Six-scenario summary columns match the nine-scenario summary exactly "
          f"({len(six_columns)} columns, same order).")


def run_batch(scenarios=SCENARIOS, output_dir=OUTPUT_DIR, summary_file=SUMMARY_FILE):
    os.makedirs(output_dir, exist_ok=True)

    # ── Overwrite safeguard: refuse to silently clobber an existing summary ──
    if os.path.exists(summary_file):
        raise FileExistsError(
            f"The harmonised summary already exists: {summary_file}. "
            "Use a new versioned output directory or delete the existing harmonised file deliberately."
        )

    all_results = []
    total_scenarios = len(scenarios)

    print("=" * 100)
    print(f"RUNNING {total_scenarios} HARMONISED CORE SCENARIOS (matching nine-scenario column structure)")
    print("=" * 100)

    for i, scenario in enumerate(scenarios, start=1):
        print(f"\n[{i}/{total_scenarios}] {scenario['id']}: wave={scenario['wave_mw']}MW, "
              f"battery_mult={scenario['battery_mult']}")

        result, _ = run_scenario(scenario, output_dir=output_dir)
        all_results.append(result)

        running_df = pd.DataFrame(all_results).reindex(columns=FINAL_COLUMN_ORDER)
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
            print(f"  -> Scenario did not solve optimally: status={result['status']}, condition={result['condition']}")

        print("\n  --- Running summary so far ---")
        print(running_df[CONSOLE_SUMMARY_COLUMNS].to_string(index=False))

    final_df = pd.DataFrame(all_results).reindex(columns=FINAL_COLUMN_ORDER)
    final_df.to_csv(summary_file, index=False)

    print("\n" + "=" * 100)
    print(f"ALL {total_scenarios} SCENARIOS COMPLETE")
    print("=" * 100)
    print(final_df[CONSOLE_SUMMARY_COLUMNS].to_string(index=False))
    print(f"\nSaved FULL harmonised summary (all fields): {summary_file}")

    build_audit(final_df)
    check_column_compatibility(final_df)

    print("\nNOTE: critical_hour_pf_pass / pf_slack_review_needed remain pending -- populate separately via")
    print("the critical-hour PF validation process if a fully cross-checked summary is needed.")

    return final_df


def main():
    run_batch(SCENARIOS, OUTPUT_DIR, SUMMARY_FILE)


if __name__ == "__main__":
    main()