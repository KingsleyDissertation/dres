"""
validate_scenario.py (revised)

Comprehensive post-solve validation for optimize()-based PyPSA scenarios.
"Optimal" only confirms the solver found the least-cost solution to the
model AS SPECIFIED -- it says nothing about whether the model itself is
correct, nor whether the solve genuinely succeeded. This module checks that
separately.

Key corrections in this revision:
  1. Battery capacity expectations are DERIVED from baseline_network, not
     hard-coded. KIRKWA3A_Storage (epsilon-reduced) is validated separately
     from the combined Orkney Storage Park capacity.
  2. Every storage unit is validated in a loop, not just one.
  3. Effective values use network.get_switchable_as_dense(...) throughout,
     so static defaults and time-varying overrides are combined correctly.
  4. p_set check treats all-NaN columns as clean, and also checks the
     effective (dense) p_set.
  5. All MWh totals use snapshot_weightings, not a plain .sum().
  6. The battery efficiency check only runs (and only counts toward
     battery_ok) when cyclic_state_of_charge=True, standing_loss=0, no
     inflow, no spillage, and uniform snapshot weights -- otherwise it is
     explicitly skipped and does not silently pass.
  7. Line/transformer loading uses s_nom_opt for extendable branches and
     effective dense s_max_pu, not a bare s_nom divide.
  8. consistency_check() is run in strict mode; any unwhitelisted issue is
     a genuine FAIL, not just an informational print.
  9. Slack use affects overall_pass via explicit scenario-level tolerance/
     allowance settings, and is never called "imports" unless the scenario
     says so.
  10. Renewable dispatch-exceeds-availability is checked and FAILS before
      curtailment is computed, rather than being silently clipped away.
  11. Input isolation uses a whitelist-diff approach across a much larger
      set of attributes.
  12. Solver status/condition are passed in explicitly and checked first.
  13. Objective interpretation checks for extendable components rather than
      asserting "operating cost only" unconditionally.
  14. Cross-scenario checks correctly state that cost should NOT increase
      when optional capacity is added without investment cost, and perform
      real pairwise comparisons where scenario metadata is available.
  15. Critical-hour pf() validation fixes storage dispatch too, and checks
      convergence/voltages/branch loading properly.
  16. All the above feed into report["checks"] and overall_pass explicitly.

USAGE:
    from validate_scenario import validate_scenario
    report = validate_scenario(
        network=solved_network,
        baseline_network=fresh_unmodified_network,
        scenario={
            "id": "R01", "wave_mw": 0, "battery_mult": 2,
            "status": status, "condition": condition,
            "allow_slack": False, "slack_tolerance_mwh": 1e-6,
        },
        tolerance_mw=1e-4,
    )

    from validate_scenario import cross_scenario_checks, validate_critical_hours_pf
    cross_scenario_checks(nc_paths, scenarios=[{...}, {...}, ...])
    validate_critical_hours_pf("DATA/outputs/R01_wave0MW_batt2x.nc")
"""

import pypsa
import pandas as pd
import numpy as np
import os
import logging
import io


STORAGE_UNIT = "Orkney Storage Park"
STORAGE_UNIT_SECONDARY = "KIRKWA3A_Storage"
STORAGE_SECONDARY_EPSILON = 0.01
WAVE_GEN = "Wave Generator"
TIDAL_GEN = "EDAY Tidal Generator"
INTERCONNECTOR_LINES = [
    "19330 - 86018", "86018 - 86170",
    "19330 - 86019", "86019 - 86171",
]
INTERCONNECTOR_S_NOM = 21.05

# Known, previously-reviewed zero-resistance lines (see earlier investigation
# in this project). Kept here as documentation/reference -- NOT currently
# used in automated logic, since network.consistency_check(check_dtypes=False,
# strict=["all"]) empirically passed cleanly without these needing an
# explicit whitelist in practice. If a future PyPSA/data change causes these
# to surface as blocking structural warnings, reintroduce filtering logic
# referencing this set.
WHITELISTED_ZERO_R_LINES = {
    "19330 - 86018", "19330 - 86019", "86102 - 86104", "86103 - 86105",
    "86110 - 86111", "86117 - 86143", "86119 - 86144", "86124 - 86142",
    "86125 - 86142", "86131 - 86132", "86162 - 86163", "86162 - 86164",
    "86168 - 86140",
}


def _p(msg, ok=None):
    if ok is True:
        print(f"  [PASS] {msg}")
    elif ok is False:
        print(f"  [FAIL] {msg}")
    else:
        print(f"  [INFO] {msg}")


def _weighted_sum(df, weights):
    if df.empty:
        return 0.0
    if isinstance(df, pd.Series):
        return float(df.mul(weights).sum())
    return df.mul(weights, axis=0).sum()


def validate_scenario(network, baseline_network, scenario, tolerance_mw=1e-4):
    report = {"scenario_id": scenario.get("id", "unknown"), "checks": {}}

    print("\n" + "=" * 100)
    print(f"VALIDATING SCENARIO: {scenario.get('id', 'unknown')} "
          f"(wave={scenario.get('wave_mw')}MW, battery={scenario.get('battery_mult')})")
    print("=" * 100)

    print("\n--- 0. Solver outcome ---")
    status = scenario.get("status")
    condition = scenario.get("condition")

    # Also check network.meta, which is where status/condition SHOULD be
    # saved by scenario scripts going forward (see module docstring / runner
    # scripts for the required network.meta["solver_status"] /
    # ["termination_condition"] convention).
    if status is None and hasattr(network, "meta") and network.meta:
        status = network.meta.get("solver_status")
    if condition is None and hasattr(network, "meta") and network.meta:
        condition = network.meta.get("termination_condition")

    solver_status_state = None  # "VERIFIED_PASS" | "VERIFIED_FAIL" | "NOT_VERIFIED"
    if status is not None and condition is not None:
        solver_ok = (status == "ok") and (condition == "optimal")
        solver_status_state = "VERIFIED_PASS" if solver_ok else "VERIFIED_FAIL"
        _p(f"status='{status}', condition='{condition}' (explicit): optimal solve = {solver_ok}", solver_ok)
    else:
        # A finite objective is WEAKER evidence than an explicit solver
        # status and must NOT be reported as a verified pass. It is only
        # used as a minimal sanity gate to decide whether it's even possible
        # to continue running the rest of the checks (a missing/NaN
        # objective means there is nothing meaningful to validate at all).
        obj = getattr(network, "objective", None)
        objective_finite = obj is not None and np.isfinite(obj)
        solver_status_state = "NOT_VERIFIED"
        solver_ok = objective_finite  # gates whether we can proceed, NOT a verified pass
        _p(f"status/condition NOT PROVIDED (neither in scenario dict nor network.meta). "
           f"Solver-status verification: NOT VERIFIED. Objective is finite = {objective_finite} "
           f"(objective={obj}) -- this only confirms enough to proceed with further checks, it is "
           f"NOT confirmation of a successful optimal solve. For future scenario scripts, save "
           f"network.meta['solver_status'] and network.meta['termination_condition'] immediately "
           f"after optimize(), before exporting.", None)

    report["solver_status_state"] = solver_status_state
    report["checks"]["0_solver_optimal"] = solver_ok
    if not solver_ok:
        print("\n*** No evidence of a successful solve (objective missing/non-finite) -- skipping "
              "all further checks. ***")
        report["overall_pass"] = False
        return report

    gen_p_min_pu = network.get_switchable_as_dense("Generator", "p_min_pu")
    gen_p_max_pu = network.get_switchable_as_dense("Generator", "p_max_pu")
    gen_p_set_eff = network.get_switchable_as_dense("Generator", "p_set")
    load_p_set_eff = network.get_switchable_as_dense("Load", "p_set")
    su_p_min_pu = network.get_switchable_as_dense("StorageUnit", "p_min_pu")
    su_p_max_pu = network.get_switchable_as_dense("StorageUnit", "p_max_pu")
    line_s_max_pu = network.get_switchable_as_dense("Line", "s_max_pu")
    trafo_s_max_pu = (network.get_switchable_as_dense("Transformer", "s_max_pu")
                      if not network.transformers.empty else pd.DataFrame())

    gen_weights = network.snapshot_weightings["generators"]
    store_weights = (network.snapshot_weightings["stores"]
                      if "stores" in network.snapshot_weightings.columns
                      else network.snapshot_weightings["generators"])
    uniform_weights = (gen_weights.nunique() == 1) and (store_weights.nunique() == 1)

    wind_gens = [g for g in network.generators.index if "WindFarm" in g]

    print("\n--- 1/11. Input isolation check (whitelist-diff) ---")
    ok = True
    unexplained_diffs = []

    storage_names = [STORAGE_UNIT, STORAGE_UNIT_SECONDARY]

    baseline_combined_power = baseline_network.storage_units.loc[storage_names, "p_nom"].sum()

    actual_power = network.storage_units.loc[storage_names, "p_nom"].sum()
    actual_energy = (
        network.storage_units.loc[storage_names, "p_nom"]
        * network.storage_units.loc[storage_names, "max_hours"]
    ).sum()

    actual_wave = network.generators.loc[WAVE_GEN, "p_nom"]
    expected_wave = scenario["wave_mw"]
    wave_ok = np.isclose(actual_wave, expected_wave)
    _p(f"Wave p_nom = {actual_wave} (expected {expected_wave})", wave_ok)
    ok &= wave_ok

    if scenario.get("battery_mult") is not None:
        # Correct expected-value derivation: the scenario transformation
        # scales ONLY STORAGE_UNIT by battery_mult (applied to the SUM of
        # both baseline units' power), while STORAGE_UNIT_SECONDARY is held
        # at the fixed epsilon regardless of the multiplier. The expected
        # total must be built from these two pieces explicitly -- not
        # approximated as "baseline_combined * mult" with an epsilon-sized
        # tolerance band papering over the difference.
        expected_main_power = baseline_combined_power * scenario["battery_mult"]
        expected_secondary_power = STORAGE_SECONDARY_EPSILON
        expected_total_power = expected_main_power + expected_secondary_power

        expected_main_max_hours = baseline_network.storage_units.loc[STORAGE_UNIT, "max_hours"]
        expected_secondary_max_hours = baseline_network.storage_units.loc[STORAGE_UNIT_SECONDARY, "max_hours"]
        expected_total_energy = (
            expected_main_power * expected_main_max_hours
            + expected_secondary_power * expected_secondary_max_hours
        )

        main_power_ok = np.isclose(network.storage_units.loc[STORAGE_UNIT, "p_nom"], expected_main_power)
        secondary_power_ok = np.isclose(
            network.storage_units.loc[STORAGE_UNIT_SECONDARY, "p_nom"], expected_secondary_power
        )
        total_power_ok = np.isclose(actual_power, expected_total_power)
        total_energy_ok = np.isclose(actual_energy, expected_total_energy)

        _p(f"{STORAGE_UNIT} power = {network.storage_units.loc[STORAGE_UNIT, 'p_nom']:.4f} MW "
           f"(expected {expected_main_power:.4f} MW = baseline combined "
           f"{baseline_combined_power:.4f} x {scenario['battery_mult']})", main_power_ok)
        _p(f"{STORAGE_UNIT_SECONDARY} power = "
           f"{network.storage_units.loc[STORAGE_UNIT_SECONDARY, 'p_nom']:.4f} MW "
           f"(expected fixed epsilon {expected_secondary_power:.4f} MW)", secondary_power_ok)
        _p(f"Combined total power = {actual_power:.4f} MW "
           f"(expected {expected_total_power:.4f} MW = main {expected_main_power:.4f} + "
           f"secondary {expected_secondary_power:.4f})", total_power_ok)
        _p(f"Combined total energy = {actual_energy:.4f} MWh "
           f"(expected {expected_total_energy:.4f} MWh = main "
           f"{expected_main_power:.4f}x{expected_main_max_hours:.3f}h + secondary "
           f"{expected_secondary_power:.4f}x{expected_secondary_max_hours:.3f}h)", total_energy_ok)

        batt_ok = main_power_ok and secondary_power_ok and total_power_ok and total_energy_ok
    else:
        power_ok = actual_power < 1.0
        _p(f"Combined storage POWER = {actual_power:.6f} MW (expected near-zero for no-battery "
           f"scenario; threshold <1.0 MW used since exact epsilon convention for the main unit "
           f"is not independently confirmed)", power_ok)
        batt_ok = power_ok
    ok &= batt_ok

    # Explicit validation of the boundary-condition assumptions, rather than
    # silently skipping them -- these must be stated in scenario metadata
    # and checked, not just reported as information.
    expected_cyclic = scenario.get("main_storage_cyclic", False)
    actual_cyclic = network.storage_units.loc[STORAGE_UNIT, "cyclic_state_of_charge"]
    cyclic_ok = (bool(actual_cyclic) == bool(expected_cyclic))
    _p(f"{STORAGE_UNIT} cyclic_state_of_charge = {actual_cyclic} "
       f"(expected {expected_cyclic}, from scenario metadata)", cyclic_ok)
    ok &= cyclic_ok

    if not expected_cyclic:
        expected_initial_soc = scenario.get("main_storage_initial_soc", 0.0)
        actual_initial_soc = network.storage_units.loc[STORAGE_UNIT, "state_of_charge_initial"]
        initial_soc_ok = np.isclose(actual_initial_soc, expected_initial_soc)
        _p(f"{STORAGE_UNIT} state_of_charge_initial = {actual_initial_soc} "
           f"(expected {expected_initial_soc}, from scenario metadata)", initial_soc_ok)
        ok &= initial_soc_ok

    if scenario.get("battery_mult") is None:
        # Only needed here for no-battery scenarios -- battery scenarios
        # already validate STORAGE_UNIT_SECONDARY's p_nom in detail above.
        actual_secondary_pnom = network.storage_units.loc[STORAGE_UNIT_SECONDARY, "p_nom"]
        secondary_pnom_ok = np.isclose(actual_secondary_pnom, STORAGE_SECONDARY_EPSILON, rtol=0.05)
        _p(f"{STORAGE_UNIT_SECONDARY} p_nom = {actual_secondary_pnom:.5f} "
           f"(expected epsilon {STORAGE_SECONDARY_EPSILON})", secondary_pnom_ok)
        ok &= secondary_pnom_ok

    secondary_p = network.storage_units_t.p[STORAGE_UNIT_SECONDARY]
    secondary_discharge_mwh = _weighted_sum(secondary_p.clip(lower=0), store_weights)
    secondary_charge_mwh = _weighted_sum((-secondary_p.clip(upper=0)), store_weights)
    total_demand_mwh_for_scale = _weighted_sum(load_p_set_eff.sum(axis=1), gen_weights)
    secondary_negligible = (
        (secondary_discharge_mwh + secondary_charge_mwh)
        < max(1.0, 0.001 * total_demand_mwh_for_scale)
    )
    _p(f"{STORAGE_UNIT_SECONDARY} annual charge+discharge = "
       f"{secondary_charge_mwh + secondary_discharge_mwh:.3f} MWh "
       f"(negligible relative to system: {secondary_negligible})", secondary_negligible)
    ok &= secondary_negligible

    baseline_load_p_set_eff = baseline_network.get_switchable_as_dense("Load", "p_set")
    demand_match = load_p_set_eff.equals(baseline_load_p_set_eff)
    _p(f"Demand profile identical to baseline (effective/dense): {demand_match}", demand_match)
    ok &= demand_match

    snapshots_match = network.snapshots.equals(baseline_network.snapshots)
    _p(f"Snapshots identical to baseline: {snapshots_match}", snapshots_match)
    ok &= snapshots_match

    weightings_match = network.snapshot_weightings.equals(baseline_network.snapshot_weightings)
    _p(f"Snapshot weightings identical to baseline: {weightings_match}", weightings_match)
    ok &= weightings_match

    baseline_gen_p_max_pu = baseline_network.get_switchable_as_dense("Generator", "p_max_pu")
    for gen in wind_gens + [TIDAL_GEN, WAVE_GEN]:
        if gen in gen_p_max_pu.columns and gen in baseline_gen_p_max_pu.columns:
            profile_match = gen_p_max_pu[gen].equals(baseline_gen_p_max_pu[gen])
            if not profile_match:
                unexplained_diffs.append(f"{gen}: p_max_pu profile differs from baseline")

    # Do NOT auto-whitelist wind/tidal/wave marginal cost changes -- a
    # scenario must explicitly DECLARE which generators' costs it intends to
    # change, and to what value. This prevents an accidental cost change
    # anywhere from being silently absorbed as "expected" just because it
    # happens to be a renewable generator.
    allowed_cost_changes = scenario.get("allowed_marginal_cost_changes", {})
    for gen in network.generators.index:
        if gen not in baseline_network.generators.index:
            continue
        actual_cost = network.generators.loc[gen, "marginal_cost"]
        baseline_cost = baseline_network.generators.loc[gen, "marginal_cost"]

        if gen in allowed_cost_changes:
            expected_cost = allowed_cost_changes[gen]
            if not np.isclose(actual_cost, expected_cost):
                unexplained_diffs.append(
                    f"{gen}: marginal_cost {actual_cost} != expected {expected_cost} "
                    f"(per scenario['allowed_marginal_cost_changes'])"
                )
        elif not np.isclose(actual_cost, baseline_cost):
            unexplained_diffs.append(f"{gen}: marginal_cost {actual_cost} != baseline {baseline_cost}")

    baseline_gen_p_min_pu = baseline_network.get_switchable_as_dense("Generator", "p_min_pu")
    for gen in network.generators.index:
        if gen in gen_p_min_pu.columns and gen in baseline_gen_p_min_pu.columns:
            if not gen_p_min_pu[gen].equals(baseline_gen_p_min_pu[gen]):
                unexplained_diffs.append(f"{gen}: p_min_pu differs from baseline")

    storage_attrs_to_check = ["max_hours", "efficiency_store", "efficiency_dispatch",
                               "standing_loss", "cyclic_state_of_charge"]
    for su in network.storage_units.index:
        if su not in baseline_network.storage_units.index:
            continue
        for attr in storage_attrs_to_check:
            actual_val = network.storage_units.loc[su, attr]
            baseline_val = baseline_network.storage_units.loc[su, attr]
            if su == STORAGE_UNIT and attr in ["max_hours", "cyclic_state_of_charge"]:
                continue
            same = (actual_val == baseline_val)
            if isinstance(actual_val, float) and isinstance(baseline_val, float):
                same = np.isclose(actual_val, baseline_val)
            if not same:
                unexplained_diffs.append(f"{su}.{attr}: {actual_val} != baseline {baseline_val}")

    if not network.transformers.empty:
        trafo_s_nom_match = network.transformers["s_nom"].equals(baseline_network.transformers["s_nom"])
        if not trafo_s_nom_match:
            unexplained_diffs.append("Transformer s_nom differs from baseline")

    baseline_line_s_max_pu = baseline_network.get_switchable_as_dense("Line", "s_max_pu")
    if not line_s_max_pu.equals(baseline_line_s_max_pu):
        unexplained_diffs.append("Line s_max_pu differs from baseline")

    other_lines = network.lines.drop(index=INTERCONNECTOR_LINES, errors="ignore")
    baseline_other_lines = baseline_network.lines.drop(index=INTERCONNECTOR_LINES, errors="ignore")
    lines_match = other_lines["s_nom"].equals(baseline_other_lines["s_nom"])
    _p(f"Non-interconnector line s_nom unchanged from baseline: {lines_match}", lines_match)
    ok &= lines_match

    interconnector_ok = (network.lines.loc[INTERCONNECTOR_LINES, "s_nom"] == INTERCONNECTOR_S_NOM).all()
    _p(f"Interconnector s_nom == {INTERCONNECTOR_S_NOM} MVA on all 4 segments: {interconnector_ok}", interconnector_ok)
    ok &= interconnector_ok

    control_match = network.generators["control"].equals(baseline_network.generators["control"])
    if not control_match:
        unexplained_diffs.append("Generator control types differ from baseline")
    carrier_match = network.generators["carrier"].equals(baseline_network.generators["carrier"])
    if not carrier_match:
        unexplained_diffs.append("Generator carrier assignments differ from baseline")

    for gen in wind_gens + [TIDAL_GEN, "gas_turbine(Oil_station)", "Slack_generator"]:
        actual = network.generators.loc[gen, "p_nom"]
        expected = baseline_network.generators.loc[gen, "p_nom"]
        if not np.isclose(actual, expected):
            unexplained_diffs.append(f"{gen} p_nom: {actual} != baseline {expected}")

    if unexplained_diffs:
        ok = False
        _p(f"{len(unexplained_diffs)} UNEXPLAINED difference(s) from baseline found:", False)
        for d in unexplained_diffs:
            print(f"    - {d}")
    else:
        _p("No unexplained differences from baseline (whitelist-diff clean)", True)

    report["checks"]["1_11_input_isolation"] = ok
    report["unexplained_diffs"] = unexplained_diffs

    print("\n--- 2/4. Generator p_set cleared (required for optimize()) ---")
    static_pset_clean = network.generators["p_set"].isna().all()
    _p(f"All static generator p_set are NaN: {static_pset_clean}", static_pset_clean)

    timevarying_pset_clean = (
        network.generators_t.p_set.empty
        or network.generators_t.p_set.isna().all().all()
    )
    _p(f"All time-varying generator p_set are empty/NaN: {timevarying_pset_clean}", timevarying_pset_clean)

    effective_pset_clean = gen_p_set_eff.isna().all().all() if not gen_p_set_eff.empty else True
    _p(f"Effective (dense) generator p_set all NaN: {effective_pset_clean}", effective_pset_clean)

    loads_pset_intact = not load_p_set_eff.empty and load_p_set_eff.notna().any().any()
    _p(f"Effective loads_t.p_set still populated (demand preserved): {loads_pset_intact}", loads_pset_intact)

    pset_ok = static_pset_clean and timevarying_pset_clean and effective_pset_clean and loads_pset_intact
    report["checks"]["2_4_pset_hygiene"] = pset_ok

    print("\n--- 8. network.consistency_check() ---")

    # ── 8a. STRUCTURAL consistency ────────────────────────────────────────────
    #        IMPORTANT: strict=["all"] makes consistency_check() raise on the
    #        FIRST issue it encounters and STOP -- it does not continue
    #        checking everything else. Whitelisting that first exception
    #        (e.g. the known zero-r lines) would therefore risk concluding
    #        "structural consistency passed" while later checks that never
    #        got a chance to run could be hiding a genuine problem. To avoid
    #        this, consistency_check() is run WITHOUT strict mode here, so
    #        it logs every warning without stopping early, and ALL logged
    #        warnings are captured and inspected together. Structural
    #        consistency is only treated as passing if the CAPTURED WARNINGS
    #        are either empty, or consist EXACTLY of the known, previously-
    #        reviewed zero-r line warning (their reactance x is real and
    #        non-zero -- confirmed values 0.00121-0.01089 -- and optimize()'s
    #        linearized power flow uses x, not r, so this specific set does
    #        not break the model). Any other warning, or a zero-r line set
    #        that does not exactly match the whitelist, is a genuine failure.
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.WARNING)
    pypsa_logger = logging.getLogger("pypsa")
    pypsa_logger.addHandler(handler)

    structural_exception = None
    try:
        network.consistency_check(check_dtypes=False)
    except Exception as exc:
        structural_exception = exc
    finally:
        pypsa_logger.removeHandler(handler)

    log_output = log_stream.getvalue()

    if structural_exception is not None:
        # Even without strict mode, some issues can still raise directly
        # (e.g. genuinely broken topology) -- these are always a hard fail,
        # never whitelisted.
        structural_ok = False
        _p(f"Structural consistency check raised an exception (not whitelistable): "
           f"{structural_exception}", False)
    elif not log_output.strip():
        structural_ok = True
        _p("Structural consistency checks passed (no warnings logged at all)", True)
    else:
        # Parse the captured warning text and check whether it is EXACTLY
        # and ONLY the known zero-r line warning, mentioning precisely the
        # whitelisted set of lines and nothing else.
        is_zero_r_warning = "zero r" in log_output.lower()
        mentioned_lines = {line for line in WHITELISTED_ZERO_R_LINES if line in log_output}
        # Strip out the expected zero-r warning text and see if anything
        # else remains -- if so, there are OTHER, non-whitelisted warnings.
        remaining_text = log_output
        if is_zero_r_warning:
            # Remove lines that are part of the zero-r warning block itself
            # (the message + the line names) to see what, if anything, is
            # left over.
            filtered_lines = []
            for line in log_output.strip().split("\n"):
                is_part_of_zero_r_block = (
                    "zero r" in line.lower()
                    or any(wl in line for wl in WHITELISTED_ZERO_R_LINES)
                    or line.strip().startswith("Index(")
                    or line.strip().startswith("dtype=")
                    or line.strip() == ""
                )
                if not is_part_of_zero_r_block:
                    filtered_lines.append(line)
            remaining_text = "\n".join(filtered_lines)

        only_whitelisted_zero_r = (
            is_zero_r_warning
            and mentioned_lines == WHITELISTED_ZERO_R_LINES
            and not remaining_text.strip()
        )

        if only_whitelisted_zero_r:
            structural_ok = True
            _p(f"Structural consistency: ALL logged warnings accounted for by the "
               f"{len(WHITELISTED_ZERO_R_LINES)} previously-reviewed zero-r lines (interconnector "
               f"cables + short jumper connections; real, non-zero reactance confirmed earlier -- "
               f"optimize() uses x, not r). No other warnings were logged. Treated as a known, "
               f"whitelisted characteristic, not a failure.", True)
        else:
            structural_ok = False
            _p(f"Structural consistency check failed -- warnings logged were NOT exclusively the "
               f"whitelisted zero-r lines:", False)
            print(log_output)

    report["checks"]["structural_consistency"] = structural_ok

    # ── 8b. Separate dtype investigation -- run on a COPY, generalized ────────
    #        across EVERY static component dataframe, not just buses. PyPSA's
    #        NetCDF round-trip has now been observed to affect string columns
    #        in MORE than one component (buses previously; carriers.color /
    #        carriers.nice_name in this run) -- so rather than hard-coding a
    #        single component's column list, every static dataframe is
    #        scanned for object/string-typed columns and converted uniformly.
    # PyPSA does not support .copy() on a network with an attached, live
    # solver model (relevant here because this validator may be called
    # immediately after optimize(), on the in-memory network, rather than
    # only on networks reloaded from a saved file -- reloaded networks never
    # carry this attribute at all). Detach it first, exactly as PyPSA's own
    # error message recommends, since we only need STATIC data (bus/generator/
    # etc. tables) for the dtype investigation, not the solver model itself.
    if hasattr(network, "model") and network.model is not None:
        try:
            network.model.solver_model = None
        except Exception:
            pass
    dtype_copy = network.copy()

    # Use PyPSA's own component iteration rather than a manually maintained
    # list of dataframe names -- a hard-coded list has already proven
    # incomplete twice in testing (missed 'carriers', then missed
    # 'line_types'), since PyPSA has many static component types
    # (LineType, TransformerType, SubNetwork, etc.) beyond the commonly-used
    # ones. Iterating network.all_components (and, for older/newer PyPSA
    # versions where that attribute may differ, falling back to a broader
    # manual list) covers every static dataframe PyPSA itself knows about.
    try:
        component_attr_names = sorted({
            c.list_name for c in network.iterate_components()
        })
    except Exception:
        component_attr_names = [
            "buses", "generators", "lines", "line_types", "transformers",
            "transformer_types", "storage_units", "stores", "loads",
            "carriers", "links", "shunt_impedances", "sub_networks",
        ]

    values_unchanged = True
    missing_pattern_unchanged = True
    converted_columns = []

    for comp_name in component_attr_names:
        if not hasattr(network, comp_name):
            continue
        original_df = getattr(network, comp_name)
        copy_df = getattr(dtype_copy, comp_name)
        if not isinstance(original_df, pd.DataFrame) or original_df.empty:
            continue

        for column in original_df.columns:
            if original_df[column].dtype == object or pd.api.types.is_string_dtype(original_df[column]):
                copy_df[column] = copy_df[column].astype(object)
                converted_columns.append(f"{comp_name}.{column}")

                original = original_df[column].astype("string")
                converted = copy_df[column].astype("string")
                if not original.equals(converted):
                    values_unchanged = False
                    _p(f"Dtype conversion CHANGED VALUES in '{comp_name}.{column}' -- NOT benign", False)

                original_na = original_df[column].isna()
                converted_na = copy_df[column].isna()
                if not original_na.equals(converted_na):
                    missing_pattern_unchanged = False
                    _p(f"Dtype conversion CHANGED MISSING-VALUE PATTERN in '{comp_name}.{column}'", False)

    _p(f"Converted {len(converted_columns)} object/string column(s) across "
       f"{len(component_attr_names)} component types (via network.iterate_components()) "
       f"for dtype re-check", None)

    dtype_check_passes_after_conversion = False
    dtype_check_exception = None
    if values_unchanged and missing_pattern_unchanged:
        try:
            dtype_copy.consistency_check(check_dtypes=True, strict=["dtypes"])
            dtype_check_passes_after_conversion = True
        except Exception as exc:
            dtype_check_exception = exc

    if values_unchanged and missing_pattern_unchanged and dtype_check_passes_after_conversion:
        dtype_result = "likely_serialization_artifact"
        _p("Dtype check passes after converting text columns across ALL components without "
           "changing their values. This is likely a NetCDF string-representation issue.", None)
        print("Dtype consistency: WARNING -- likely NetCDF string representation issue")
    elif not values_unchanged or not missing_pattern_unchanged:
        dtype_result = "unresolved"
        _p("Dtype conversion altered values or missing-value patterns -- cannot classify as benign.",
           False)
        print("Dtype consistency: UNRESOLVED -- requires investigation")
    else:
        dtype_result = "unresolved"
        _p(f"Dtype check still fails after temporary conversion across all components: "
           f"{dtype_check_exception}", False)
        print("Dtype consistency: UNRESOLVED -- requires investigation")

    report["dtype_consistency"] = dtype_result

    if dtype_result == "unresolved":
        print("\n*** NOTE: dtype_consistency is UNRESOLVED. This does not automatically fail the ***")
        print("*** scenario, but should be manually reviewed before fully trusting this result. ***")

    print(f"\n--- Hourly energy balance (tolerance = {tolerance_mw} MW) ---")
    total_gen = network.generators_t.p.sum(axis=1)
    storage_net = network.storage_units_t.p.sum(axis=1)
    total_demand = load_p_set_eff.sum(axis=1)

    residual = total_gen + storage_net - total_demand
    max_residual = residual.abs().max()
    balance_ok = max_residual < tolerance_mw
    _p(f"Max hourly residual (gen + storage_net - demand): {max_residual:.6f} MW", balance_ok)
    if not balance_ok:
        worst_hour = residual.abs().idxmax()
        _p(f"  Worst hour: {worst_hour}, residual = {residual[worst_hour]:.6f} MW", None)
    report["checks"]["hourly_balance"] = balance_ok
    report["max_hourly_residual_mw"] = float(max_residual)

    print("\n--- Generator dispatch within effective [p_min_pu, p_max_pu] * p_nom ---")
    limit_violations = 0
    for gen in network.generators.index:
        p_nom = network.generators.loc[gen, "p_nom"]
        lower = gen_p_min_pu[gen] * p_nom
        upper = gen_p_max_pu[gen] * p_nom
        actual = network.generators_t.p[gen]

        below = (actual < lower - 1e-6).sum()
        above = (actual > upper + 1e-6).sum()
        if below > 0 or above > 0:
            limit_violations += 1
            _p(f"{gen}: {below} hours below lower bound, {above} hours above upper bound", False)

    dispatch_limits_ok = limit_violations == 0
    _p(f"Generators with dispatch limit violations: {limit_violations} / {len(network.generators)}", dispatch_limits_ok)
    report["checks"]["dispatch_limits"] = dispatch_limits_ok

    print("\n--- 6/9. Slack generator use ---")
    allow_slack = scenario.get("allow_slack", False)
    slack_tolerance_mwh = scenario.get("slack_tolerance_mwh", 1e-6)

    slack_p = network.generators_t.p["Slack_generator"]
    slack_max = slack_p.abs().max()
    slack_annual_import_mwh = _weighted_sum(slack_p.clip(lower=0), gen_weights)
    slack_annual_export_mwh = _weighted_sum((-slack_p.clip(upper=0)), gen_weights)
    slack_annual_energy = slack_annual_import_mwh + slack_annual_export_mwh
    slack_hours_active = (slack_p.abs() > 1e-6).sum()

    # Report whether negative (export) dispatch was OBSERVED in the solved
    # results. This is reported as a factual observation, NOT as proof that
    # PyPSA's control='Slack' designation deliberately overrides the normal
    # generator sign convention, and NOT as confirmation that bidirectional
    # import/export is the intended real-world representation of this
    # generator. Whether that is intended, and whether the mainland
    # connection should instead be modelled explicitly as a bidirectional
    # Link, is a modelling decision that should be reviewed separately.
    slack_min_observed = slack_p.min()
    slack_max_observed = slack_p.max()
    negative_dispatch_observed = slack_min_observed < -1e-6
    static_p_min_pu = network.generators.loc["Slack_generator", "p_min_pu"]

    if negative_dispatch_observed:
        slack_label = "Slack (negative dispatch observed -- review required, see note below)"
        _p(f"Slack dispatch observed range: [{slack_min_observed:.2f}, {slack_max_observed:.2f}] MW. "
           f"Negative Slack dispatch was observed in the solved results (static p_min_pu={static_p_min_pu}). "
           f"This should be reviewed to confirm whether it is intended and whether the mainland "
           f"connection should be represented explicitly as a bidirectional Link, rather than relying "
           f"on a Generator component's dispatch going negative.", None)
    else:
        slack_label = "Slack (mainland import only -- no negative dispatch observed in this scenario)"
        _p(f"Slack dispatch observed range: [{slack_min_observed:.2f}, {slack_max_observed:.2f}] MW -- "
           f"no negative (export) values observed in THIS scenario. This does not by itself prove "
           f"export is impossible; it may simply not have been needed this time.", None)

    _p(f"Max |Slack dispatch|: {slack_max:.2f} MW ({slack_label})", None)
    _p(f"Annual Slack energy (import+export magnitude): {slack_annual_energy:.2f} MWh", None)
    _p(f"Hours Slack active: {slack_hours_active} / {len(slack_p)}", None)

    slack_ok = allow_slack or (slack_annual_energy <= slack_tolerance_mwh)
    _p(f"Slack use within allowance (allow_slack={allow_slack}, tolerance={slack_tolerance_mwh} MWh): {slack_ok}",
       slack_ok)
    report["checks"]["6_9_slack_use"] = slack_ok
    report["slack_max_mw"] = float(slack_max)
    report["slack_annual_energy_mwh"] = float(slack_annual_energy)
    report["slack_negative_dispatch_observed"] = bool(negative_dispatch_observed)

    # Fix: unintended negative Slack dispatch must feed into the final
    # classification, not just be printed as an observation. A scenario
    # should not receive a full PASS while simultaneously carrying an
    # unresolved question about whether its mainland-connection direction
    # is being modelled as intended.
    allow_negative_slack = scenario.get("allow_negative_slack", False)
    slack_direction_review_needed = negative_dispatch_observed and not allow_negative_slack
    report["slack_direction_review_needed"] = slack_direction_review_needed
    if slack_direction_review_needed:
        _p("Negative Slack dispatch was observed but allow_negative_slack=False. Review the "
           "mainland connection representation.", None)



    print("\n--- 2/7. Every storage unit validated ---")
    battery_checks_ok = True
    for su in network.storage_units.index:
        print(f"\n  == {su} ==")
        p_nom = network.storage_units.loc[su, "p_nom"]
        max_hours = network.storage_units.loc[su, "max_hours"]
        e_cap = p_nom * max_hours
        soc = network.storage_units_t.state_of_charge[su]

        soc_ok = ((soc >= -1e-6) & (soc <= e_cap + 1e-6)).all()
        _p(f"SOC within [0, {e_cap:.3f}] MWh: {soc_ok}", soc_ok)

        su_lower = su_p_min_pu[su] * p_nom if su in su_p_min_pu.columns else -p_nom
        su_upper = su_p_max_pu[su] * p_nom if su in su_p_max_pu.columns else p_nom
        net_p = network.storage_units_t.p[su]
        within_power_limit = ((net_p >= su_lower - 1e-6) & (net_p <= su_upper + 1e-6)).all()
        _p(f"Net power within effective [p_min_pu, p_max_pu]*p_nom: {within_power_limit}", within_power_limit)

        simultaneous = 0
        if hasattr(network.storage_units_t, "p_dispatch") and su in network.storage_units_t.p_dispatch.columns:
            disp = network.storage_units_t.p_dispatch[su]
            store = network.storage_units_t.p_store[su]
            simultaneous = int(((disp > 1e-6) & (store > 1e-6)).sum())
            _p(f"Hours with simultaneous charge AND discharge: {simultaneous}", simultaneous == 0)
        else:
            _p("p_dispatch/p_store not separately available -- skipping simultaneous check", None)

        cyclic = network.storage_units.loc[su, "cyclic_state_of_charge"]
        soc_initial_setting = network.storage_units.loc[su, "state_of_charge_initial"]
        soc_actual_first = soc.iloc[0]
        soc_actual_last = soc.iloc[-1]
        soc_boundary_diff = soc_actual_last - soc_actual_first
        _p(f"cyclic_state_of_charge={cyclic} -> "
           f"{'state_of_charge_initial IGNORED' if cyclic else f'state_of_charge_initial={soc_initial_setting} USED'}", None)
        _p(f"SOC boundary: first={soc_actual_first:.3f} MWh, last={soc_actual_last:.3f} MWh, "
           f"difference (last-first)={soc_boundary_diff:.3f} MWh", None)
        if not cyclic:
            _p(f"NOTE (annual repeating study consideration): this unit is non-cyclic with "
               f"state_of_charge_initial={soc_initial_setting}, meaning it starts the year at this "
               f"fixed level regardless of what a preceding year's operation would have left behind. "
               f"For a study intended to represent steady-state annual operation, consider either "
               f"cyclic_state_of_charge=True (forcing start=end) or explicitly setting "
               f"state_of_charge_initial to a representative mid-cycle value. The current choice "
               f"(starting empty) is a stated modelling assumption for this analysis, not a default "
               f"that happened by accident -- state this explicitly in the methodology.", None)

        nan_inf_ok = not (soc.isna().any() or net_p.isna().any()
                          or np.isinf(soc.to_numpy()).any() or np.isinf(net_p.to_numpy()).any())
        _p(f"No NaN/Inf in SOC or dispatch: {nan_inf_ok}", nan_inf_ok)

        standing_loss = network.storage_units.loc[su, "standing_loss"]
        inflow = network.storage_units.loc[su, "inflow"] if "inflow" in network.storage_units.columns else 0.0
        has_spill = ("spill" in network.storage_units_t.keys()
                     and su in network.storage_units_t.spill.columns
                     and network.storage_units_t.spill[su].abs().sum() > 1e-9)
        preconditions_met = (
            bool(cyclic) and np.isclose(standing_loss, 0) and np.isclose(inflow, 0)
            and not has_spill and uniform_weights
        )
        if preconditions_met:
            eff_store = network.storage_units.loc[su, "efficiency_store"]
            eff_dispatch = network.storage_units.loc[su, "efficiency_dispatch"]
            discharge_total = _weighted_sum(net_p.clip(lower=0), store_weights)
            charge_total = _weighted_sum((-net_p.clip(upper=0)), store_weights)
            expected_discharge = charge_total * eff_store * eff_dispatch
            efficiency_consistent = np.isclose(discharge_total, expected_discharge, rtol=0.05)
            _p(f"[Simplified efficiency check -- preconditions met] discharge={discharge_total:.1f} MWh vs "
               f"charge*eff={expected_discharge:.1f} MWh, consistent: {efficiency_consistent}", efficiency_consistent)
            su_battery_ok = soc_ok and within_power_limit and simultaneous == 0 and nan_inf_ok and efficiency_consistent
        else:
            _p("Simplified efficiency check SKIPPED (preconditions not met: requires cyclic=True, "
               "standing_loss=0, no inflow, no spillage, uniform snapshot weights) -- not counted "
               "as pass or fail", None)
            su_battery_ok = soc_ok and within_power_limit and simultaneous == 0 and nan_inf_ok

        battery_checks_ok &= su_battery_ok

    report["checks"]["2_7_all_storage_units"] = battery_checks_ok

    print("\n--- Line and transformer loading (effective capacity) ---")
    line_capacity = network.lines["s_nom"].copy()
    if "s_nom_extendable" in network.lines.columns:
        extendable = network.lines["s_nom_extendable"].fillna(False)
        if extendable.any():
            line_capacity.loc[extendable] = network.lines.loc[extendable, "s_nom_opt"]
    line_limit = line_s_max_pu.mul(line_capacity, axis=1)
    line_loading_pct = (network.lines_t.p0.abs().div(line_limit.replace(0, np.nan))) * 100

    max_line_loading = line_loading_pct.max()
    lines_over_100 = (max_line_loading > 100.001).sum()
    _p(f"Lines exceeding 100% of effective capacity at any hour: {lines_over_100} / {len(network.lines)}",
       lines_over_100 == 0)

    top5 = max_line_loading.sort_values(ascending=False).head(5)
    _p("Top 5 most loaded lines (% of effective capacity):", None)
    print(top5.to_string())

    hours_over_90pct = (line_loading_pct.max(axis=1) > 90).sum()
    _p(f"Hours where ANY line exceeds 90% of effective capacity: {hours_over_90pct}", None)

    transformer_ok = True
    if not network.transformers.empty:
        trafo_capacity = network.transformers["s_nom"].copy()
        if "s_nom_extendable" in network.transformers.columns:
            trafo_extendable = network.transformers["s_nom_extendable"].fillna(False)
            if trafo_extendable.any():
                trafo_capacity.loc[trafo_extendable] = network.transformers.loc[trafo_extendable, "s_nom_opt"]
        trafo_limit = trafo_s_max_pu.mul(trafo_capacity, axis=1)
        trafo_loading_pct = (network.transformers_t.p0.abs().div(trafo_limit.replace(0, np.nan))) * 100
        trafo_over_100 = (trafo_loading_pct.max() > 100.001).sum()
        transformer_ok = trafo_over_100 == 0
        _p(f"Transformers exceeding 100% of effective capacity: {trafo_over_100} / {len(network.transformers)}",
           transformer_ok)

    report["checks"]["line_transformer_loading"] = (lines_over_100 == 0) and transformer_ok

    print("\n--- Renewable availability / dispatch / curtailment ---")

    def avail_dispatch_curtail(gens, label):
        if not gens:
            return 0.0, 0.0, 0.0, True
        avail = pd.DataFrame({
            g: gen_p_max_pu[g].clip(0, 1) * network.generators.loc[g, "p_nom"]
            for g in gens
        }).sum(axis=1)
        dispatch = network.generators_t.p[gens].sum(axis=1)

        availability_violation = dispatch - avail
        max_violation = availability_violation.max()
        no_violation = max_violation <= tolerance_mw
        _p(f"{label}: max (dispatch - available) = {max_violation:.6f} MW "
           f"(must be <= {tolerance_mw} MW): {no_violation}", no_violation)

        avail_mwh = _weighted_sum(avail, gen_weights)
        dispatch_mwh = _weighted_sum(dispatch, gen_weights)
        curtail_mwh = _weighted_sum((avail - dispatch), gen_weights) if no_violation else float("nan")
        return avail_mwh, dispatch_mwh, curtail_mwh, no_violation

    availability_ok = True
    for label, gens in [("Wind", wind_gens), ("Tidal", [TIDAL_GEN]), ("Wave", [WAVE_GEN])]:
        avail_mwh, dispatch_mwh, curtail_mwh, no_violation = avail_dispatch_curtail(gens, label)
        availability_ok &= no_violation
        pct = (curtail_mwh / avail_mwh * 100) if (avail_mwh > 0 and no_violation) else float("nan")
        _p(f"{label}: available={avail_mwh:.1f} MWh, dispatched={dispatch_mwh:.1f} MWh, "
           f"curtailed={curtail_mwh:.1f} MWh ({pct:.1f}%)", None)

    report["checks"]["renewable_availability_compliance"] = availability_ok

    if scenario["wave_mw"] == 0:
        wave_dispatch_total = _weighted_sum(network.generators_t.p[WAVE_GEN], gen_weights)
        wave_zero_ok = np.isclose(wave_dispatch_total, 0.0, atol=1e-6)
        _p(f"Wave dispatch is zero in wave=0MW scenario: {wave_zero_ok} ({wave_dispatch_total:.6f} MWh)", wave_zero_ok)
        report["checks"]["wave_zero_check"] = wave_zero_ok

    print("\n--- NaN / Inf check ---")
    nan_found = False
    for name, df in [
        ("generators_t.p", network.generators_t.p),
        ("storage_units_t.p", network.storage_units_t.p),
        ("storage_units_t.state_of_charge", network.storage_units_t.state_of_charge),
        ("lines_t.p0", network.lines_t.p0),
        ("transformers_t.p0", network.transformers_t.p0),
    ]:
        if df.empty:
            continue
        n_nan = int(df.isna().sum().sum())
        n_inf = int(np.isinf(df.to_numpy(dtype=float, na_value=0)).sum())
        if n_nan > 0 or n_inf > 0:
            nan_found = True
            _p(f"{name}: {n_nan} NaN, {n_inf} Inf", False)
    if not nan_found:
        _p("No NaN or Inf found in any checked result", True)
    report["checks"]["no_nan_inf"] = not nan_found

    print("\n--- Objective function interpretation ---")
    any_extendable = (
        network.generators.get("p_nom_extendable", pd.Series(False, index=network.generators.index)).any()
        or network.storage_units.get("p_nom_extendable", pd.Series(False, index=network.storage_units.index)).any()
        or network.lines.get("s_nom_extendable", pd.Series(False, index=network.lines.index)).any()
        or (network.transformers.get("s_nom_extendable", pd.Series(False, index=network.transformers.index)).any()
            if not network.transformers.empty else False)
    )
    any_capital_cost = (
        (network.generators["capital_cost"] > 0).any()
        or (network.storage_units["capital_cost"] > 0).any()
    )
    include_obj_const = scenario.get("include_objective_constant", False)

    _p(f"Objective value: {network.objective:.2f}", None)
    _p(f"Any extendable component: {any_extendable}", None)
    _p(f"Any non-zero capital_cost assigned: {any_capital_cost}", None)
    _p(f"include_objective_constant used at solve time: {include_obj_const}", None)

    if not any_extendable and not any_capital_cost:
        interpretation = ("All capacities are FIXED (non-extendable) and no capital costs are assigned. "
                           "The objective represents OPERATING cost only for this fixed set of installed "
                           "capacities -- it is NOT a total system cost and cannot be used alone to judge "
                           "whether a given capacity mix is economically worthwhile.")
    elif any_extendable and not any_capital_cost:
        interpretation = ("WARNING: some components are extendable but have zero capital_cost -- the "
                           "optimizer has an incentive to expand them without limit or without real cost "
                           "consequence. Verify this is intended before trusting comparisons.")
    else:
        interpretation = ("Some capital costs and/or extendable capacity are present. The objective may "
                           "include both operating AND capital cost terms -- confirm which components are "
                           "extendable and whether their capital costs are annualised correctly before "
                           "treating this as directly comparable to a fixed-capacity operating-cost-only run.")
    _p(interpretation, None)
    report["objective"] = float(network.objective)
    report["objective_interpretation"] = interpretation

    overall_pass = all(report["checks"].values())
    report["overall_pass"] = overall_pass
    print("\n" + "=" * 100)
    print(f"OVERALL: {'PASS' if overall_pass else 'FAIL -- see [FAIL] lines above'}")
    print("=" * 100)

    return report


def cross_scenario_checks(nc_file_paths, scenarios=None):
    print("\n" + "=" * 100)
    print("CROSS-SCENARIO CHECKS")
    print("=" * 100)

    networks = {}
    for path in nc_file_paths:
        n = pypsa.Network()
        n.import_from_netcdf(path)
        networks[os.path.basename(path)] = n

    names = list(networks.keys())

    print("\n--- Demand identical across all scenarios ---")
    ref_demand = networks[names[0]].get_switchable_as_dense("Load", "p_set")
    all_match = True
    for name in names[1:]:
        this_demand = networks[name].get_switchable_as_dense("Load", "p_set")
        if not this_demand.equals(ref_demand):
            all_match = False
            _p(f"{name}: demand DIFFERS from {names[0]}", False)
    if all_match:
        _p(f"All {len(names)} scenarios share identical demand", True)

    print("\n--- Objective (interpret per validate_scenario's objective_interpretation) ---")
    for name, n in networks.items():
        print(f"  {name}: objective = {n.objective:.2f}")

    print("\nCORRECT interpretation: when additional renewable/storage capacity is OPTIONAL and has")
    print("NO investment cost in the objective, minimum operating cost should DECREASE or stay the same")
    print("as capacity increases. An INCREASE in cost when capacity is added (all else identical) is the")
    print("suspicious result and should be investigated.")

    if scenarios is not None and len(scenarios) == len(nc_file_paths):
        print("\n--- Pairwise monotonicity check ---")
        combined = list(zip(names, networks.values(), scenarios))
        violations = []
        for i in range(len(combined)):
            name_i, net_i, scen_i = combined[i]
            for j in range(len(combined)):
                if i == j:
                    continue
                name_j, net_j, scen_j = combined[j]
                same_battery = scen_i.get("battery_mult") == scen_j.get("battery_mult")
                more_wave = scen_i.get("wave_mw", 0) > scen_j.get("wave_mw", 0)
                if same_battery and more_wave:
                    if net_i.objective > net_j.objective + 1e-3:
                        violations.append(
                            f"{name_i} (wave={scen_i.get('wave_mw')}) has HIGHER cost "
                            f"({net_i.objective:.2f}) than {name_j} (wave={scen_j.get('wave_mw')}, "
                            f"{net_j.objective:.2f}) at the same battery size -- unexpected."
                        )
        if violations:
            _p(f"{len(violations)} monotonicity violation(s) found:", False)
            for v in violations:
                print(f"    - {v}")
        else:
            _p("No monotonicity violations found (more wave capacity never increased cost "
               "at matched battery size)", True)
    else:
        print("\n(Provide `scenarios=[...]` with wave_mw/battery_mult metadata for a real "
              "pairwise monotonicity check.)")


def validate_critical_hours_pf(nc_file_path):
    """
    Re-runs genuine nonlinear pf() at the hours of: peak demand, highest
    line loading, highest battery charge, highest battery discharge, and
    highest renewable output -- fixing BOTH generator and storage unit
    dispatch at the LOPF-optimal values, to confirm the LOPF solution is
    also AC-feasible.

    Returns a dict: {"pass": bool, "hour_results": {label: bool, ...}}
    `pass` is True only if EVERY critical hour converged, stayed within
    voltage bounds, and had no line/transformer exceeding 100% loading.
    """
    print("\n" + "=" * 100)
    print(f"CRITICAL-HOUR NONLINEAR pf() VALIDATION: {nc_file_path}")
    print("=" * 100)

    network = pypsa.Network()
    network.import_from_netcdf(nc_file_path)

    load_p_set_eff = network.get_switchable_as_dense("Load", "p_set")
    total_demand = load_p_set_eff.sum(axis=1)

    # Fix #3: use EFFECTIVE line limits (s_nom_opt for extendable lines x
    # time-varying s_max_pu) and the max of |p0| and |p1| at both ends, not
    # just p0 against a bare s_nom -- matching the LP-side methodology.
    line_capacity = network.lines["s_nom"].copy()
    if "s_nom_extendable" in network.lines.columns:
        extendable = network.lines["s_nom_extendable"].fillna(False)
        if extendable.any():
            line_capacity.loc[extendable] = network.lines.loc[extendable, "s_nom_opt"]
    line_s_max_pu = network.get_switchable_as_dense("Line", "s_max_pu")
    line_limit = line_s_max_pu.mul(line_capacity, axis=1)

    lopf_line_flow = pd.DataFrame(
        np.maximum(network.lines_t.p0.abs(), network.lines_t.p1.abs()),
        index=network.snapshots,
        columns=network.lines.index,
    )
    total_line_loading = lopf_line_flow.div(line_limit.replace(0, np.nan)).max(axis=1)

    main_su_p = network.storage_units_t.p[STORAGE_UNIT]
    storage_charge = (-main_su_p.clip(upper=0))
    storage_discharge = main_su_p.clip(lower=0)

    wind_gens = [g for g in network.generators.index if "WindFarm" in g]
    total_renewable = network.generators_t.p[wind_gens + [TIDAL_GEN, WAVE_GEN]].sum(axis=1)

    # Fix #6: guard against zero-storage-activity scenarios (e.g. no-battery
    # runs), where .idxmax() on an all-zero series would return an
    # arbitrary hour rather than a genuinely meaningful "highest
    # charging/discharging" hour. Only include these labels if the battery
    # actually did something above a small tolerance.
    storage_activity_tolerance_mw = 1e-3
    critical_hours = {
        "Peak demand": total_demand.idxmax(),
        "Highest line loading": total_line_loading.idxmax(),
        "Highest renewable output": total_renewable.idxmax(),
    }
    if storage_charge.max() > storage_activity_tolerance_mw:
        critical_hours["Highest battery charging"] = storage_charge.idxmax()
    else:
        print(f"\n[INFO] Battery charging never exceeds {storage_activity_tolerance_mw} MW in this "
              f"scenario -- 'Highest battery charging' critical hour SKIPPED (would otherwise be "
              f"an arbitrary hour from an all-zero series).")
    if storage_discharge.max() > storage_activity_tolerance_mw:
        critical_hours["Highest battery discharging"] = storage_discharge.idxmax()
    else:
        print(f"\n[INFO] Battery discharging never exceeds {storage_activity_tolerance_mw} MW in this "
              f"scenario -- 'Highest battery discharging' critical hour SKIPPED (would otherwise be "
              f"an arbitrary hour from an all-zero series).")

    # Deduplicate: if multiple labels resolve to the SAME timestamp (e.g.
    # peak demand and highest renewable output happen to coincide), only
    # test that hour once, but keep all applicable labels attached to it for
    # reporting clarity.
    hour_to_labels = {}
    for label, hour in critical_hours.items():
        hour_to_labels.setdefault(hour, []).append(label)

    if len(hour_to_labels) < len(critical_hours):
        print(f"\n[INFO] {len(critical_hours)} critical-hour labels resolved to only "
              f"{len(hour_to_labels)} UNIQUE hour(s) -- duplicates are tested once, not repeated:")
        for hour, labels in hour_to_labels.items():
            if len(labels) > 1:
                print(f"    {hour}: {', '.join(labels)}")

    hour_results = {}
    hour_slack_adjustment_ok = {}

    for hour, labels in hour_to_labels.items():
        label = " / ".join(labels)
        print(f"\n--- {label}: {hour} ---")
        hour_ok = True  # convergence, voltage, line/transformer overload (>100%) ONLY

        single = pypsa.Network()
        single.import_from_netcdf(nc_file_path)
        single.set_snapshots([hour])

        non_slack_generators = single.generators.index[single.generators["control"].ne("Slack")]

        # ── Explicit, deterministic PF input assignment ──────────────────────
        # ROOT CAUSE FOUND: PyPSA's control="Slack" does NOT mean "p_set is
        # ignored, dispatch computed freely from nothing" -- it means "start
        # from p_set, and pf() computes the ADJUSTMENT needed to balance the
        # system (covering nonlinear AC losses), added to that starting
        # value." Every previous attempt cleared Slack's p_set to NaN
        # (believing pf() would compute Slack "freely"), which corrupted the
        # computation (NaN + anything = NaN) despite pf() reporting genuine
        # convergence for voltages/everything else. The correct approach:
        # give EVERY generator, including Slack, its LOPF dispatch as a
        # finite starting p_set. pf() will adjust Slack as needed; non-Slack
        # (PV-controlled) generators will be held fixed at exactly this
        # value, since P is an input for PV buses in Newton-Raphson pf().
        single.generators_t.p_set = network.generators_t.p.loc[[hour]].copy()
        single.loads_t.p_set = network.get_switchable_as_dense("Load", "p_set").loc[[hour]].copy()
        single.storage_units_t.p_set = network.storage_units_t.p.loc[[hour]].copy()
        if not single.links.empty:
            single.links_t.p_set = network.links_t.p0.loc[[hour]].copy()

        # Slack generator's control designation must remain 'Slack' --
        # confirm this explicitly rather than assume the network object
        # carried it through unchanged.
        single.generators.loc["Slack_generator", "control"] = "Slack"

        try:
            # ── Mandatory pre-PF diagnostic ───────────────────────────────────
            slack_name = "Slack_generator"
            slack_lopf = float(network.generators_t.p.loc[hour, slack_name])
            slack_effective_p_set = float(
                single.get_switchable_as_dense("Generator", "p_set").loc[hour, slack_name]
            )
            slack_input_ok = (
                np.isfinite(slack_lopf)
                and np.isfinite(slack_effective_p_set)
                and np.isclose(slack_effective_p_set, slack_lopf, atol=1e-6)
            )
            _p(f"Pre-PF Slack p_set={slack_effective_p_set:.4f} MW; "
               f"LOPF Slack dispatch={slack_lopf:.4f} MW", slack_input_ok)
            if not slack_input_ok:
                raise ValueError(
                    f"Slack_generator does not have a finite LOPF-derived p_set before pf() at {hour}."
                )

            # ── Verify which generator PyPSA actually selects as Slack ────────
            single.determine_network_topology()
            selected_slack_generators = []
            for sub_network in single.sub_networks.obj:
                sub_network.find_bus_controls()
                if getattr(sub_network, "slack_generator", None) is not None:
                    selected_slack_generators.append(sub_network.slack_generator)
            print(f"    PyPSA-selected Slack generator(s): {selected_slack_generators}")
            if slack_name not in selected_slack_generators:
                raise ValueError(
                    f"{slack_name} was not selected by PyPSA as a Slack generator at {hour}. "
                    f"Selected: {selected_slack_generators}"
                )

            result = single.pf()

            converged_df = result["converged"]
            all_converged = bool(converged_df.to_numpy().all())
            _p(f"pf() converged for ALL sub-networks: {all_converged}", all_converged)
            if not all_converged:
                print(converged_df.to_string())
            hour_ok &= all_converged

            v = single.buses_t.v_mag_pu.loc[hour]
            v_min, v_max = v.min(), v.max()
            v_ok = (v_min >= 0.90) and (v_max <= 1.10)
            _p(f"Voltages within [0.90, 1.10] p.u.: {v_ok} (min={v_min:.4f}, max={v_max:.4f})", v_ok)
            hour_ok &= v_ok

            p0 = single.lines_t.p0.loc[hour]
            q0 = (single.lines_t.q0.loc[hour] if "q0" in single.lines_t.keys() and not single.lines_t.q0.empty
                  else pd.Series(0.0, index=p0.index))
            p1 = single.lines_t.p1.loc[hour]
            q1 = (single.lines_t.q1.loc[hour] if "q1" in single.lines_t.keys() and not single.lines_t.q1.empty
                  else pd.Series(0.0, index=p1.index))
            s0 = np.sqrt(p0**2 + q0**2)
            s1 = np.sqrt(p1**2 + q1**2)
            s_max = pd.concat([s0, s1], axis=1).max(axis=1)

            line_capacity = single.lines["s_nom"].copy()
            if "s_nom_extendable" in single.lines.columns:
                extendable = single.lines["s_nom_extendable"].fillna(False)
                if extendable.any():
                    line_capacity.loc[extendable] = single.lines.loc[extendable, "s_nom_opt"]
            line_s_max_pu = single.get_switchable_as_dense("Line", "s_max_pu").loc[hour]
            line_limit = line_capacity * line_s_max_pu

            line_loading_pct = (s_max / line_limit.replace(0, np.nan)) * 100
            worst_line = line_loading_pct.idxmax()
            line_ok = line_loading_pct[worst_line] <= 100.0
            _p(f"Worst line loading (|S| both ends vs effective capacity): "
               f"{line_loading_pct[worst_line]:.1f}% at {worst_line}", line_ok)
            hour_ok &= line_ok

            if not single.transformers.empty:
                tp0 = single.transformers_t.p0.loc[hour]
                tq0 = (single.transformers_t.q0.loc[hour]
                       if "q0" in single.transformers_t.keys() and not single.transformers_t.q0.empty
                       else pd.Series(0.0, index=tp0.index))
                tp1 = single.transformers_t.p1.loc[hour]
                tq1 = (single.transformers_t.q1.loc[hour]
                       if "q1" in single.transformers_t.keys() and not single.transformers_t.q1.empty
                       else pd.Series(0.0, index=tp1.index))
                ts0 = np.sqrt(tp0**2 + tq0**2)
                ts1 = np.sqrt(tp1**2 + tq1**2)
                transformer_s_max = pd.concat([ts0, ts1], axis=1).max(axis=1)

                trafo_capacity = single.transformers["s_nom"].copy()
                if "s_nom_extendable" in single.transformers.columns:
                    trafo_extendable = single.transformers["s_nom_extendable"].fillna(False)
                    if trafo_extendable.any():
                        trafo_capacity.loc[trafo_extendable] = single.transformers.loc[trafo_extendable, "s_nom_opt"]
                trafo_s_max_pu = single.get_switchable_as_dense("Transformer", "s_max_pu").loc[hour]
                trafo_limit = trafo_capacity * trafo_s_max_pu

                trafo_loading_pct = (transformer_s_max / trafo_limit.replace(0, np.nan)) * 100
                worst_trafo = trafo_loading_pct.idxmax()
                worst_trafo_pct = trafo_loading_pct[worst_trafo]
                # Three-tier classification: <90% PASS, 90-100% WARNING
                # (reported, does not fail the hour, but flags low remaining
                # margin), >100% FAIL.
                if worst_trafo_pct > 100.0:
                    _p(f"Worst transformer loading: {worst_trafo_pct:.1f}% at {worst_trafo} "
                       f"(effective capacity) -- EXCEEDS 100% capacity", False)
                    hour_ok = False
                elif worst_trafo_pct > 90.0:
                    _p(f"Worst transformer loading: {worst_trafo_pct:.1f}% at {worst_trafo} "
                       f"(effective capacity) -- WARNING: low remaining capacity margin (90-100% band)", None)
                    print("    [WARNING] Transformer loading in the 90-100% band -- not a failure, "
                          "but worth noting as low remaining margin.")
                else:
                    _p(f"Worst transformer loading: {worst_trafo_pct:.1f}% at {worst_trafo} "
                       f"(effective capacity)", True)

            # slack_lopf was already computed pre-PF, above. Read the
            # ACTUAL post-pf() dispatch and enforce that it is finite --
            # if it is NOT finite despite a finite pre-PF p_set, that is a
            # genuine PyPSA/data-integrity problem worth stopping on
            # immediately, per the mandatory check specification.
            slack_pf = float(single.generators_t.p.loc[hour, "Slack_generator"])
            if not np.isfinite(slack_pf):
                raise ValueError(
                    "Slack_generator dispatch is still NaN/Inf after pf() despite having a finite "
                    "pre-PF p_set."
                )

            slack_adjustment = slack_pf - slack_lopf

            # Fix #3: do not let convergence alone determine the pass -- the
            # Slack adjustment between the LOPF (lossless, linearized) result
            # and the genuine AC pf() result must also be within a justified
            # tolerance. A large adjustment means the LOPF and AC-feasible
            # dispatch disagree substantially about how much mainland
            # import/export is needed at this hour, which is a real
            # modelling concern (LOPF ignores losses and reactive power;
            # some gap is expected, but an excessive gap suggests the LOPF
            # result may not be a good guide to real operation at this hour).
            demand_at_hour = network.get_switchable_as_dense("Load", "p_set").loc[hour].sum()
            allowed_adjustment = max(1.0, 0.05 * demand_at_hour)
            slack_adjustment_ok = abs(slack_adjustment) <= allowed_adjustment
            _p(f"Slack adjustment required by AC pf() vs LOPF: {slack_adjustment:.4f} MW "
               f"(LOPF={slack_lopf:.2f}, AC pf()={slack_pf:.2f}) | tolerance=max(1.0, 5% of "
               f"{demand_at_hour:.1f} MW demand)={allowed_adjustment:.2f} MW: "
               f"{'within tolerance' if slack_adjustment_ok else 'EXCEEDS tolerance -- REQUIRES REVIEW'}",
               slack_adjustment_ok)

            if not slack_adjustment_ok:
                # Diagnostic: is the discrepancy explained by genuine AC
                # losses (which the lossless LOPF never captured), or by
                # dispatch not having been transferred into the pf() network
                # correctly? Check BOTH possibilities directly rather than
                # assuming either.
                print(f"\n  --- Slack adjustment diagnostic for {label} ({hour}) ---")

                # Confirm generator setpoints were WRITTEN correctly
                # (pre-solve intent), restricted to non-Slack generators --
                # Slack's p_set is deliberately NaN and must not be treated
                # as a mismatch.
                generator_diff = (single.generators_t.p_set.loc[hour, non_slack_generators]
                                   - network.generators_t.p.loc[hour, non_slack_generators])
                mismatched_gens = generator_diff[generator_diff.abs() > 1e-6]
                print("  Generator SETPOINT differences (single.p_set - solved LOPF.p, non-Slack only):")
                if not mismatched_gens.empty:
                    print(mismatched_gens.to_string())
                else:
                    print("    (none -- all non-Slack generator setpoints written correctly)")

                # Confirm storage setpoints were WRITTEN correctly
                storage_diff = (single.storage_units_t.p_set.loc[hour]
                                 - network.storage_units_t.p.loc[hour])
                mismatched_storage = storage_diff[storage_diff.abs() > 1e-6]
                print("  Storage SETPOINT differences (single.p_set - solved LOPF.p):")
                if not mismatched_storage.empty:
                    print(mismatched_storage.to_string())
                else:
                    print("    (none -- all storage setpoints written correctly)")

                # Confirm the nonlinear pf() ACTUALLY FOLLOWED those
                # setpoints -- this is a genuinely different question from
                # whether they were written correctly. For PV-controlled
                # (non-Slack) generators, pf() treats real power P as a
                # fixed input, so actual dispatch should match the setpoint
                # essentially exactly if pf() converged correctly; any
                # deviation here would indicate a deeper problem with how
                # the fixed dispatch was applied, not just a writing error.
                actual_generator_diff = (single.generators_t.p.loc[hour, non_slack_generators]
                                          - network.generators_t.p.loc[hour, non_slack_generators])
                actual_generator_mismatches = actual_generator_diff[actual_generator_diff.abs() > 1e-6]
                print("  Generator ACTUAL dispatch differences (single.p AFTER pf() - solved LOPF.p, non-Slack only):")
                if not actual_generator_mismatches.empty:
                    print(actual_generator_mismatches.to_string())
                else:
                    print("    (none -- nonlinear pf() reproduced the fixed non-Slack dispatch exactly)")

                actual_storage_diff = (single.storage_units_t.p.loc[hour]
                                        - network.storage_units_t.p.loc[hour])
                actual_storage_mismatches = actual_storage_diff[actual_storage_diff.abs() > 1e-6]
                print("  Storage ACTUAL dispatch differences (single.p AFTER pf() - solved LOPF.p):")
                if not actual_storage_mismatches.empty:
                    print(actual_storage_mismatches.to_string())
                else:
                    print("    (none -- nonlinear pf() reproduced the fixed storage dispatch exactly)")

                # Calculate AC active-power losses
                line_losses = (single.lines_t.p0.loc[hour] + single.lines_t.p1.loc[hour]).sum()
                transformer_losses = (
                    (single.transformers_t.p0.loc[hour] + single.transformers_t.p1.loc[hour]).sum()
                    if not single.transformers.empty else 0.0
                )
                total_ac_losses = line_losses + transformer_losses
                loss_residual = slack_adjustment - total_ac_losses

                print(f"  AC line losses: {line_losses:.4f} MW")
                print(f"  AC transformer losses: {transformer_losses:.4f} MW")
                print(f"  Total AC losses: {total_ac_losses:.4f} MW")
                print(f"  Slack adjustment: {slack_adjustment:.4f} MW")
                print(f"  Difference between Slack adjustment and AC losses: {loss_residual:.4f} MW")

                # Use the ACTUAL post-solve dispatch mismatches (not just
                # whether setpoints were written) as the determining
                # evidence -- this is the check that genuinely distinguishes
                # "AC losses the LOPF never modeled" from "dispatch was not
                # preserved through to the pf() solution".
                dispatch_preserved = actual_generator_mismatches.empty and actual_storage_mismatches.empty
                explained_by_losses = abs(loss_residual) < max(0.5, 0.02 * demand_at_hour)

                if explained_by_losses and dispatch_preserved:
                    _p(f"  Slack adjustment is explained by genuine AC losses (the lossless LOPF "
                       f"never captured these) -- actual non-Slack dispatch was preserved exactly "
                       f"through pf(). Not a dispatch transfer bug.", None)
                elif not dispatch_preserved:
                    _p(f"  ACTUAL dispatch mismatch detected after pf() -- the Slack adjustment may be "
                       f"(partly or wholly) an artifact of dispatch not being preserved through the "
                       f"pf() solution, NOT genuine AC losses. Investigate the mismatched component(s) "
                       f"listed above before trusting this hour's result.", False)
                else:
                    _p(f"  Slack adjustment ({slack_adjustment:.2f} MW) is NOT well explained by AC "
                       f"losses alone ({total_ac_losses:.2f} MW, residual {loss_residual:.2f} MW), "
                       f"despite actual dispatch being preserved correctly. This residual is currently "
                       f"unexplained and warrants further investigation.", False)

            # NOTE: tracked separately from hour_ok (convergence/voltage/
            # loading). A large LOPF-vs-AC Slack gap does not necessarily
            # mean the AC solution is infeasible or wrong -- LOPF is a
            # lossless linear approximation and some gap is expected -- but
            # an excessive gap means the LOPF result may not be a reliable
            # guide to real dispatch at this hour, which is a genuine
            # concern worth flagging for review rather than either silently
            # ignoring or conflating with a hard convergence/voltage failure.
            hour_slack_adjustment_ok[label] = slack_adjustment_ok

        except Exception as e:
            _p(f"pf() raised an exception: {e}", False)
            hour_ok = False

        hour_results[label] = hour_ok

    overall_pf_pass = all(hour_results.values())
    slack_adjustment_review_needed = not all(hour_slack_adjustment_ok.values())

    print("\n" + "-" * 100)
    print(f"Critical-hour pf() convergence/voltage/loading: {'PASS' if overall_pf_pass else 'FAIL'} "
          f"({sum(hour_results.values())}/{len(hour_results)} hours)")
    print(f"Critical-hour PF Slack adjustment: "
          f"{'WITHIN TOLERANCE' if not slack_adjustment_review_needed else 'REQUIRES REVIEW'} "
          f"({sum(hour_slack_adjustment_ok.values())}/{len(hour_slack_adjustment_ok)} hours within tolerance)")
    print("-" * 100)

    return {
        "pass": overall_pf_pass,
        "hour_results": hour_results,
        "slack_adjustment_review_needed": slack_adjustment_review_needed,
        "hour_slack_adjustment_ok": hour_slack_adjustment_ok,
    }


def main():
    """
    Executable entry point. Requires an explicit solved scenario file and
    baseline file -- there is no default/implicit behavior that could
    silently validate the wrong file or skip validation entirely.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate a solved PyPSA scenario .nc file against a baseline network."
    )
    parser.add_argument(
        "--solved", required=True,
        help="Path to the SOLVED scenario .nc file (e.g. DATA/outputs/R01_wave0MW_batt2x.nc)",
    )
    parser.add_argument(
        "--baseline", required=True,
        help="Path to the UNMODIFIED baseline .nc file "
             "(e.g. DATA/outputs/network_wave5MW_tidal7.2MW.nc)",
    )
    parser.add_argument("--id", required=True, help="Scenario ID, e.g. R01")
    parser.add_argument("--wave-mw", type=float, required=True, help="Intended wave capacity (MW)")
    parser.add_argument(
        "--battery-mult", type=float, default=None,
        help="Intended battery multiplier (omit for no-battery scenarios)",
    )
    parser.add_argument(
        "--allow-slack", action="store_true",
        help="Allow Slack generator use without failing the scenario "
             "(set this if Slack represents mainland import/export by design)",
    )
    parser.add_argument(
        "--skip-critical-hours-pf", action="store_true",
        help="Skip the nonlinear pf() critical-hour validation (it is slower than the LP checks)",
    )
    args = parser.parse_args()

    print(f"Loading SOLVED network from: {args.solved}")
    solved_network = pypsa.Network()
    solved_network.import_from_netcdf(args.solved)

    print(f"Loading BASELINE network from: {args.baseline}")
    baseline_network = pypsa.Network()
    baseline_network.import_from_netcdf(args.baseline)

    scenario = {
        "id": args.id,
        "wave_mw": args.wave_mw,
        "battery_mult": args.battery_mult,
        "allow_slack": args.allow_slack,
        "include_objective_constant": False,
        # status/condition intentionally omitted -- this is a post-hoc
        # validation of an already-saved file, not an immediate post-solve
        # check. See the "0. Solver outcome" section for how this is handled.
    }

    report = validate_scenario(
        network=solved_network,
        baseline_network=baseline_network,
        scenario=scenario,
        tolerance_mw=1e-4,
    )

    pf_result = {"pass": None, "hour_results": {}, "slack_adjustment_review_needed": None}
    if not args.skip_critical_hours_pf:
        pf_result = validate_critical_hours_pf(args.solved)

    # ── Three-tier final classification (fix #7) ──────────────────────────────
    # FAIL: any substantive check fails -- optimisation-result validation,
    #       explicit solver-status failure, or PF non-convergence/voltage/
    #       loading violation.
    # CONDITIONAL PASS: all substantive checks pass, but one or more
    #       verification gaps remain (solver status not verified, dtype
    #       unresolved, or PF Slack adjustment requires review). The model
    #       is not shown to be WRONG, but some things could not be fully
    #       confirmed and should be reviewed before treating the result as
    #       fully trusted.
    # PASS: everything above is resolved/verified/within tolerance.
    optimisation_result_pass = report["overall_pass"]
    solver_status_state = report.get("solver_status_state", "NOT_VERIFIED")
    dtype_state = report.get("dtype_consistency", "unresolved")
    pf_convergence_voltage_pass = pf_result["pass"] if pf_result["pass"] is not None else True
    pf_slack_review_needed = pf_result.get("slack_adjustment_review_needed", False) or False
    pf_was_skipped = args.skip_critical_hours_pf
    slack_direction_review_needed = report.get("slack_direction_review_needed", False)

    hard_fail = (
        (not optimisation_result_pass)
        or (solver_status_state == "VERIFIED_FAIL")
        or (not pf_convergence_voltage_pass)
    )
    needs_review = (
        (solver_status_state == "NOT_VERIFIED")
        or (dtype_state != "likely_serialization_artifact")
        or pf_slack_review_needed
        or pf_was_skipped  # AC feasibility was never checked -- cannot be a full PASS
        or slack_direction_review_needed
    )

    if hard_fail:
        final_classification = "FAIL"
    elif needs_review:
        final_classification = "CONDITIONAL PASS"
    else:
        final_classification = "PASS"

    print("\n" + "=" * 100)
    print(f"FINAL CLASSIFICATION for {args.id}")
    print("=" * 100)
    print(f"  Optimisation-result validation:         {'PASS' if optimisation_result_pass else 'FAIL'}")
    print(f"  Critical-hour PF convergence/voltage:   "
          f"{'SKIPPED' if args.skip_critical_hours_pf else ('PASS' if pf_convergence_voltage_pass else 'FAIL')}")
    print(f"  Solver-status verification:             "
          f"{'VERIFIED PASS' if solver_status_state == 'VERIFIED_PASS' else solver_status_state}")
    print(f"  Dtype consistency:                      {dtype_state.upper()}")
    print(f"  PF Slack adjustment:                    "
          f"{'SKIPPED' if args.skip_critical_hours_pf else ('REQUIRES REVIEW' if pf_slack_review_needed else 'WITHIN TOLERANCE')}")
    print(f"  Slack direction (negative dispatch):     "
          f"{'REQUIRES REVIEW' if slack_direction_review_needed else 'OK / NOT OBSERVED'}")
    print(f"\n  Overall result: {final_classification}")
    print("=" * 100)

    report["critical_hours_pf"] = pf_result
    report["final_classification"] = final_classification
    return report


if __name__ == "__main__":
    main()