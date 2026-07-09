"""
run_validation_R01.py

Standalone runner: loads the SOLVED R01 scenario file and the UNMODIFIED
baseline file, then runs the full validation suite against them. This is
the simplest way to validate a specific, already-solved scenario file
without needing command-line arguments.

For validating other scenarios, copy this file and change SOLVED_FILE and
the `scenario` dict accordingly -- or use validate_scenario.py's
command-line interface instead:

    python validate_scenario.py --solved DATA/outputs/R02_wave5MW_batt2x.nc \\
        --baseline DATA/outputs/network_wave5MW_tidal7.2MW.nc \\
        --id R02 --wave-mw 5 --battery-mult 2 --allow-slack
"""

import pypsa
from validate_scenario import validate_scenario, validate_critical_hours_pf

SOLVED_FILE = "DATA/outputs/R01_wave0MW_batt2x.nc"
BASELINE_FILE = "DATA/outputs/network_wave5MW_tidal7.2MW.nc"


def main():
    print(f"Loading SOLVED network from: {SOLVED_FILE}")
    solved_network = pypsa.Network()
    solved_network.import_from_netcdf(SOLVED_FILE)

    print(f"Loading BASELINE network from: {BASELINE_FILE}")
    baseline_network = pypsa.Network()
    baseline_network.import_from_netcdf(BASELINE_FILE)

    scenario = {
        "id": "R01",
        "wave_mw": 0,
        "battery_mult": 2,
        # Explicit boundary-condition assumptions for the main storage unit,
        # validated (not just reported) inside validate_scenario().
        "main_storage_cyclic": False,
        "main_storage_initial_soc": 0.0,
        # R01's actual scenario script applies a renewable dispatch-priority
        # marginal cost ordering (Wave < Tidal < Wind) -- this MUST be
        # declared explicitly now, since the validator no longer
        # auto-whitelists any change to renewable generators' costs.
        # Wind's marginal cost (30.00) happens to already match the
        # baseline, so no entry is needed for wind turbines.
        "allowed_marginal_cost_changes": {
            "Wave Generator": 29.90,
            "EDAY Tidal Generator": 29.99,
        },
        # The direction of Slack dispatch observed in the solved results is
        # reported empirically. This does not by itself prove intended
        # bidirectional mainland operation.
        "allow_slack": True,
        "include_objective_constant": False,
        # status/condition intentionally omitted here -- this scenario was
        # solved and saved before network.meta solver-status logging was
        # introduced. validate_scenario() will correctly report
        # "NOT_VERIFIED" rather than silently treating this as confirmed.
    }

    report = validate_scenario(
        network=solved_network,
        baseline_network=baseline_network,
        scenario=scenario,
        tolerance_mw=1e-4,
    )

    pf_result = validate_critical_hours_pf(SOLVED_FILE)

    # ── Three-tier final classification (see validate_scenario.py for full logic) ──
    optimisation_result_pass = report["overall_pass"]
    solver_status_state = report.get("solver_status_state", "NOT_VERIFIED")
    dtype_state = report.get("dtype_consistency", "unresolved")
    pf_convergence_voltage_pass = pf_result["pass"]
    pf_slack_review_needed = pf_result.get("slack_adjustment_review_needed", False)
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
        or slack_direction_review_needed
    )

    if hard_fail:
        final_classification = "FAIL"
    elif needs_review:
        final_classification = "CONDITIONAL PASS"
    else:
        final_classification = "PASS"

    print("\n" + "=" * 100)
    print("FINAL CLASSIFICATION for R01")
    print("=" * 100)
    print(f"  Optimisation-result validation:         {'PASS' if optimisation_result_pass else 'FAIL'}")
    print(f"  Critical-hour PF convergence/voltage:   {'PASS' if pf_convergence_voltage_pass else 'FAIL'}")
    print(f"  Solver-status verification:             "
          f"{'VERIFIED PASS' if solver_status_state == 'VERIFIED_PASS' else solver_status_state}")
    print(f"  Dtype consistency:                      {dtype_state.upper()}")
    print(f"  PF Slack adjustment:                    "
          f"{'REQUIRES REVIEW' if pf_slack_review_needed else 'WITHIN TOLERANCE'}")
    print(f"  Slack direction (negative dispatch):     "
          f"{'REQUIRES REVIEW' if slack_direction_review_needed else 'OK / NOT OBSERVED'}")
    print(f"\n  Overall result: {final_classification}")
    print("=" * 100)

    report["critical_hours_pf"] = pf_result
    report["final_classification"] = final_classification
    return report


if __name__ == "__main__":
    main()