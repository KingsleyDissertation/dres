import pypsa
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the network
network = pypsa.Network()
network.import_from_netcdf("DATA/outputs/network.nc")

# ── 1. Generation time series ──────────────────────────────────────────────
slack = ["Slack_generator"]
non_slack = [g for g in network.generators.index if g not in slack]

wind_gens  = [g for g in non_slack if "WindFarm" in g]
wave_gens  = [g for g in non_slack if "Wave" in g]
tidal_gens = [g for g in non_slack if "Tidal" in g]

# Wind power = p_max_pu * p_nom for each turbine
wind_p = pd.DataFrame({
    g: network.generators_t.p_max_pu[g] * network.generators.loc[g, "p_nom"]
    for g in wind_gens
}).sum(axis=1)

# Wave and tidal - handles both p_set and p_max_pu cases
if len(wave_gens) > 0 and wave_gens[0] in network.generators_t.p_max_pu.columns:
    wave_p = pd.DataFrame({
        g: network.generators_t.p_max_pu[g] * network.generators.loc[g, "p_nom"]
        for g in wave_gens
    }).sum(axis=1)
else:
    wave_p = network.generators_t.p[wave_gens].sum(axis=1)

if len(tidal_gens) > 0 and tidal_gens[0] in network.generators_t.p_max_pu.columns:
    tidal_p = pd.DataFrame({
        g: network.generators_t.p_max_pu[g] * network.generators.loc[g, "p_nom"]
        for g in tidal_gens
    }).sum(axis=1)
else:
    tidal_p = network.generators_t.p[tidal_gens].sum(axis=1)

total_renewable = wind_p + wave_p + tidal_p
total_demand    = network.loads_t.p_set.sum(axis=1)
net_balance     = total_renewable - total_demand
monthly_excess    = net_balance.clip(lower=0).resample('ME').sum()
monthly_shortfall = net_balance.clip(upper=0).resample('ME').sum()

# ── 2. Print summary statistics ────────────────────────────────────────────
print("=== BASELINE ORKNEY ENERGY BALANCE (2019) ===")
print(f"Total wind generation      : {wind_p.sum():.1f} MWh")
print(f"Total wave generation      : {wave_p.sum():.1f} MWh")
print(f"Total tidal generation     : {tidal_p.sum():.1f} MWh")
print(f"Total renewable generation : {total_renewable.sum():.1f} MWh")
print(f"Total demand               : {total_demand.sum():.1f} MWh")
print(f"Annual net balance         : {net_balance.sum():.1f} MWh")
print(f"Hours of shortfall         : {(net_balance < 0).sum()}")
print(f"Hours of excess            : {(net_balance > 0).sum()}")
print(f"Max excess power           : {net_balance.max():.1f} MW")
print(f"Max shortfall power        : {net_balance.min():.1f} MW")

# ── 3. Export to Excel ─────────────────────────────────────────────────────
excel_path = "DATA/outputs/baseline_balance.xlsx"
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

    # Sheet 1 — Hourly time series
    hourly_df = pd.DataFrame({
        'Wind (MW)':            wind_p,
        'Wave (MW)':            wave_p,
        'Tidal (MW)':           tidal_p,
        'Total Renewable (MW)': total_renewable,
        'Demand (MW)':          total_demand,
        'Net Balance (MW)':     net_balance,
    })
    hourly_df.to_excel(writer, sheet_name='Hourly Time Series')

    # Sheet 2 — Monthly summary
    monthly_df = pd.DataFrame({
        'Month':             monthly_excess.index.strftime('%Y-%m'),
        'Excess (MWh)':      monthly_excess.values,
        'Shortfall (MWh)':   monthly_shortfall.values,
        'Net (MWh)':         monthly_excess.values + monthly_shortfall.values,
    })
    monthly_df.to_excel(writer, sheet_name='Monthly Summary', index=False)

    # Sheet 3 — Summary statistics
    summary_df = pd.DataFrame({
        'Metric': [
            'Total wind generation (MWh)',
            'Total wave generation (MWh)',
            'Total tidal generation (MWh)',
            'Total renewable generation (MWh)',
            'Total demand (MWh)',
            'Annual net balance (MWh)',
            'Hours of shortfall',
            'Hours of excess',
            'Max excess power (MW)',
            'Max shortfall power (MW)',
        ],
        'Value': [
            round(wind_p.sum(), 1),
            round(wave_p.sum(), 1),
            round(tidal_p.sum(), 1),
            round(total_renewable.sum(), 1),
            round(total_demand.sum(), 1),
            round(net_balance.sum(), 1),
            int((net_balance < 0).sum()),
            int((net_balance > 0).sum()),
            round(net_balance.max(), 1),
            round(net_balance.min(), 1),
        ]
    })
    summary_df.to_excel(writer, sheet_name='Summary Statistics', index=False)

print(f"Excel saved to {excel_path}")

# ── 4. Plot ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

axes[0].plot(wind_p.index,  wind_p.values,  label="Wind",  alpha=0.8)
axes[0].plot(wave_p.index,  wave_p.values,  label="Wave",  alpha=0.8)
axes[0].plot(tidal_p.index, tidal_p.values, label="Tidal", alpha=0.8)
axes[0].plot(total_demand.index, total_demand.values, 'k--', label="Demand", linewidth=1)
axes[0].set_ylabel("Power (MW)")
axes[0].set_title("Orkney Baseline: Generation vs Demand (2019)")
axes[0].legend()

axes[1].fill_between(net_balance.index, net_balance.values, 0,
                     where=net_balance > 0, alpha=0.5, color='green', label="Excess")
axes[1].fill_between(net_balance.index, net_balance.values, 0,
                     where=net_balance < 0, alpha=0.5, color='red',   label="Shortfall")
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].set_ylabel("Power (MW)")
axes[1].set_title("Net Balance (Renewable - Demand)")
axes[1].legend()

axes[2].bar(monthly_excess.index,    monthly_excess.values,    width=20, color='green', alpha=0.6, label="Excess")
axes[2].bar(monthly_shortfall.index, monthly_shortfall.values, width=20, color='red',   alpha=0.6, label="Shortfall")
axes[2].set_ylabel("Energy (MWh)")
axes[2].set_title("Monthly Excess / Shortfall")
axes[2].legend()

plt.tight_layout()
plt.savefig("DATA/outputs/baseline_balance.png", dpi=150)
plt.show()
print("Plot saved to DATA/outputs/baseline_balance.png")