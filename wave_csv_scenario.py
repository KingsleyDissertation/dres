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

# Wind power = p_max_pu * p_nom
wind_p = pd.DataFrame({
    g: network.generators_t.p_max_pu[g] * network.generators.loc[g, "p_nom"]
    for g in wind_gens
}).sum(axis=1)

# Wave power - handles both p_set and p_max_pu cases
if len(wave_gens) > 0 and wave_gens[0] in network.generators_t.p_max_pu.columns:
    wave_p = pd.DataFrame({
        g: network.generators_t.p_max_pu[g] * network.generators.loc[g, "p_nom"]
        for g in wave_gens
    }).sum(axis=1)
else:
    wave_p = network.generators_t.p[wave_gens].sum(axis=1)

# Tidal power - handles both p_set and p_max_pu cases
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


# ── 2. Print summary statistics ────────────────────────────────────────────
print("=== ORKNEY ENERGY BALANCE - CORPOWER 5MW WAVE + REAL TIDAL (2019) ===")
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
print(f"Wave capacity factor       : {wave_p.mean()/7:.3f}")

# ── 3. Plot ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# 3a. Wind generation
axes[0].fill_between(wind_p.index, wind_p.values, alpha=0.6, color='steelblue', label="Wind")
axes[0].plot(total_demand.index, total_demand.values, 'k--', label="Demand", linewidth=1)
axes[0].set_ylabel("Power (MW)")
axes[0].set_title("Orkney Baseline: Wind Generation vs Demand (2019)")
axes[0].legend()

# 3b. Wave and tidal on their own scale
axes[1].fill_between(wave_p.index,  wave_p.values,  alpha=0.7, color='orange', label="Wave (CorPower 5MW)")
axes[1].fill_between(tidal_p.index, tidal_p.values, alpha=0.5, color='green', label="Tidal (Westray-South 7.2MW)")
axes[1].set_title("Wave and Tidal Generation (own scale)")
axes[1].legend()

# 3c. Net balance
axes[2].fill_between(net_balance.index, net_balance.values, 0,
                     where=net_balance > 0, alpha=0.5, color='green', label="Excess")
axes[2].fill_between(net_balance.index, net_balance.values, 0,
                     where=net_balance < 0, alpha=0.5, color='red',   label="Shortfall")
axes[2].axhline(0, color='black', linewidth=0.8)
axes[2].set_ylabel("Power (MW)")
axes[2].set_title("Net Balance (Renewable - Demand)")
axes[2].legend()

# 3d. Monthly summary
monthly_excess    = net_balance.clip(lower=0).resample('ME').sum()
monthly_shortfall = net_balance.clip(upper=0).resample('ME').sum()
axes[3].bar(monthly_excess.index,    monthly_excess.values,    width=20, color='green', alpha=0.6, label="Excess")
axes[3].bar(monthly_shortfall.index, monthly_shortfall.values, width=20, color='red',   alpha=0.6, label="Shortfall")
axes[3].set_ylabel("Energy (MWh)")
axes[3].set_title("Monthly Excess / Shortfall")
axes[3].legend()

plt.tight_layout()
plt.savefig("DATA/outputs/scenario_corpower5MW.png", dpi=150)
plt.show()
print("Plot saved to DATA/outputs/baseline_flat5MW.png")