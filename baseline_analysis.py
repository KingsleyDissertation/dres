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

wave_p  = network.generators_t.p[wave_gens].sum(axis=1)
tidal_p = network.generators_t.p[tidal_gens].sum(axis=1)

total_renewable = wind_p + wave_p + tidal_p
total_demand    = network.loads_t.p_set.sum(axis=1)
net_balance     = total_renewable - total_demand

# ── 2. Print summary statistics ────────────────────────────────────────────
print("=== BASELINE ORKNEY ENERGY BALANCE (2019) ===")
print(f"Total renewable generation : {total_renewable.sum():.1f} MWh")
print(f"Total demand               : {total_demand.sum():.1f} MWh")
print(f"Annual net balance         : {net_balance.sum():.1f} MWh")
print(f"Hours of shortfall         : {(net_balance < 0).sum()}")
print(f"Hours of excess            : {(net_balance > 0).sum()}")
print(f"Max excess power           : {net_balance.max():.1f} MW")
print(f"Max shortfall power        : {net_balance.min():.1f} MW")

# ── 3. Plot ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# 3a. Generation stack
axes[0].plot(wind_p.index,  wind_p.values,  label="Wind",  alpha=0.8)
axes[0].plot(wave_p.index,  wave_p.values,  label="Wave",  alpha=0.8)
axes[0].plot(tidal_p.index, tidal_p.values, label="Tidal", alpha=0.8)
axes[0].plot(total_demand.index, total_demand.values, 'k--', label="Demand", linewidth=1)
axes[0].set_ylabel("Power (MW)")
axes[0].set_title("Orkney Baseline: Generation vs Demand (2019)")
axes[0].legend()

# 3b. Net balance
axes[1].fill_between(net_balance.index, net_balance.values, 0,
                     where=net_balance > 0, alpha=0.5, color='green', label="Excess")
axes[1].fill_between(net_balance.index, net_balance.values, 0,
                     where=net_balance < 0, alpha=0.5, color='red',   label="Shortfall")
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].set_ylabel("Power (MW)")
axes[1].set_title("Net Balance (Renewable - Demand)")
axes[1].legend()

# 3c. Monthly summary
monthly_excess    = net_balance.clip(lower=0).resample('ME').sum()
monthly_shortfall = net_balance.clip(upper=0).resample('ME').sum()
axes[2].bar(monthly_excess.index,    monthly_excess.values,    width=20, color='green', alpha=0.6, label="Excess")
axes[2].bar(monthly_shortfall.index, monthly_shortfall.values, width=20, color='red',   alpha=0.6, label="Shortfall")
axes[2].set_ylabel("Energy (MWh)")
axes[2].set_title("Monthly Excess / Shortfall")
axes[2].legend()

plt.tight_layout()
plt.savefig("DATA/outputs/baseline_balance.png", dpi=150)
plt.show()
print("Plot saved to DATA/outputs/baseline_balance.png")