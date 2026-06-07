#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
`dafni_entrypoint.py`
This script runs a packaged power network model (PyPSA-based) configured to run using generic data input
(avoiding hard-coded model context). This has been developed specifically to run as a containerised Docker
model, which can be run locally or on the online Data Analytics Facility for National Infrastructure (DAFNI)
platform.
"""


###############################################################################################################


# %% 
# Standard Python Libraries
import os
import numpy as np
import json
import pandas as pd
import pickle
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Custom Libraries
import dres
import dres.base_model as base_model
import dres.ev_optimise as ev_optimise
from dres.dafni_utilities import performance, message_api, generate_run_metadata
from dres.DRES_Sim import DRES_Sim



###############################################################################################################

# %%
sim = dres.go("simulation-1-spaghetti")


###############################################################################################################

# %% 
# Build baseline power network model
network = sim.build_baseline_model()
network.export_to_netcdf(os.path.join(sim.paths.outputs, "base_net0.nc"))
# base_model.add_enhanced_storage_to_network(network)
base_model.add_control(network, sim)
network.export_to_netcdf(os.path.join(sim.paths.outputs, "base_net.nc"))

###############################################################################################################


# %% 
# Run baseline power network model
network = base_model.run_power_flow(network, sim.paths.outputs)


###########################################################################################################


# %% 
# Optimal EV scheduling (main process)
if sim.params.ev_model == "exclude_evs":
    t1 = performance(t0=sim.t0, msg="FULL MODEL RUN COMPLETE", always_verbose=True)

    scenario_name = f"wave{int(sim.params.wave_capacity)}MW_tidal{sim.params.tidal_capacity}MW"
    network.export_to_netcdf(os.path.join(sim.paths.outputs, f"network_{scenario_name}.nc"))
    network.export_to_netcdf(os.path.join(sim.paths.outputs, "network.nc"))

    generate_run_metadata({
        'machine_id': sim.paths.machine_id,
        'model_version':'v0.3.1',
        'ev_model': sim.params.ev_model,
        'start_date': sim.params.start_date,
        'end_date': sim.params.end_date,
        'compute_time': f"finished in {t1 - sim.t0 :0.4f} seconds"
    }, sim.paths.outputs)


# %% 
if sim.params.ev_model == "include_evs":

    # 4.1 EV Usage and Scheduling Setup
    lb_array, ub_array, delta_soc, vehicle_power, connected_cars = ev_optimise.ev_scheduling_setup(sim.paths.inputs, 'processed_ev_usage.csv')


    
    # 4.3 Running the GWO Optimization
    nc_path = os.path.join(sim.paths.outputs, "base_net.nc")
    gwo = ev_optimise.GWO(
        pop_size = sim.params.pop_size,
        dim = len(lb_array),
        max_iter = sim.params.max_iter,
        lb_array = lb_array,
        ub_array = ub_array,
        initial_soc = sim.params.initial_soc,
        delta_t = sim.params.delta_t,
        capacity = sim.params.capacity,
        min_soc = sim.params.min_soc,
        max_soc = sim.params.max_soc,
        charging_price = sim.params.charging_price,
        discharging_price = sim.params.discharging_price,
        delta_soc = delta_soc,
        nc_path = nc_path,
    )

    pareto_front = gwo.optimize()
    message_api(msg=f"Connected cars per hour:{connected_cars}")
    message_api(msg=f"LB array:{lb_array}")
    message_api(msg=f"UB array:{ub_array}")
    message_api(msg=f"Delta SOC per hour:{delta_soc}")
    message_api(msg=f"Pareto Front Solutions:")


    
    # 4.4 Saving and Plotting Pareto Front Results
    ev_optimise.save_pareto_front(pareto_front, filename=os.path.join(sim.paths.outputs,"pareto_front_extreme_V2G.csv"))


    
    # 4.5 Selecting an Optimal Solution
    optimal_solution = ev_optimise.select_closest_to_ideal(pareto_front)
    message_api(msg=f"Optimal Solution (Voltage Variation, Cost):{optimal_solution}")
    optimal_power_schedule = optimal_solution['position']
    message_api(msg=f"Optimal Power Schedule:{optimal_power_schedule}")

    # Save EV model outputs for visualization
    ev_mod = {
        'pareto_front': pareto_front,
        'optimal_power_schedule': optimal_power_schedule.tolist() if hasattr(optimal_power_schedule, 'tolist') else list(optimal_power_schedule),
        'delta_soc': delta_soc.tolist() if hasattr(delta_soc, 'tolist') else list(delta_soc),
        'gwo_best_scores': gwo.best_scores,
        'connected_cars': connected_cars.tolist() if hasattr(connected_cars, 'tolist') else list(connected_cars),
        'lb_array': lb_array.tolist() if hasattr(lb_array, 'tolist') else list(lb_array),
        'ub_array': ub_array.tolist() if hasattr(ub_array, 'tolist') else list(ub_array),
    }
    with open(os.path.join(sim.paths.outputs, 'ev_model_outputs.pkl'), 'wb') as f:
        pickle.dump(ev_mod, f)


    ##################################

    
    # 5.1 Running Power Flow and Analyzing Results

    # 
    # After assigning all elements, we run a power flow to determine bus voltages, line flows, and losses.
    # 

    if len(network.snapshots) != len(optimal_power_schedule):
        network.set_snapshots(network.snapshots[: len(optimal_power_schedule)])
    network.storage_units_t.p_set.loc[:, "KIRKWA3A_Storage"] = optimal_power_schedule
    network.pf()
    voltage_magnitudes = network.buses_t.v_mag_pu
    message_api(voltage_magnitudes["KIRKWA3A"])


    
    # 5.2 Reviewing Results: Generation, Loads, and Losses

    # 
    # We can inspect generation data, load data, and power losses again after applying the optimal schedule.
    # 

    generator_p = network.generators_t.p
    generator_q = network.generators_t.q
    message_api(msg=f"Active Power Generation (MW):")
    message_api(generator_p)
    message_api(msg=f"Reactive Power Generation (MVar):")
    message_api(generator_q)

    message_api(msg=f"Total Load per Snapshot:")
    message_api(network.loads_t.p.sum(axis=1))

    non_slack_generators = [gen for gen in network.generators.index if gen not in ["Slack_generator"]]
    total_non_slack_generation = network.generators_t.p[non_slack_generators].sum(axis=1)
    message_api(msg=f"Total Generation (Excluding Slack) per Snapshot:")
    message_api(total_non_slack_generation)

    line_losses = network.lines_t.p0 + network.lines_t.p1
    message_api(msg=f"Line Power Losses (MW):")
    message_api(line_losses)


    
    # 5.3 Voltage Profiles and Visualization

    # 
    # Visualize the bus voltage profile over time, or even in 3D to show time and bus indices:
    # 

    bus_voltage = network.buses_t.v_mag_pu
    bus_list = ['KIRKWA3A']
    bus_voltage.to_csv(f'{sim.paths.outputs}bus_voltage_data_extremeonedayV2G.csv')

    """
    plt.figure(figsize=(10,6))
    for bus in bus_list:
        plt.plot(bus_voltage.index, bus_voltage[bus], label=bus)
    plt.title('Bus Voltage Magnitude Over Time')
    plt.xlabel('Time')
    plt.ylabel('Voltage Magnitude (p.u.)')
    plt.legend(title='Buses')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    """


    
    # 5.4 3D Plot of Voltages and voltage distribution box diagram



    # Plot a voltage distribution box diagram
    # Convert voltage data to a long format for use in Seaborn plotting
    bus_voltage_long = bus_voltage.reset_index().melt(id_vars='snapshot', var_name='Bus', value_name='Voltage Magnitude')
    bus_voltage_long.rename(columns={'index': 'Time'}, inplace=True)


    # Plot a voltage distribution box diagram

    bus_voltage = network.buses_t.v_mag_pu

    df = bus_voltage.reset_index()
    message_api(df.columns)

    df.rename(columns={'snapshot': 'index'}, inplace=True)
    bus_voltage_long = df.melt(id_vars='index', var_name='Bus', value_name='Voltage Magnitude')

    # Convert voltage data to a long format for use in Seaborn plotting
    #bus_voltage_long = bus_voltage.reset_index().melt(id_vars='index', var_name='Bus', value_name='Voltage Magnitude')
    bus_voltage_long.rename(columns={'index': 'Time'}, inplace=True)



    
    # 5.5 Power Schedule, Cost and SOC Visualization

    # 
    # Plot the chosen 24-hour charging/discharging schedule and track SOC changes.
    # 


    #24 hour charge and discharge scheduling profits
    car_charging_cost = 0.0
    car_discharging_revenue = 0.0
    for i, power in enumerate(optimal_power_schedule):
        if power < 0:  # charging
            car_charging_cost += abs(power) * sim.params.charging_price[i]
        else:  # discharging
            car_discharging_revenue += abs(power) * sim.params.discharging_price[i]

    message_api(msg=f"Total charging cost :{car_charging_cost}")
    message_api(msg=f"Total discharging revenue :{car_discharging_revenue}")

    # Calculate SOC over time:
    soc_values = [sim.params.initial_soc]
    soc = sim.params.initial_soc
    for t, power in enumerate(optimal_power_schedule):
        soc -= (power*sim.params.delta_t)/sim.params.capacity
        soc += delta_soc[t]
        soc_values.append(soc)


    
    # 5.6 Saving the Optimal Result

    # 
    # We can save the best result found during optimization to JSON for future runs as an initial guess.
    # 

    def save_optimal_result(optimal_result, filename=f"{sim.paths.outputs}best_result_oneday_extreme_V2G.json"):
        with open(filename, 'w') as file:
            json.dump(optimal_result.tolist(), file)
        message_api(f"Optimal result saved to {filename}")

    save_optimal_result(gwo.alpha_pos)


    # ---
    # 
    # # 6. Further Statistics and Plots

    # We can create summary plots for load, generation, and supply-demand differences over the 24-hour period. This helps to visualize the system state and confirm feasibility of solutions.

    
    # 6.1 Hourly Load

    hourly_load = network.loads_t.p.sum(axis=1)


    
    # 6.2 Hourly Generation

    hourly_generation = network.generators_t.p.sum(axis=1)


    
    # 6.3 Renewable Generation (wind, tidal, wave)

    all_generators = network.generators.index
    wind_generators = network.generators[network.generators.carrier == "wind"].index
    tidal_generators = [g for g in all_generators if "Tidal" in g]
    wave_generators = [g for g in all_generators if "Wave" in g]
    renewable_generators = wind_generators.tolist() + tidal_generators + wave_generators

    hourly_renewable = network.generators_t.p[renewable_generators].sum(axis=1)

     
    # 6.4 Supply-Demand Difference

    supply_demand_diff = hourly_generation - hourly_load



     
    # 6.5 Excluding Slack and Storage Generators

    slack_generators = ["Slack_generator"]
    storage_generators = network.generators[network.generators.carrier == 'battery'].index if 'battery' in network.generators.carrier.values else []
    valid_generators = [g for g in network.generators.index if g not in slack_generators and g not in storage_generators]
    hourly_generation_filtered = network.generators_t.p[valid_generators].sum(axis=1)



     
    # 6.6 Calculate Transformer Load Rate
    # 

    transformer_loading = pd.DataFrame(index=network.snapshots, columns=network.transformers.index, data=0.0)

    for trafo_name in network.transformers.index:
        s_nom_trafo = network.transformers.at[trafo_name, "s_nom"] 
        p_flow_trafo = network.transformers_t.p0[trafo_name]
        q_flow_trafo = network.transformers_t.q0[trafo_name]
        flow_trafo_mva = np.sqrt(p_flow_trafo**2 + q_flow_trafo**2)
        transformer_loading[trafo_name] = flow_trafo_mva / s_nom_trafo * 100

    message_api(msg=f"Transformer Load Rate(%)：")
    message_api(transformer_loading)

     
    # 6.7 CO2 emission calculation

    charging_efficiency = 0.99
    discharging_efficiency = 0.99

    # Read the total power generation of the wind turbine during each period(MW)
    wind_generation = network.generators_t.p[wind_generators].sum(axis=1)

    wind_emission_factor = 0.011  # kg/kWh 
    ev_emission_factor = 0.011      # kg/kWh 
    coal_emission_factor = 1.94   # kg/kWh
    natural_gas_emission_factor = 0.437  # kg/kWh

    co2_charging = 0.0         # CO2 emission from EV charging
    co2_discharging = 0.0      # CO2 emission from EV discharging
    co2_coal_replace = 0.0     # CO2 emissions if coal-fired power plants are used instead of discharge (kg)
    co2_naturalgas_replace = 0.0    # CO2 emissions if natural gas plants are used instead of discharge (kg)

    for t, power_mw in enumerate(optimal_power_schedule):
        # Convert MW to kWh, power_mw(MW) * 1000(kWh/MWh)
        power_kwh = abs(power_mw) * 1000.0

        # Judge whether to charge or discharge
        if power_mw < 0:
            # charging
            if wind_generation.iloc[t] > 1e-6:
                # 11 g/kWh => 0.011 kg/kWh
                co2_charging += power_kwh * wind_emission_factor
            else:
                co2_charging += power_kwh * coal_emission_factor
        else:
            # discharging
            co2_discharging += power_kwh * ev_emission_factor * (charging_efficiency * discharging_efficiency)
            co2_coal_replace += power_kwh * coal_emission_factor
            co2_naturalgas_replace += power_kwh * natural_gas_emission_factor

    message_api(msg=f"1) CO2 emissions from EV charging (kg)：{co2_charging}")
    message_api(msg=f"2) CO2 emission of EV discharge process itself (kg)：{co2_discharging}")
    message_api(msg=f"3) CO2 emissions if coal-fired power plants are used instead of discharge (kg)：{co2_coal_replace}")
    message_api(msg=f"4) CO2 emissions if natural gas plants are used instead of discharge (kg):{co2_naturalgas_replace}")

    total_co2 = co2_charging + co2_discharging
    message_api(msg=f"EV CO2 emission:{total_co2}")


     
    # Log performance
    t1 = performance(t0=sim.t0, msg="FULL MODEL RUN COMPLETE", always_verbose=True)

    network.export_to_netcdf(os.path.join(sim.paths.outputs, "network.nc"))

    generate_run_metadata({
        'machine_id': sim.paths.machine_id,
        'model_version':'v0.3.1',
        'ev_model': sim.params.ev_model,
        'start_date': sim.params.start_date,
        'end_date': sim.params.end_date,
        'charging_price': sim.params.charging_price,
        'discharging_price': sim.params.discharging_price,
        'pop_size': sim.params.pop_size,
        'max_iter': sim.params.max_iter,
        'initial_soc': sim.params.initial_soc,
        'delta_t': sim.params.delta_t,
        'capacity': sim.params.capacity,
        'min_soc': sim.params.min_soc,
        'max_soc': sim.params.max_soc,
        'vehicle_power': sim.params.vehicle_power,
        'compute_time': f"finished in {t1 - sim.t0 :0.4f} seconds"
    }, sim.paths.outputs)
