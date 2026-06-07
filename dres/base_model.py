#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
`base_model.py`
A packaged power network model (PyPSA-based) configured to run using generic data input (avoiding hard-coded
model context). This has been developed specifically to run as a containerised Docker model, which can be run 
locally or on the online Data Analytics Facility for National Infrastructure (DAFNI) platform.
"""


###############################################################################################################
# Standard Python Libraries
import pandas as pd
import numpy as np
import pypsa
import socket
import os
import json
from datetime import datetime, date

# Custom Libraries
from .dafni_utilities import performance, message_api
from .nemo import define_assets
from dres import weather
from .ev_optimise import simulate_evs


###############################################################################################################


def create_pypsa_network():
    t0 = performance()

    # Create a new PyPSA network
    network = pypsa.Network()
    
    performance(t0)
    return network


def create_bus_network(network, sim):
    t0 = performance()

    network.add("Carrier", "AC")

    # Read CSV file for bus input
    file_path = os.path.join(sim.paths.inputs,"Bus_Input.csv")
    buses = pd.read_csv(file_path)

    # Input bus data from the DataFrame
    if sim.params.legacy:
        for index, row in buses.iterrows():
            network.add(
                "Bus",
                name=row["Name"],
                # Ensure column name matches the DataFrame
                v_nom=row["v _nom(kv)"],
                control=row["Control"],
                carrier="AC"
            )
    else:
        for index, row in buses.iterrows():
            network.add(
                "Bus",
                name=row["Name"],
                # Ensure column name matches the DataFrame
                v_nom=row["v _nom(kv)"],
                control=row["Control"],
                x=row["x"],
                y=row["y"],
                carrier="AC"
            )
    
    performance(t0)


def create_transmission_network(network, sim):
    t0 = performance()

    # Read CSV file for Transmission Line input
    filename = 'Transmission_input.csv'
    full_filename = os.path.join(sim.paths.inputs, filename)
    df = pd.read_csv(full_filename)

    if sim.params.legacy:
        for index, row in df.iterrows():
            network.add(
                "Line",
                name=row["Name"],
                bus0=row["bus0 name"],
                bus1=row["bus1 name"],
                x=row["Reactance[ohm]"],
                r=row["Resistance[ohm]"],
                b=row["Susceptance[ohm^-1]"],
                s_nom=50.0,
                carrier="AC",
                capital_cost=1000
            )
    else:
        for index, row in df.iterrows():
            network.add(
                "Line",
                name=row["Name"],
                bus0=row["bus0 name"],
                bus1=row["bus1 name"],
                x=row["Reactance[ohm]"],
                r=row["Resistance[ohm]"],
                b=row["Susceptance[ohm^-1]"],
                length=row["length[km]"],
                s_nom=50.0,
                carrier="AC",
                capital_cost=1000
            )

    performance(t0)


def add_loads_to_network(network, sim):
    t0 = performance()

    # Load the data from the CSV file for load profiles
    file_path_load = os.path.join(sim.paths.inputs,"Load_Profile.csv")
    df = pd.read_csv(file_path_load)
    df.DateTime = pd.to_datetime(df.DateTime)

    # Set the 'DateTime' column as the index
    df.set_index("DateTime", inplace=True)

    # Select specific date range from data (using `start_date` and `end_date`)
    df = df.loc[sim.params.start_date:sim.params.end_date]

    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv  ENVIRONMENT VARIABLE (int:17)  vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # Split the DataFrame into active and reactive power DataFrames
    active_power_df = df.iloc[:, :17]*sim.params.rescale_load  # First 17 columns are active power
    reactive_power_df = df.iloc[:, 17:]  # Next 17 columns are reactive power
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  ENVIRONMENT VARIABLE (int:17)  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    # Remove units from column names
    active_power_df.columns = active_power_df.columns.str.replace(
        r" \[MW\]", "", regex=True
    )
    reactive_power_df.columns = reactive_power_df.columns.str.replace(
        r" \[MVar\]", "", regex=True
    )

    # Set the snapshots (timestamps)
    network.set_snapshots(df.index)

    # Add loads to the network
    for bus in active_power_df.columns:
        if bus in network.buses.index:
            network.add(
                "Load",
                name=f"{bus}_load",
                bus=bus,
                p_set=active_power_df[bus],  # Assigning directly
                carrier="AC",
                q_set=reactive_power_df[bus] if bus in reactive_power_df.columns else 0,
            )  # Assigning directly if available

    # Assign the active power load profile
    for bus in active_power_df.columns:
        if f"{bus}_load" in network.loads.index:
            network.loads_t.p_set[f"{bus}_load"] = active_power_df[bus]
        # else:
            # message_api(f"Bus {bus} not found in the network")

    # Assign the reactive power load profile
    for bus in reactive_power_df.columns:
        if f"{bus}_load" in network.loads.index:
            network.loads_t.q_set[f"{bus}_load"] = reactive_power_df[bus]
        # else:
            # message_api(f"Bus {bus} not found in the network")

    performance(t0)


def add_wind_turbines_to_network(network, sim, weather_state, storm_wind_speeds):
    t0 = performance()

    # Wind Turbine data handling
    # Generate datetime index for the wind turbine data
    
    # Loop through each wind turbine data file and add to network
    for wind_farm in sim.assets['WindFarms']:

        # Read wind turbine data
        fullfilename = os.path.join(
            sim.paths.inputs, sim.assets['WindFarms'][wind_farm]['file'])
        wind_turbine_data = pd.read_csv(fullfilename)

        # Set the 'DateTime' column as the index
        wind_turbine_data.DateTime = pd.to_datetime(wind_turbine_data.DateTime)
        wind_turbine_data.set_index("DateTime", inplace=True)

        # Select specific date range from data (using `start_date` and `end_date`)
        wind_turbine_data = wind_turbine_data.loc[sim.params.start_date:sim.params.end_date]

        
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # NOTES TBC - KQ HELP NEEDED
        # Eval max power 
        max_power = wind_turbine_data.iloc[:, 0].max()
        wind_turbine_p_max_pu = wind_turbine_data.iloc[:, 0] / max_power
        pf = 0.95  # 设定功率因数
        tan_phi = np.tan(np.arccos(pf))  # 计算 tan(θ)

        # 读取 CSV 文件中的无功功率时间序列
        q_series = wind_turbine_data.iloc[:, 1]

        # 计算基于 q_set 的 ±20% 变化范围
        q_max_series = q_series * 1.2
        q_min_series = q_series * 0.8

        # 计算基于 P_max 的最大可允许 Q
        q_max_pu_limit = wind_turbine_p_max_pu * tan_phi
        q_min_pu_limit = -q_max_pu_limit  # 对称限制

        # 取两个约束的交集，确保不会违反 P-Q 关系
        q_max_final = np.minimum(q_max_series, q_max_pu_limit * max_power)  # 不能超过 P 限制
        q_min_final = np.maximum(q_min_series, q_min_pu_limit * max_power)  # 不能超过 P 限制
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # Define wind turbine name (include conditional for legacy naming)
        windfarm_name = f"{wind_farm.replace('WindTurbine','WF')}_({sim.assets['WindFarms'][wind_farm]['name']})"


        # Loop to add the specified number of turbines for each bus
        for j in range(sim.assets['WindFarms'][wind_farm]['n_turbines']):
            turbine_name = f"{windfarm_name}_Turbine_{j+1}"

            if 'alt_name' in sim.assets['WindFarms'][wind_farm]:
                if sim.assets['WindFarms'][wind_farm]['alt_name'] is not None:
                    turbine_name = sim.assets['WindFarms'][wind_farm]['alt_name']

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Storm conditions
            # if weather_state == "extreme":
            #     # Use wind speed to adjust power output
            #     adjusted_power_output = []
            #     for wind_speed in storm_wind_speeds:
            #         if wind_speed > 25:  # Over cut wind speed, stop
            #             adjusted_power_output.append(0)
            #         else:
            #             # Power output curve adjustment based on wind speed (assuming active power is proportional to the cubic of wind speed)
            #             adjusted_power_output.append(min(
            #                 wind_speed ** 3/1000, wind_turbine_data.iloc[:, 0].max()))  # Rated power limit
            # else:
            #     adjusted_power_output = wind_turbine_data.iloc[:, 0].to_list()
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Storm conditions

            # Add the wind turbine generator to the network
            network.add(
                "Generator",
                name=turbine_name,
                bus=sim.assets['WindFarms'][wind_farm]['bus'],
                control="PQ",
                p_nom=max_power,
                carrier="AC",
                marginal_cost=30
            )

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # NOTES TBC - KQ HELP NEEDED
            network.generators_t.p_max_pu[turbine_name] = wind_turbine_p_max_pu
            if "q_max" not in network.generators.columns:
                network.generators["q_max"] = np.nan
            if "q_min" not in network.generators.columns:
                network.generators["q_min"] = np.nan
            network.generators.loc[turbine_name, "q_max"] = q_max_final.max()  # 取时间序列中的最大值
            network.generators.loc[turbine_name, "q_min"] = q_min_final.min()  # 取时间序列中的最小值
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


            # Assign the time-variant power generation data to the wind turbine generator
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Storm conditions
            # network.generators_t.p_set[turbine_name] = [output * 0.4 for output in adjusted_power_output]  # Active power scaled by 0.4
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Storm conditions

            # # Check if the second column for reactive power exists
            # if wind_turbine_data.shape[1] > 1:
            #     network.generators_t.q_set[turbine_name] = wind_turbine_data.iloc[
            #         :, 1
            #     ]  # Reactive power
            # else:
            #     network.generators_t.q_set[turbine_name] = (
            #         0  # Set reactive power to 0 if column is missing
            #     )

    performance(t0)


def add_generators_to_network(network, assets):
    t0 = performance()

    # Add all (remaining) generators
    define_assets(network, assets, asset_class="Generators")

    performance(t0)


def add_storage_to_network(network, sim):
    t0 = performance()

    # Add all storage units
    define_assets(network, sim.assets, asset_class="StorageUnits")

    performance(t0)


def define_transformers(network, sim):
    t0 = performance()

    # message_api('Read the CSV file')
    # Read the CSV file
    file_path = (
        # message_api('Update this with the correct file path')
        # Update this with the correct file path
        os.path.join(sim.paths.inputs,"Transformer_types_definition.csv")
    )
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.replace('\r\n', '\n')

    # message_api('Define the transformer parameters based on your provided data')
    # Define the transformer parameters based on your provided data
    for index, row in df.iterrows():
        transformer_data = {
            "name": row["Name"],
            "f_nom": row["f_nom\n[HZ] "],
            "s_nom": row["s_nom\n[MVA]"],
            "v_nom_0": row["v_nom_0\n[KV]"],
            "v_nom_1": row["v_nom_1\n[KV]"],
            "vsc": row["vsc\n[%]"],
            "vscr": row["vscr\n[%]"],
            "pfe": row["pfe\n[Kw]"],
            "i0": row["i0\n[%]"],
            "phase_shift": row["phase_shift\n[degree]"],
            "tap_side": row["tap_side\n[0 or 1]"],
            "tap_neutral": row["tap_neutral\n"],
            "tap_min": row["tap_min"],
            "tap_max": row["tap_max"],
            "tap_step": row["tap_step\n[%]"],
        }

        # message_api('Add transformer type to the network')
        # Add transformer type to the network
        network.transformer_types.loc[transformer_data["name"]
                                      ] = transformer_data

    # Read CSV file
    file_path = os.path.join(sim.paths.inputs,"Transformer_input_with_type_definition.csv")
    buses = pd.read_csv(file_path)

    # message_api('Input bus data from CSV file "Bus Input Final(Orkney)"')
    # Input bus data from CSV file "Bus Input Final(Orkney)"
    for index, row in buses.iterrows():
        network.add(
            "Transformer",
            type=row["Type"],
            name=row["Name"],
            bus0=row["bus0"],
            bus1=row["bus1"],
            x=row["x[pu]"],  # S_base = S_NOM
            r=row["r[pu]"],  # S_base = S_NOM
            s_nom=row["s_nom[MVA]"],
            carrier="AC",
            capital_cost=5000
        )  # unit = MVA

    
    performance(t0)


def add_shunts_to_network(network, sim):
    t0 = performance()

    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv  HARD-CODED CONTEXT (TBC)  vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv  ??YAML??
    # Define constant reactive power values for shunt capacitors and reactors
    shunt_reactor_86109 = []
    # shunt_reactor_86163 = []
    # for t in network.snapshots:
    #     if network.generators_t.p_set.at[t, "WindFarm_3_(None)_Turbine_1"] > 0:
    #         shunt_reactor_86109.append(4.0)
    #         shunt_reactor_86163.append(6.0)
    #     else:
    #         shunt_reactor_86109.append(-1.0)
    #         shunt_reactor_86163.append(4.0)
    shunt_reactor_86109 = 8.0  # Negative value for inductive reactive power
    shunt_reactor_86144 = 1.6  # Negative value for inductive reactive power
    shunt_reactor_86143 = 0.4  # Negative value for inductive reactive power

    # Create time series data for the reactive power of shunt reactors
    # constant_reactive_power_86109 = pd.Series(
    #     shunt_reactor_86109, index=network.snapshots)
    # constant_reactive_power_86144 = pd.Series(
    #     shunt_reactor_86144, index=network.snapshots)
    # constant_reactive_power_86143 = pd.Series(
    #     shunt_reactor_86143, index=network.snapshots)
    # constant_reactive_power_86163 = pd.Series(shunt_reactor_86163, index=network.snapshots)

    # Add the shunt reactors as loads consuming reactive power
    network.add(
        "Load",
        name="Shunt_Reactor_86109",
        bus="BURGAR1A",
        carrier="AC",
        p_nom=0,
        q_nom=shunt_reactor_86109
    )

    network.add(
        "Load",
        name="Shunt_Reactor_86144",
        bus="STRONS3B",
        carrier="AC",
        p_nom=0,
        q_nom=shunt_reactor_86144
    )

    network.add(
        "Load",
        name="Shunt_Reactor_86143",
        bus="SANDAY3B",
        carrier="AC",
        p_nom=0,
        q_nom=shunt_reactor_86143
    )
    
    # network.add("Load",
    #     name="Shunt_Reactor_86163",
    #     bus="NEWBIG1B",
    #     p_set=pd.Series(0.0, index=network.snapshots),  # No active power consumption
    #     q_set=constant_reactive_power_86163  # Reactive power consumption time series
    # )
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  HARD-CODED CONTEXT (TBC)  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # This may include shunt reactors
    # define_assets(network, assets, asset_class="Loads")

    performance(t0)


def add_offshore_marine_to_network(network, sim):
    t0 = performance()

    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv  MODIFIED TIDAL SECTION  vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv  ??YAML??
    # Add constant tidal generators
    # Load real tidal time series (Westray-South, Pennock et al.)
    tidal_file = os.path.join(sim.paths.inputs, "tidal_orkney_2019.csv")
    tidal_cf = pd.read_csv(tidal_file)
    tidal_cf['time'] = pd.to_datetime(tidal_cf['time'])
    tidal_cf.set_index('time', inplace=True)
    tidal_cf = tidal_cf.loc[sim.params.start_date:sim.params.end_date]
    tidal_p_max_pu = tidal_cf['power (pu)'].clip(0, 1)

    # 7.2 MW — first two CfD projects (Orbital Marine Eday 1 + 2)
    EDAY_tidal_power = 7.2  # MW

    network.add(
        "Generator",
        name="EDAY Tidal Generator",
        bus="NEWBIG1A",
        p_nom=EDAY_tidal_power,
        carrier="AC",
        marginal_cost=30
    )
    network.generators_t.p_max_pu["EDAY Tidal Generator"] = tidal_p_max_pu

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  MODIFIED WAVE SECTION  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   # Add wave generator with real CorPower hourly time series
    wave_file = os.path.join(sim.paths.inputs, "wave_orkney_2019.csv")
    wave_cf = pd.read_csv(wave_file)
    wave_cf['time'] = pd.to_datetime(wave_cf['time'])
    wave_cf.set_index('time', inplace=True)
    wave_cf = wave_cf.loc[sim.params.start_date:sim.params.end_date]
    wave_p_max_pu = wave_cf['power (pu)'].clip(0, 1)

    EMEC_wave_power = 5  # MW, CorPower Phase 1 planned capacity

    network.add(
        "Generator",
        name="Wave Generator",
        bus="STROMN3A",
        p_nom=EMEC_wave_power,
        carrier="AC",
        marginal_cost=35
    )
    network.generators_t.p_max_pu["Wave Generator"] = wave_p_max_pu

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  HARD-CODED CONTEXT (TBC)  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    performance(t0)


def add_control(network, sim):
    t0 = performance()
    if sim.params.legacy:
        network.generators.loc["Slack_generator", "control"] = "Slack"
    else:
        None

    network.buses["v_mag_pu_min"] = 0.90 
    network.buses["v_mag_pu_max"] = 1.10 
    network.generators.control = "PV"


    performance(t0)


def add_enhanced_storage_to_network(network):
    t0 = performance()
    

    if "KIRKWA3A_Store_bus" not in network.buses.index:
        network.add("Bus", 
                    name="KIRKWA3A_Store_bus",
                    v_nom=33,  
                    carrier="AC")

    p_nom   =4      # MW  
    e_cap   = 18     # MWh 
    eta_in  = 1      
    eta_out = 1     
    soc0    = 0.30     

    network.add("StorageUnit",
                name="Battery_Store",
                bus="KIRKWA3A_Store_bus",
                e_nom=e_cap,
                e_initial=e_cap * soc0,
                e_cyclic=False)      

    network.add("Link",
                name="Battery_Charge",
                bus0="KIRKWA3A",           
                bus1="KIRKWA3A_Store_bus",  
                efficiency=eta_in,
                p_nom=p_nom,
                controllable=False)        

    network.add("Link",
                name="Battery_Discharge",
                bus0="KIRKWA3A_Store_bus",
                bus1="KIRKWA3A",
                efficiency=eta_out,
                p_nom=p_nom,
                controllable=False)
    
    performance(t0)


def run_power_flow(network, OUTPUT_FOLDER):
    message_api(msg="# Run baseline power network model")

    t0 = performance()

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # NOTES TBC - KQ HELP NEEDED
    network.buses["v_mag_pu_min"] = 0.90  # 最低电压 0.95 p.u.
    network.buses["v_mag_pu_max"] = 1.10  # 最高电压 1.05 p.u.
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # Run the power flow
    network.pf()

    # Retrieve power generation data
    generator_p = network.generators_t.p  # Active power generation
    # Reactive power generation (if available)
    generator_q = network.generators_t.q

    # Identify the slack generator(s)
    slack_generators = ["Slack_generator"]  # Adjust the name if it's different

    # Filter out slack generator(s)
    non_slack_generators = [
        gen for gen in network.generators.index if gen not in slack_generators
    ]

    # Calculate total generation excluding slack generator(s)
    total_non_slack_generation = network.generators_t.p[non_slack_generators].sum(
        axis=1)


    # Calculate the losses for each transmission line
    line_losses = network.lines_t.p0 + network.lines_t.p1

    # Calculate the losses for each transmission line
    line_losses = network.lines_t.q0 + network.lines_t.q1

    # Get the voltage magnitude in per unit for each bus and each snapshot
    voltage_magnitudes = network.buses_t.v_mag_pu
    
    


    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> TO BE REMOVED 2025-05-26
    # Save the voltage magnitudes to a CSV file
    output_file_path = f"{OUTPUT_FOLDER}voltage_levels_after_revision.csv"
    voltage_magnitudes.to_csv(output_file_path)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< TO BE REMOVED 2025-05-26



    # Assuming `line_losses` is your DataFrame containing the power losses
    line_losses = network.lines_t.q0 + network.lines_t.q1



    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> TO BE REMOVED 2025-05-26
    # Define the filename
    filename = f"{OUTPUT_FOLDER}reactive_power_losses.csv"

    # Save the DataFrame to a CSV file
    line_losses.to_csv(filename)

    # Assuming `line_losses` is your DataFrame containing the power losses
    line_losses = network.transformers_t.q0 + network.transformers_t.q1

    # Define the filename
    filename = f"{OUTPUT_FOLDER}reactive_power_losses_transformers.csv"

    # Save the DataFrame to a CSV file
    line_losses.to_csv(filename)
    line_losses = network.transformers_t.q0 + network.transformers_t.q1

    network.transformers_t.q1

    network.transformers_t.q0
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< TO BE REMOVED 2025-05-26

    # Identify the slack generator(s)
    slack_generators = ["Slack_generator"]  # Adjust the name if it's different

    # Calculate reactive power load
    total_reactive_load = network.loads_t.q.sum(axis=1)

    # Calculate reactive power generation
    total_reactive_generation = network.generators_t.q.sum(axis=1)

    # Calculate reactive power generation excluding slack generators
    total_reactive_generation_excl_slack = network.generators_t.q[non_slack_generators].sum(
        axis=1
    )

    total_losses = total_reactive_generation - total_reactive_load

    performance(t0)

    return network
