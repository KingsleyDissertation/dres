#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
`ev_optimise.py`
{DESCRIPTION TO FOLLOW}
"""


###############################################################################################################
# Standard Python Libraries
import os
import numpy as np
import pandas as pd
import pypsa
from datetime import datetime
import json
import random
import copy
from joblib import Parallel, delayed, parallel_backend

# Custom Libraries
from .dafni_utilities import performance, message_api

_worker_net = None

###############################################################################################################
# PM

def load_ev_schedule_data(INPUT_FOLDER, filename):
    """
    Function description to follow.
    """
    t0 = performance()


    # Full path to file
    full_filename = os.path.join(INPUT_FOLDER, filename)
    
    # Open csv
    df = pd.read_csv(full_filename)

    # Create column with complete DateTime obj for Arrival and Departure
    try:
        # Spaghetti model
        df['Departure DateTime'] = [pd.to_datetime(f"2019-01-01 {row['Departure Time']}") + pd.Timedelta(days=row.Day-1) for (i,row) in df.iterrows()]
        df['Arrival DateTime'] = [pd.to_datetime(f"2019-01-01 {row['Arrival Time']}") + pd.Timedelta(days=row.Day-1) for (i,row) in df.iterrows()]
    except:
        # Data
        df['Start Date'] = pd.to_datetime(df['Start Date'])
        df['End Date'] = pd.to_datetime(df['End Date'])
    
    performance(t0)
    return df

def ev_scheduling_setup(INPUT_FOLDER, filename):
    """
    Function description to follow.
    """
    t0 = performance()


    message_api(msg="# Full path to file")
    # Full path to file
    full_filename = os.path.join(INPUT_FOLDER, filename)
    
    message_api(msg="# Open csv")
    # Open csv
    df = pd.read_csv(full_filename)

    departure_full_times = pd.to_datetime(df.iloc[:, 7], errors='coerce')
    arrival_full_times = pd.to_datetime(df.iloc[:, 8], errors='coerce')
    message_api(msg="# Some measure of energy consumption")
    column_7 = df.iloc[:, 6].values  # Some measure of energy consumption

    hours = np.arange(24)
    connected_cars = []

    message_api(msg="# For each hour, count how many cars are connected and can participate in V2G.")
    # For each hour, count how many cars are connected and can participate in V2G.
    for t in range(24):
        num_connected = 0
        for i in range(len(departure_full_times)):
            departure_hour = departure_full_times.iloc[i].hour
            arrival_hour = arrival_full_times.iloc[i].hour
            # Car not connected exactly at its departure or arrival hour
            if t == departure_hour or t == arrival_hour:
                continue
            else:
                num_connected += 1
        connected_cars.append(num_connected)

    connected_cars = np.array(connected_cars)

    message_api(msg="# Max charge/discharge power per vehicle (MW), if ENV exists, is this value, otherwise default to 0.011")
    vehicle_power = float(os.getenv("vehicle_power", 0.011))  # Max charge/discharge power per vehicle (MW), if ENV exists, is this value, otherwise default to 0.011
    message_api(msg="# Negative: max charging")
    lb_array = -connected_cars * vehicle_power  # Negative: max charging
    message_api(msg="# Positive: max discharging")
    ub_array = connected_cars * vehicle_power   # Positive: max discharging

    message_api(msg="# Calculate delta_soc to account for SOC changes due to trips")
    # Calculate delta_soc to account for SOC changes due to trips
    delta_soc = np.zeros(24)
    for t in hours:
        soc_change = 0
        for i in range(len(departure_full_times)):
            departure_hour = departure_full_times[i].hour
            arrival_hour = arrival_full_times[i].hour
            soc_consumption = column_7[i]/2/235
            if t == departure_hour:
                soc_change -= soc_consumption
            if t == arrival_hour:
                soc_change -= soc_consumption
        delta_soc[t] = soc_change

    performance(t0)
    return lb_array, ub_array, delta_soc, vehicle_power, connected_cars


###############################################################################################################
# KQ

def simulate_evs(network, INPUT_FOLDER, assets):
    """
    Function description to follow.
    """    
    t0 = performance()


    # Gets a list of all bus bars
    buses = network.buses.index.tolist()
    # Select a specific three bus bars
    selected_buses = ['KIRKWA1A']  # 'KIRKWA3A','KIRKWA1B'
    # Calculate the number of electric vehicles assigned to each bus bar
    total_ev = 235  # 235， 5625，11250，22500
    num_buses = len(selected_buses)
    # The number of electric vehicles assigned to each bus
    ev_per_bus = total_ev // num_buses
    # Residual electric vehicles (after equal distribution)
    remaining_ev = total_ev % num_buses

    # Define the parameters of an electric vehicle (assuming every electric vehicle is the same)
    ev_capacity = 0.075  # Battery capacity per electric vehicle (MWh)0.075
    # The maximum charging power (MW) of each EV is 0.022
    ev_charge_power = 0.007
    efficiency_store = 0.9  # Charging efficiency
    efficiency_dispatch = 0.9  # Discharge efficiency
    min_soc = 0.1  # Minimum SOC
    max_soc = 0.9  # Maximum SOC

    # Create a DataFrame to store the time series charge-discharge behavior of all vehicles

    p_set = pd.DataFrame(index=network.snapshots)
    q_set = pd.DataFrame(index=network.snapshots)
    # Set reactive power by power factor
    pf = 0.95  # Set power factor

    # Set charging period and discharging period
    charge_period_1_start = datetime.strptime(
        '1/1/2019 01:00', '%m/%d/%Y %H:%M')
    charge_period_1_end = datetime.strptime(
        '1/1/2019 10:00', '%m/%d/%Y %H:%M')  # 12

    charge_period_2_start = datetime.strptime(
        '1/2/2019 12:00', '%m/%d/%Y %H:%M')
    charge_period_2_end = datetime.strptime('1/2/2019 23:59', '%m/%d/%Y %H:%M')

    file_path = f"{INPUT_FOLDER}processed_ev_usage.csv"
    df = pd.read_csv(file_path)
    # Gets the data for columns 3, 7, 8 and 9 of the table
    departure_full_times = pd.to_datetime(df.iloc[:, 7], errors='coerce')
    arrival_full_times = pd.to_datetime(df.iloc[:, 8], errors='coerce')
    column_3 = df.iloc[:, 2].values
    column_7 = df.iloc[:, 6].values
    matrixSOC = np.column_stack((column_3, column_7))
    # message_api(departure_full_times)
    # message_api(matrixSOC)
    # Traverse all bus lines and add electric vehicle energy storage units
    for i, bus in enumerate(selected_buses):
        # Calculate the number of electric vehicles on the bus
        # ev_per_bus + (1 if i < remaining_ev else 0)  # 将剩余的电动汽车逐个分配到前面的母线上
        num_ev_for_bus = total_ev

        # Create an energy storage unit for each electric vehicle on the bus
        for j in range(num_ev_for_bus):
            # Create a unique name for each electric vehicle
            ev_name = f"EV_{bus}_{j + 1}"
            # Get the initial SOC
            soc = matrixSOC[j, 0]
            # Add energy storage units to the network to simulate electric vehicles
            network.add("StorageUnit",
                        name=ev_name,
                        bus=bus,
                        control='PQ',  # Operate in PQ mode
                        p_nom=ev_charge_power,  # The maximum charge/discharge power of each electric vehicle
                        # energy_nom=ev_capacity,  # The battery capacity of each electric vehicle
                        efficiency_store=efficiency_store,  # Charging efficiency
                        efficiency_dispatch=efficiency_dispatch,  # Discharge efficiency
                        # The time required for the battery to be fully charged
                        max_hours=ev_capacity / ev_charge_power,
                        state_of_charge_initial=soc,  # Initial charging state
                        cyclic_state_of_charge=True)  # Cyclic charge and discharge state

            # Initialize the charging and discharging time series of the energy storage unit
            departure_time = departure_full_times.iloc[j]
            arrival_time = arrival_full_times.iloc[j]
            charge_discharge_profile = []
            charge_period_1_start = datetime.strptime(
                '1/1/2019 00:00', '%m/%d/%Y %H:%M')
            # charge_period_1_end = datetime.strptime('1/1/2019 10:00', '%m/%d/%Y %H:%M')
            charge_period_1_end = datetime.strptime(
                departure_time.strftime('%m/%d/%Y %H:%M'), '%m/%d/%Y %H:%M')
            charge_period_2_start = datetime.strptime(
                arrival_time.strftime('%m/%d/%Y %H:%M'), '%m/%d/%Y %H:%M')
            # charge_period_2_start = datetime.strptime('1/2/2019 12:00', '%m/%d/%Y %H:%M')
            charge_period_2_end = datetime.strptime(
                '1/2/2019 23:59', '%m/%d/%Y %H:%M')
            # Calculate the charge and discharge behavior for 24 hours and adjust the power per hour according to the SOC
            for t in range(len(network.snapshots)):

                current_time = datetime.strptime(
                    network.snapshots[t], '%m/%d/%Y %H:%M')
                if charge_period_1_start <= current_time < charge_period_1_end:
                    #
                    if soc < max_soc:
                        # The charging power is negative
                        power = -min(ev_charge_power, (max_soc-soc)
                                     * ev_capacity/efficiency_store)
                        # Update SOC (Charge to increase SOC)
                        soc = min(max_soc, soc - power *
                                  efficiency_store / ev_capacity)
                    else:
                        power = 0  #
                elif charge_period_2_start <= current_time < charge_period_2_end:
                    #
                    if current_time == charge_period_2_start:
                        soc = soc-matrixSOC[j, 1]
                    if soc < max_soc:
                        # The charging power is negative
                        power = -min(ev_charge_power, (max_soc-soc)
                                     * ev_capacity/efficiency_store)
                        # Update SOC (Charge to increase SOC)
                        soc = min(max_soc, soc - power *
                                  efficiency_store / ev_capacity)
                    else:
                        power = 0  # full
                else:
                    power = 0  #

                # Add the hourly charge and discharge power to the time series
                charge_discharge_profile.append(power)

            # Save the charge-discharge time series to the DataFrame
            p_set[ev_name] = charge_discharge_profile
            q_set[ev_name] = p_set[ev_name] * np.tan(np.arccos(pf))

        # Set the charging and discharging behavior of the energy storage unit
    network.storage_units_t.p_set = p_set
    network.storage_units_t.q_set = q_set
    # Print result
    # for ev_name in p_set.columns:
        # message_api(f"{ev_name} charge and discharge behavior：")
        # message_api(p_set[ev_name])

    #
    # message_api(network.storage_units_t.p_set)
    performance(t0)


def init_worker(nc_path):
    global _worker_net
    _worker_net = pypsa.Network(nc_path)
    

class GWO:
    def __init__(self, pop_size, dim, max_iter, lb_array, ub_array, initial_soc, delta_t, capacity, min_soc, max_soc,
                 charging_price, discharging_price, delta_soc, nc_path):
        self.pop_size = pop_size
        self.dim = dim
        self.max_iter = max_iter
        self.lb_array = lb_array
        self.ub_array = ub_array
        self.alpha_score = float("inf")
        self.alpha_pos = np.zeros(dim)
        self.beta_score = float("inf")
        self.beta_pos = np.zeros(dim)
        self.delta_score = float("inf")
        self.delta_pos = np.zeros(dim)
        self.initial_soc = initial_soc
        self.delta_t = delta_t
        self.capacity = capacity
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.charging_price = charging_price
        self.discharging_price = discharging_price
        self.best_scores = []  # Add a list to record the best score for each generation
        self.optimal_voltage_variation = None # Added property to record optimal voltage fluctuations
        self.optimal_total_cost = None  # Add property to record the optimal total cost
        self.archive = []  # External file, store the non-dominant solution <-- make sure this line exists
        self.delta_soc = delta_soc  # Additional SOC changes per hour
        self.nc_path = nc_path
        self.network = None  # Do not store the full PyPSA network to keep the object picklable


    def initialize_wolves(self):
        wolves = np.random.uniform(self.lb_array, self.ub_array, (self.pop_size, self.dim))
        # Read the best saved result and use it as the initial position of one of the wolves
        try:
            with open("best_result_oneday_extreme_V2G.json", 'r') as file:
                best_result = json.load(file)
                wolves[0] = np.array(best_result)  # Set the first Wolf to the best result
                print("Loaded best result as initial position for one wolf.")
        except FileNotFoundError:
            print("No previous optimal result found, initializing randomly.")

        for i in range(len(wolves)):
            wolves[i] = self.apply_soc_constraints(wolves[i])
        return wolves

    def _min_hours_needed_rev(self, energy_gap, limit_array, t):
        acc = 0.0
        hours = 0
        for h in range(t, -1, -1):
            acc += limit_array[h] * self.delta_t
            hours += 1
            if abs(acc) >= abs(energy_gap):
                return hours, acc 
        return hours, acc 

    def objective_function(self, power_schedule):
        # Create a deep copy of network
        global _worker_net
        if _worker_net is None:
            _worker_net = pypsa.Network(self.nc_path)
        network_copy = copy.deepcopy(_worker_net)
        network_copy.lines.loc[network_copy.lines['r'] == 0, 'r'] = 1e-6
        total_cost = 0
        # Apply SOC constraints
        for t in range(self.dim):
            if power_schedule[t] < 0:  # Charging cost
                total_cost += abs(power_schedule[t]) * self.charging_price[t]
            else:  # Discharge revenue
                total_cost -= abs(power_schedule[t]) * self.discharging_price[t]
        # ------------------------------------------------------------
        # Align the network to the EV schedule length if it is shorter than the full snapshot set.
        if len(network_copy.snapshots) != len(power_schedule):
            if len(network_copy.snapshots) < len(power_schedule):
                raise ValueError(
                    "EV schedule length is longer than network snapshots. "
                    "Please use a matching schedule or smaller network period."
                )
            network_copy.set_snapshots(network_copy.snapshots[: len(power_schedule)])
        p_ser = pd.Series(power_schedule, index=network_copy.snapshots)

        network_copy.links_t.p_set.loc[:, "Battery_Charge"] = (-p_ser).clip(lower=0)
        network_copy.links_t.p_set.loc[:, "Battery_Discharge"] = p_ser.clip(lower=0)
        network_copy.optimize(solver_name="highs")
        optimal_generator_t_p=network_copy.generators_t.p
        optimal_storage_t_p=network_copy.storage_units_t.p
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        powerflow_network = pypsa.Network()
        powerflow_network.import_from_netcdf(self.nc_path)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        if len(powerflow_network.snapshots) != len(optimal_generator_t_p.index):
            powerflow_network.set_snapshots(powerflow_network.snapshots[: len(optimal_generator_t_p.index)])
        powerflow_network.generators_t.p_set = optimal_generator_t_p
        wind_gens = [g for g in powerflow_network.generators.index
             if "WindTurbine_" in g]         

        pf = 0.95                          
        tan_phi = np.tan(np.arccos(pf))        

        for g in wind_gens:
            p_series = powerflow_network.generators_t.p_set[g]  
            q_series = -p_series * tan_phi*0.6
            powerflow_network.generators.loc[g, "control"] = "PQ"
            powerflow_network.generators_t.q_set[g] = q_series  
        if len(powerflow_network.snapshots) != len(optimal_storage_t_p.index):
            powerflow_network.set_snapshots(powerflow_network.snapshots[: len(optimal_storage_t_p.index)])
        powerflow_network.storage_units_t.p_set = optimal_storage_t_p
        powerflow_network.add("StorageUnit",
                    name="KIRKWA3A_Storage",
                    bus="KIRKWA3A",
                    p_nom=2.585,
                    max_hours=16.45 / 2.585,
                    efficiency_store=0.91,
                    efficiency_dispatch=0.91,
                    controllable=False  
                    )
        powerflow_network.storage_units_t.p_set.loc[:, "KIRKWA3A_Storage"] = power_schedule
    
        # Run the power flow
        powerflow_network.pf()
        v_ref = 1.0
        v_dev = powerflow_network.buses_t.v_mag_pu - v_ref
        v_dev_squared_sum = (v_dev ** 2).sum(axis=1)
        total_v_dev_squared_sum = v_dev_squared_sum.sum()
        return np.array([total_v_dev_squared_sum, total_cost])

    def pareto_sort(self, objectives):
        """
        Pareto sort, which returns the rank of each individual.
        :param objectives: List of individual objectives, such as [[obj1_1, obj2_1], [obj1_2, obj2_2],...]
        :return: A list of levels for each individual
        """
        num_individuals = len(objectives)
        domination_counts = np.zeros(num_individuals, dtype=int)
        dominated_sets = [[] for _ in range(num_individuals)]
        pareto_fronts = [[]]

        for p in range(num_individuals):
            for q in range(num_individuals):
                if self.dominates(objectives[p], objectives[q]):
                    dominated_sets[p].append(q)
                elif self.dominates(objectives[q], objectives[p]):
                    domination_counts[p] += 1
            if domination_counts[p] == 0:
                pareto_fronts[0].append(p)

        i = 0
        while pareto_fronts[i]:
            next_front = []
            for p in pareto_fronts[i]:
                for q in dominated_sets[p]:
                    domination_counts[q] -= 1
                    if domination_counts[q] == 0:
                        next_front.append(q)
            i += 1
            pareto_fronts.append(next_front)
        return pareto_fronts[:-1]

    def dominates(self, ind1, ind2):
        """
        Determine whether ind1 dominates ind2.
        """
        return all(x <= y for x, y in zip(ind1, ind2)) and any(x < y for x, y in zip(ind1, ind2))

    def calculate_crowding_distance(self, objectives):
        num_individuals = len(objectives)
        num_objectives = len(objectives[0])
        distances = np.zeros(num_individuals)
        for i in range(num_objectives):
            objective_values = np.array([obj[i] for obj in objectives])
            sorted_indices = np.argsort(objective_values)
            max_val = objective_values[sorted_indices[-1]]
            min_val = objective_values[sorted_indices[0]]
            distances[sorted_indices[0]] = distances[sorted_indices[-1]] = 1e6
            # Do not set the distance of boundary individuals directly to inf when calculating crowding distance. inf can be replaced by a large finite value:
            if max_val == min_val:
                continue
            for j in range(1, num_individuals - 1):
                distances[sorted_indices[j]] += (
                        (objective_values[sorted_indices[j + 1]] - objective_values[sorted_indices[j - 1]]) / (
                            max_val - min_val)
                )
        return distances
    
    def select_alpha_beta_delta(self, wolves, objectives,pareto_fronts=None):
        fronts = self.pareto_sort(objectives)
        first_front = fronts[0]             
        need = 3 - len(first_front)
        if need > 0 and len(fronts) > 1:
            first_front += fronts[1][:need]
        obj_arr = np.array(objectives)
        sub_obj = obj_arr[first_front]
        distances = self.calculate_crowding_distance(sub_obj)
        sorted_idx = np.argsort(-np.array(distances))
        alpha_idx = first_front[sorted_idx[0]]
        beta_idx  = first_front[sorted_idx[1 if len(sorted_idx)>1 else 0]]
        delta_idx = first_front[sorted_idx[2 if len(sorted_idx)>2 else 0]]

        return wolves[alpha_idx], wolves[beta_idx], wolves[delta_idx]

    #
    def optimize(self):
        wolves = self.initialize_wolves()
        n_jobs = -1
        parallel = Parallel(n_jobs=n_jobs, backend="threading")
        for iter in range(self.max_iter):
            print(f"Generation {iter + 1}/{self.max_iter}")
            objectives = parallel(delayed(self.objective_function)(wolf.copy()) for wolf in wolves)
            pareto_fronts = self.pareto_sort(list(objectives))
            # Update external files
            self.archive = self.update_external_archive(self.archive, wolves, objectives)
            # Select Alpha, Beta, and Delta wolves from the current population
            self.alpha_pos, self.beta_pos, self.delta_pos = self.select_alpha_beta_delta(wolves, objectives,
                                                                                         pareto_fronts)
            # Update population location
            a = 2 - iter * (2 / self.max_iter)
            # Update position of all wolves
            for i in range(self.pop_size):
                for j in range(self.dim):
                    r1 = np.random.rand()
                    r2 = np.random.rand()

                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha_pos[j] - wolves[i][j])
                    X1 = self.alpha_pos[j] - A1 * D_alpha

                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.beta_pos[j] - wolves[i][j])
                    X2 = self.beta_pos[j] - A2 * D_beta

                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.delta_pos[j] - wolves[i][j])
                    X3 = self.delta_pos[j] - A3 * D_delta

                    wolves[i][j] = (X1 + X2 + X3) / 3

                # Apply constraints
                wolves[i] = self.apply_soc_constraints(wolves[i])

        return self.archive

    def apply_soc_constraints(self, power_schedule):
        #
        x = power_schedule.copy()
        soc = self.initial_soc
        final_soc = 0.9
        for t in range(len(x)):
            # Apply dynamic lb and ub
            x[t] = np.clip(x[t], self.lb_array[t], self.ub_array[t])

            soctemporary = soc
            # Add a constraint here that makes the final SOC value equal to the initial SOC value, the constraint is in place first.
            soc_difference = final_soc - soctemporary
            power_gap = soc_difference * self.capacity / self.delta_t
            if power_gap > 0:
                #m = int(power_gap / (vehicle_power * 235)) + 1
                m,energy = self._min_hours_needed_rev(power_gap, (-self.lb_array), t)
                if (t + m) == 23:
                    if not (self.lb_array[t] <=x[t]<=(-energy+power_gap)):
                        x[t]=np.clip(x[t], self.lb_array[t],(-energy+power_gap))
                elif (t + m) > 23:
                    x[t]=self.lb_array[t]
            elif power_gap < 0:
                m,energy= self._min_hours_needed_rev((-power_gap), self.ub_array, t)
                if (t + abs(m)) == 23:
                    if not ( (energy + power_gap) <= x[t] <=self.ub_array[t]):
                        x[t] = np.clip( x[t], (energy + power_gap), self.ub_array[t])
                elif (t + abs(m)) > 23:
                    x[t] = self.ub_array[t] 
            soc -= (x[t] * self.delta_t) / self.capacity

            #
            if soc < self.min_soc and x[t] > 0:
                soc = np.clip(soc, self.min_soc- self.delta_soc[t], self.max_soc)
                x[t] = (soctemporary - soc) * self.capacity / self.delta_t
            elif soc > self.max_soc and x[t] < 0:
                soc = np.clip(soc, self.min_soc- self.delta_soc[t], self.max_soc)
                x[t] = (soctemporary - soc) * self.capacity / self.delta_t  #

            soc += self.delta_soc[t]  # Add SOC changes for vehicle departure or arrival
            if power_gap==0:
                if t==23:
                    x[t]=self.delta_soc[t]* self.capacity / self.delta_t
                    soc = soctemporary
        return x

    def update_external_archive(self, archive, wolves, objectives, max_size=2500):
        # Adds the non-dominated solution of the current population to an external file
        combined_positions = [w.copy() for w in wolves] + [ind['position'] for ind in archive]
        combined_objectives = [ obj.copy()        for obj in objectives ] \
                   + [ ind['objective']   for ind in archive ]
        # Undominated sorting of combined solutions
        pareto_fronts = self.pareto_sort(combined_objectives)
        new_archive = []
        for front in pareto_fronts:
            for idx in front:
                individual = {'position': combined_positions[idx].copy(), 'objective': combined_objectives[idx].copy()}
                new_archive.append(individual)
            if len(new_archive) >= max_size:
                break

        # If the file exceeds the maximum size, truncate it using crowding sort
        if len(new_archive) > max_size:
            distances = self.calculate_crowding_distance([ind['objective'] for ind in new_archive])
            sorted_indices = np.argsort(-np.array(distances))
            new_archive = [new_archive[i] for i in sorted_indices[:max_size]]

        return new_archive




def save_pareto_front(pareto_front, filename="pareto_front_extreme_V2G.csv"):
    """
    Function description to follow.
    """
    t0 = performance()
    
    positions = [ind['position'] for ind in pareto_front]
    objectives = [ind['objective'] for ind in pareto_front]

    df_positions = pd.DataFrame(positions)
    df_objectives = pd.DataFrame(objectives, columns=['Voltage Variation','Cost'])
    df_pareto = pd.concat([df_positions, df_objectives], axis=1)
    df_pareto.to_csv(filename, index=False)
    # message_api(f"Pareto front data saved to {filename}")
    
    performance(t0)


def select_closest_to_ideal(pareto_front):
    """
    Function description to follow.
    """
    t0 = performance()
    
    objectives = np.array([solution['objective'] for solution in pareto_front])
    # Calculate minimum and maximum values for each target
    min_obj = objectives.min(axis=0)
    max_obj = objectives.max(axis=0)
    # Calculate the range of target values to avoid division by zero errors
    ranges = max_obj - min_obj
    ranges[ranges == 0] = 1  # Prevents division by zero
    # Normalize the target value
    normalized_objectives = (objectives - min_obj) / ranges
    # Define the ideal point as a point composed of the minimum normalized target values ([0, 0]), if you want the ideal point to be defined in the original target space as a point composed of the minimum values of each target, then the ideal point will still be [0, 0] after normalization. But if you do not normalize and calculate the distance directly in the original target space
    # The ideal point is the minimum value after normalization (that is, all zeros),
    ideal_point = np.zeros(objectives.shape[1])
    #
    distances = np.linalg.norm(normalized_objectives - ideal_point, axis=1)
    # Find the solution with the smallest distance
    best_index = np.argmin(distances)
    optimal_solution = pareto_front[best_index]
    
    performance(t0)
    return optimal_solution


