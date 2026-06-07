#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
`DRES_Sim.py`

This class definition represents a single D-RES simulation instance, and includes
various runtime paramters specific to a that run. Multiple `dres_sim` objects can 
be created as part of any generic workflow, to compare results from different 
runs (within the same Python session). 

When calling the constructor (e.g. `sim = DRES_Sim("run_001")`), the user should specify the
following parameters:
    - ev_model
    - start_date
    - end_date
    - charging_price
    - discharging_price
    - pop_size
    - max_iter
    - initial_soc
    - delta_t
    - capacity
    - min_soc
    - max_soc
    - vehicle_power
"""


###############################################################################################################
# Standard Python Libraries
import os
import ast
import socket
from pathlib import Path

# Custom Libraries
from dres.dafni_utilities import performance, message_api, generate_run_metadata
import dres.nemo as nemo
import dres.base_model as base_model
from dres.config import get_or_create_config
from dres.plots import Plots
from dres.ev import EV

###############################################################################################################



# DRES class to hold paths and other global settings
class Paths():

    def __init__(self, verbose=True):

        DATA_PATH_MODE = os.getenv("DATA_PATH_MODE", 'local')

        if DATA_PATH_MODE.lower() == "docker":
            self.data_root = Path('data')
            self.inputs = self.data_root / 'inputs'
            self.outputs = self.data_root / 'outputs'
        elif DATA_PATH_MODE.lower() == "local":
            # Initialise `dres` object with data_root location
            cfg = get_or_create_config()
            self.data_root = Path(cfg['paths']['data_root'])
            self.inputs = self.data_root / 'inputs'
            self.outputs = self.data_root / 'outputs'
        else:
            raise ValueError("DATA_PATH_MODE environment variable must be set to either 'local' or 'docker'")

        # Determine whether Windows or Unix
        if os.name == 'nt':
            self.machine_id = "Windows"
        else:
            self.machine_id = socket.gethostname()
        
        # Prompt user on data_root location and how to change it
        if verbose:
            print(f"D-RES is configured to access data stored here: {self.data_root}.")
            print(f'To change this path, use:  dres.config.update_data_root("{str(Path.home() / "...")}")\n')

# ------------------------------------------------------------------------------------------------------------

# DRES main class
class Params():

    def __init__(self):

        # Interpret environment variables, second argument specifies default, 
        # these are only used during local testing (Jupyter & Docker Desktop),
        # DAFNI defaults are applied in `model_definition.yaml`
        self.ev_model = os.getenv("ev_model")
        self.start_date = os.getenv('start_date')
        self.end_date = os.getenv('end_date')
        self.charging_price = os.getenv("charging_price")
        self.discharging_price = os.getenv("discharging_price")
        self.pop_size = os.getenv("pop_size")
        self.max_iter = os.getenv("max_iter")
        self.initial_soc = os.getenv("initial_soc")
        self.delta_t = os.getenv("delta_t")
        self.capacity = os.getenv("capacity")
        self.min_soc = os.getenv("min_soc")
        self.max_soc = os.getenv("max_soc")
        self.vehicle_power = os.getenv("vehicle_power")
        self.rescale_load =  os.getenv("rescale_load")
        self.legacy =  os.getenv("legacy", "False").lower() in ("true", "1", "yes")
        self.wave_capacity = os.getenv("wave_capacity", "5")
        self.tidal_capacity = os.getenv("tidal_capacity", "7.2")


        # Type setting
        try:
            self.ev_model = self.ev_model.replace("''","'")
            self.start_date = self.start_date.replace("''","'")
            self.end_date = self.end_date.replace("''","'")
            self.charging_price = ast.literal_eval(self.charging_price)
            self.discharging_price = ast.literal_eval(self.discharging_price)
            self.pop_size = int(self.pop_size)
            self.max_iter = int(self.max_iter)
            self.initial_soc = float(self.initial_soc)
            self.delta_t = float(self.delta_t)
            self.capacity = float(self.capacity)
            self.min_soc = float(self.min_soc)
            self.max_soc = float(self.max_soc)
            self.vehicle_power = float(self.vehicle_power)
            self.rescale_load = float(self.rescale_load)
            self.wave_capacity = float(self.wave_capacity)
            self.tidal_capacity = float(self.tidal_capacity)
        except:
            raise ValueError("Error interpreting DRES simulation parameters from environment variables.\
                             Please ensure all parameters are specified correctly:\n\
                             \
                             ev_model = (str)\n \
                             start_date = (str)\n \
                             end_date = (str)\n \
                             charging_price = (dict)\n \
                             discharging_price = (dict)\n \
                             pop_size = (int)\n \
                             max_iter = (int)\n \
                             initial_soc = (float)\n \
                             delta_t = (float)\n \
                             capacity = (float)\n \
                             min_soc = (float)\n \
                             max_soc = (float)\n \
                             vehicle_power = (float)\n \
                             rescale_load = (float)\n \
                             legacy = (bool)\n \
                             ")
            

# ------------------------------------------------------------------------------------------------------------

# DRES main class
class DRES_Sim():
    # Gather simulation parameters from system environment variables
    
    def __init__(self, simulation_name):

        if not simulation_name:
            raise ValueError("Please state e.g. 'simulation-1', 'simulation-xyz', etc. when creating a DRES_Sim object.")
        self.simulation_name = simulation_name

        # Initialise `dres` object with data_root location
        self.verbose = False
        self.paths = Paths(verbose=self.verbose)
        
        # Load simulation configuration
        self.simulation_config = nemo.load_yaml(dir=self.paths.inputs, filename=f"{self.simulation_name}.yaml")
        # Gather assets
        self.assets = nemo.load_yaml(dir=self.paths.inputs, filename="assets")

        # Create data containers
        self.ev = EV(self)
        self.plots = Plots(self)
        self.params = Params()
        
        # Set code performance start time
        self.t0 = performance(always_verbose=True)


        

    # Main run sequence
    def build_baseline_model(self):
        message_api(msg="# Build baseline power network model")
        

        # LEGACY VAR - TO BE DELETED
        weather_state="normal"
        storm_wind_speeds=None
        slack_p_nom=None

        network = base_model.create_pypsa_network()
        base_model.create_bus_network(network, self)
        base_model.create_transmission_network(network, self)
        base_model.add_loads_to_network(network, self)
        base_model.add_wind_turbines_to_network(network, self, weather_state, storm_wind_speeds)
        base_model.add_generators_to_network(network, self.assets)
        base_model.add_storage_to_network(network, self)
        base_model.define_transformers(network, self)
        base_model.add_shunts_to_network(network, self)
        base_model.add_offshore_marine_to_network(network, self)

        return network

