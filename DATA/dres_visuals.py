import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pypsa
import json
import yaml
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import joblib
from geographiclib.geodesic import Geodesic
import plotly.io as pio



## Load data


def load_data(datapath):
    network = pypsa.Network()
    network.import_from_netcdf(os.path.join(datapath, 'network.nc'))
    
    with open(os.path.join(datapath, 'ev_model_outputs.pkl'), 'rb') as f:
        ev_mod = pickle.load(f)

    # Open metadata file
    with open(os.path.join(datapath, 'run_metadata.json'), 'r') as f:
        run_metadata = json.load(f)
    
    # Open assets file
    assets_path = os.path.join(datapath, 'assets.yaml')
    if not os.path.exists(assets_path):
        assets_path = os.path.join(os.path.dirname(os.path.abspath(datapath)), 'inputs', 'assets.yaml')
    if not os.path.exists(assets_path):
        assets_path = os.path.abspath(os.path.join(datapath, '..', 'inputs', 'assets.yaml'))
    if not os.path.exists(assets_path):
        raise FileNotFoundError(
            f"Could not find assets.yaml in '{datapath}' or sibling inputs folder. "
            "Please place assets.yaml in the outputs folder or pass the correct data path."
        )

    with open(assets_path, 'r') as f:
        assets = yaml.safe_load(f)

    return network, assets, ev_mod, run_metadata


## Plotting functions


# WIND
def wind_comparison(title, turbine1, turbine2):
    plt.figure(figsize=(10, 6))
    t = turbine1['data'].index
    x = pd.to_datetime(t).to_list()
    y = turbine1['data'].to_list()
    plt.plot(x, y, label=turbine1['name'], color='blue')
    t = turbine2['data'].index
    x = pd.to_datetime(t).to_list()
    y = turbine2['data'].to_list()
    plt.plot(x, y, label=turbine2['name'], color='red', linestyle='--')
    plt.title(f'Power Output Comparison for {title}')
    plt.ylabel('Power (MW)')
    plt.legend()
    dtFmt = mdates.DateFormatter('%H:%M\n%d-%b')  # define the formatting
    # apply the format to the desired axis
    plt.gca().xaxis.set_major_formatter(dtFmt)
    plt.grid(True)
    plt.show()


def wind_scatter3d(network):

    # Identify all wind assets
    all_generators = network.generators_t.p_max_pu.keys().to_list()
    wind_turbines_list = [str for str in all_generators if "Wind" in str]
    farm = [int(str.split("_")[1]) for str in wind_turbines_list]
    turbine = [int(str.split("_")[-1]) for str in wind_turbines_list]
    wind_turbines_hover = [wt.replace(
        '_Turbine', '<br>Turbine') for wt in wind_turbines_list]
    wind_turbines_hover = [wt.replace('WindFarm_', 'WF')
                           for wt in wind_turbines_hover]
    wind_turbines_hover = [wt.replace('_', ' ') for wt in wind_turbines_hover]

    # Reorder list of wind assets by (1) Farm, (2) Turbine. (This is unsorted when as a "_" separated string)
    turbines = pd.DataFrame({
        'name': wind_turbines_list,
        'hover': wind_turbines_hover,
        'farm': farm,
        'turbine': turbine,
        'clr': '#000',
    })
    turbines.sort_values(['farm', 'turbine'], ascending=[
                         True, True], inplace=True)
    turbines.reset_index(drop=True, inplace=True)

    # Create colourscale
    farm_ids = turbines.farm.unique()
    L = len(farm_ids)
    clrs = px.colors.sample_colorscale('HSV', L, 0, 1)
    for i in range(L):
        turbines.loc[turbines['farm'] == farm_ids[i], 'clr'] = clrs[i]

    traces = []
    i = 0
    for index, turbine in turbines.iterrows():
        t = network.generators_t.p_max_pu[turbine['name']].index
        x = pd.to_datetime(t).to_list()
        z = network.generators_t.p_max_pu[turbine['name']].to_list()
        y = i * np.ones(len(x))
        text = [f"{turbine['hover']}" for j in range(len(x))]
        traces.append(go.Scatter3d(
            # x=x,y=y,z=z,
            x=x, y=y, z=z,
            mode="lines",
            name=turbine['name'],
            text=text,
            hovertemplate='%{text}<extra>%{x}<br>%{z}MW</extra>',
            line=dict(
                color=turbine['clr'],
                width=5,
            )
        ))
        i += 1
    layout = go.Layout(
        title="Fig. 1: Wind turbine output under storm conditions",
        height=650,
        scene=dict(
            aspectmode="manual",
            aspectratio=dict(x=1.5, y=1, z=.65),
            camera=dict(
                eye=dict(x=-0.1, y=-1.5, z=0.4),
                center=dict(x=-0.1, y=0, z=-0.25),
                # projection=dict(type="orthographic")
            ),
            xaxis=dict(title=''),
            yaxis=dict(title='', showticklabels=False),
            zaxis=dict(title='Turbine Power [MW]'),
        ),
        showlegend=False
    )
    fig = go.Figure(traces, layout)
    return fig


def wind_raw_data(assets, asset_name, freq="h"):

    
    df = network.generators_t.p_max_pu['WindFarm_5_(Hammers Hill)_Turbine_1']

    # Access weather API (OpenMeteo) for wind data
    hourly_weather = openmeteo(
        latitude=assets['WindFarms'][asset_name]['lat'],
        longitude=assets['WindFarms'][asset_name]['lon'],
        start_date=date_range[0].strftime('%Y-%m-%d'),
        end_date=date_range[-1].strftime('%Y-%m-%d'),
        fields=["wind_speed_10m", "wind_speed_100m", "wind_gusts_10m"]
    )

    trace = go.Scatter(
        x=df.index,
        y=df.values * assets['WindFarms'][asset_name]['rescale'],
        name=f"Single turbine output at {asset_name}",
        xaxis='x',
        yaxis='y',
    )
    trace_cap = go.Scatter(
        x=[df.index[0], df.index[-1]],
        y=[assets['WindFarms'][asset_name]['turbine_output'],
            assets['WindFarms'][asset_name]['turbine_output']],
        mode='none',
        fill="tozeroy",
        fillcolor="rgba(99, 110, 250, .25)",
        name=f"Installed capacity (as fraction of farm)",
        xaxis='x',
        yaxis='y',
    )
    trace_wind10 = go.Scatter(
        x=hourly_weather.index,
        y=hourly_weather.wind_speed_10m,
        name=f"Wind speed 10m",
        xaxis='x',
        yaxis='y2',
    )
    trace_wind100 = go.Scatter(
        x=hourly_weather.index,
        y=hourly_weather.wind_speed_100m,
        name=f"Wind speed 100m",
        xaxis='x',
        yaxis='y2',
    )
    trace_gusts10 = go.Scatter(
        x=hourly_weather.index,
        y=hourly_weather.wind_gusts_10m,
        name=f"Wind gusts 10m",
        xaxis='x',
        yaxis='y2',
    )
    farmname = assets['WindFarms'][asset_name]['name']
    n_turbines = assets['WindFarms'][asset_name]['n_turbines']
    farm_capacity = assets['WindFarms'][asset_name]['farm_output']
    layout = go.Layout(
        title=f"{asset_name}: {farmname} data, 1 of {n_turbines} turbines ({farm_capacity}MW farm capacity)",
        height=900,
        legend=dict(orientation="h"),
        plot_bgcolor='#fff',
        yaxis=dict(domain=[0.05, 0.5], gridcolor="#ccc", showgrid=True, zeroline=True, zerolinewidth=1, zerolinecolor='#000000',
                   linewidth=1, linecolor='#000000', ticks='outside', showline=True, title="Active Power [MW]"),
        yaxis2=dict(domain=[0.55, 1.], gridcolor="#ccc", showgrid=True, zeroline=True, zerolinewidth=1, zerolinecolor='#000000',
                    linewidth=1, linecolor='#000000', ticks='outside', showline=True, title="Wind speed [km/h]"),
        xaxis=dict(gridcolor="#ccc", showgrid=True, zeroline=True, zerolinewidth=1,
                   zerolinecolor='#000000', ticks='outside', showline=True),
    )
    return go.Figure([trace_wind10, trace_wind100, trace_gusts10, trace_cap, trace], layout)


def wind_raw_data_curve(INPUT_FOLDER, asset_name, freq="h"):

    assets = load_yaml(dir=INPUT_FOLDER, filename="assets")

    fullfilename = os.path.join(
        INPUT_FOLDER, assets['WindFarms'][asset_name]['file'])
    df = pd.read_csv(fullfilename)
    date_range = pd.date_range(
        start=f"2019-01-01 00:00", periods=len(df), freq=freq)
    df.index = date_range

    # Access weather API (OpenMeteo) for wind data
    hourly_weather = openmeteo(
        latitude=assets['WindFarms'][asset_name]['lat'],
        longitude=assets['WindFarms'][asset_name]['lon'],
        start_date=date_range[0].strftime('%Y-%m-%d'),
        end_date=date_range[-1].strftime('%Y-%m-%d'),
        fields=["wind_speed_10m", "wind_speed_100m", "wind_gusts_10m"]
    )

    
    trace_powercurve10 = go.Scatter(
        x=hourly_weather.wind_speed_10m,
        y=df['ActivePower'] * assets['WindFarms'][asset_name]['rescale'],
        mode='markers',
        marker=dict(opacity=0.4, size=2),
        xaxis='x',
        yaxis='y',
        showlegend=False,
    )
    trace_powercurve100 = go.Scatter(
        x=hourly_weather.wind_speed_100m,
        y=df['ActivePower'] * assets['WindFarms'][asset_name]['rescale'],
        mode='markers',
        marker=dict(opacity=0.4, size=2),
        xaxis='x2',
        yaxis='y',
        showlegend=False,
    )

    farmname = assets['WindFarms'][asset_name]['name']
    n_turbines = assets['WindFarms'][asset_name]['n_turbines']
    farm_capacity = assets['WindFarms'][asset_name]['farm_output']
    layout = go.Layout(
        title=f"{asset_name}: {farmname} data, 1 of {n_turbines} turbines ({farm_capacity}MW farm capacity)",
        height=500,
        legend=dict(orientation="h", yanchor='top', y=1.05),
        margin=dict(t=120),
        plot_bgcolor='#fff',
        yaxis=dict(gridcolor="#ccc", showgrid=True, zeroline=True, zerolinewidth=1, zerolinecolor='#000000',
                   showline=False, title="Active Power [MW]"),
        xaxis=dict(domain=[0., 0.45], gridcolor="#ccc", showgrid=True, zeroline=True, zerolinewidth=1,
                   zerolinecolor='#000000', ticks='outside', showline=False, title="Wind speed @10m [km/h]"),
        xaxis2=dict(domain=[0.55, 1.], gridcolor="#ccc", showgrid=True, zeroline=True, zerolinewidth=1,
                   zerolinecolor='#000000', ticks='outside', showline=False, title="Wind speed @100m [km/h]"),
    )
    return go.Figure([trace_powercurve10, trace_powercurve100], layout)


# EV
def ev_charge_rate(df):

    dt_minutes = (df['Arrival DateTime'] - df['Departure DateTime']).dt.total_seconds()/60

    ev_charge_rate = (df['SOC on Departure'] - df['SOC on Arrival']) / dt_minutes
    max_ev_charge_rate = np.max(ev_charge_rate)
    max_ev_charge_rate

    trace = go.Histogram(
        x=ev_charge_rate,
        xbins=dict( # bins used for histogram
            size=0.0000001
        ),
    )

    layout = go.Layout(
        title='EV charge rate',
        xaxis_title='SoC charge rage [SoC{UNIT?}/min]',
        yaxis_title='freq.',
    )

    fig = go.Figure(trace, layout)
    fig.show()


# EV
def ev_soc_timeline(df, max_ev_charge_rate):

    ev_batt_capacity = max(df['SOC on Departure'])

    ev_timeline_dt = []
    ev_timeline_soc = []

    ev_timeline_dt.append(pd.to_datetime('2019-01-01 00:00'))
    ev_timeline_soc.append(df['SOC on Departure'][0])

    for (i,row) in df.iterrows():
        dt = row['Departure DateTime'] - ev_timeline_dt[-1]
        
        time_to_max_charge = (ev_batt_capacity - ev_timeline_soc[-1]) / max_ev_charge_rate
        time_to_max_charge = pd.Timedelta(minutes=time_to_max_charge)
        time_at_max_charge = time_to_max_charge + ev_timeline_dt[-1]

        # Conditional logic to achieve constant charing ramp
        full_charge_achieved_before_departure = time_at_max_charge < row['Departure DateTime']
        if full_charge_achieved_before_departure and i!=0:
            # Evaluate the point in time during charge that capacity is reached (i.e. before 'Departure DateTime')
            ev_timeline_dt.append(time_at_max_charge)
            # Ensure the 'SOC on Departure' is not exceded during charge
            ev_timeline_soc.append(np.min([ev_batt_capacity,row['SOC on Departure']]))

        ev_timeline_dt.append(row['Departure DateTime'])
        ev_timeline_soc.append(row['SOC on Departure'])
        ev_timeline_dt.append(row['Departure DateTime']+pd.Timedelta(seconds=1))
        ev_timeline_soc.append(0)
        ev_timeline_dt.append(row['Arrival DateTime']-pd.Timedelta(seconds=1))
        ev_timeline_soc.append(0)
        ev_timeline_dt.append(row['Arrival DateTime'])
        ev_timeline_soc.append(row['SOC on Arrival'])

    ev_timeline = pd.DataFrame({
        'datetime': ev_timeline_dt,
        'soc': ev_timeline_soc,
    })
    ev_timeline.set_index('datetime', inplace=True)


    trace = go.Scatter(
        x=ev_timeline.index,
        y=ev_timeline.soc,
        mode='lines',
        fill='tozeroy',
    )
    layout = go.Layout(
        title="EV SoC timeline",
        yaxis=dict(title="SoC [UNITS TBC]")
    )
    fig = go.Figure(trace, layout)
    fig.show()

    return ev_timeline


# Network Map

def network_map(network, radii=None):

    
        
    traces = []

    network.buses['Bus'] = network.buses.index
    mapping = dict(network.buses[['Bus', 'x']].values)
    network.lines['x0'] = network.lines.bus0.map(mapping)
    network.lines['x1'] = network.lines.bus1.map(mapping)
    mapping = dict(network.buses[['Bus', 'y']].values)
    network.lines['y0'] = network.lines.bus0.map(mapping)
    network.lines['y1'] = network.lines.bus1.map(mapping)


    if radii:
        
        regions = []
        
        for i in range(len(radii)):
            
            radius = radii[i] 
            line = network.lines.loc[(network.lines.bus0==radius[0]) & (network.lines.bus1==radius[1])]

            cntr = (line['y0'].to_list()[0], line['x0'].to_list()[0])
            s = line['length'].to_list()[0] * 1000 #Distance (m)

            #Define the ellipsoid
            geod = Geodesic.WGS84

            #Solve the Direct problem
            azimuths = np.linspace(start=0,stop=360,num=181)
            xs = []
            ys = []
            for i in range(len(azimuths)):
                dir = geod.Direct(cntr[0],cntr[1],azimuths[i],s)
                xs.append(dir['lon2'])
                ys.append(dir['lat2'])

            regions.append({'name':f"{radius[0]}-{radius[1]}",'xs': xs,'ys': ys})


        for i in range(len(regions)):
            traces.append(go.Scattermap(
                lat=regions[i]['ys'],
                lon=regions[i]['xs'],
                mode='lines',
                line=dict(width=0.1),
                showlegend=False,
                fill="toself",
                fillcolor='rgba(0.5,0.5,0.5,0.1)',
                text=regions[i]['name'],
            ))


    for (i,row) in network.lines.iterrows():
        traces.append(go.Scattermap(
            lat=[row['y0'], row['y1']],
            lon=[row['x0'], row['x1']],
            mode='lines',
            marker=go.scattermap.Marker(
                size=14
            ),
            # showlegend=False,
            name=f"{row['bus0']} - {row['bus1']}",
            text=f"{row['bus0']} - {row['bus1']}",
        ))

    traces.append(go.Scattermap(
        lat=network.buses['y'].to_list(),
        lon=network.buses['x'].to_list(),
        mode='markers',
        marker=go.scattermap.Marker(
            color='#000000',
            size=8
        ),
        showlegend=False,
        text=network.buses.index.to_list()
    ))

    fig = go.Figure(traces)

    fig.update_layout(
        height=1000,
        margin=dict(t=0),
        hovermode='closest',
        map=dict(
            bearing=0,
            center=go.layout.map.Center(
                lat=59,
                lon=-3
            ),
            pitch=0,
            zoom=8.5
        )
    )

    fig.show()

    pio.renderers.default = 'notebook'  # Good for JupyterLab

    return fig


# IO FILES
def from_df(df, x_title="", y_title="", type="lines+markers"):

    traces = []

    for col in df.columns:
        traces.append(go.Scatter(
            x=df[col].index,
            y=df[col].values,
            mode=type,
            name=col
        ))

    layout = go.Layout(
        height=450,
        width=1450,
        margin=dict(t=10,b=50,l=50,r=10),
        xaxis=dict(title=x_title),
        yaxis=dict(title=y_title),
        legend_orientation='h',
        )
    fig = go.Figure(traces,layout)
    fig.show()



## MATPLOTLIB


def plot_normalized_pareto_front(pareto_front):
    objectives = np.array([solution['objective'] for solution in pareto_front])
    min_obj = objectives.min(axis=0)
    max_obj = objectives.max(axis=0)
    ranges = max_obj - min_obj
    ranges[ranges == 0] = 1  # Prevents division by zero
    normalized_objectives = (objectives - min_obj) / ranges
    plt.figure(figsize=(10, 6))
    plt.scatter(normalized_objectives[:, 0], normalized_objectives[:, 1], color='b', label='Normalized Pareto Solutions')
    ax = plt.gca()
    ax.xaxis.get_major_formatter().set_useOffset(False)
    ax.yaxis.get_major_formatter().set_useOffset(False)
    plt.xlabel('Normalized Voltage Variation')
    plt.ylabel('Normalized Cost')
    plt.title('Normalized Pareto Front')
    plt.grid(True)
    plt.legend()
    plt.show(block=False)


def plot_voltage_mag_over_time(network):
    bus_voltage = network.buses_t.v_mag_pu
    bus_list = ['KIRKWA3A']  #
    bus_voltage.to_csv('bus_voltage_data_extremeonedayV2G.csv')

    plt.figure(figsize=(10, 6))

    #
    for bus in bus_list:
        plt.plot(bus_voltage.index, bus_voltage[bus], label=bus)

    #
    plt.title('Bus Voltage Magnitude Over Time')
    plt.xlabel('Time')
    plt.ylabel('Voltage Magnitude (p.u.)')
    plt.legend(title='Buses')
    #
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show(block=False)



def plot_all_voltage_mag_over_time(network, ev_mod):
    #
    bus_voltage = network.buses_t.v_mag_pu
    #
    time_index = np.arange(len(bus_voltage.index))
    #
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    #
    for idx, bus in enumerate(bus_voltage.columns):
        ax.plot(time_index, [idx] * len(time_index), bus_voltage[bus], label=bus)

    #
    ax.set_xlabel('Time (Index)')
    ax.set_ylabel('Bus Index')
    ax.set_zlabel('Voltage Magnitude (p.u.)')
    ax.set_title('Bus Voltage Magnitude Over Time (3D)')
    #
    ax.legend(loc='best', title='Buses')

    #
    plt.show(block=False)

    
    # 24hours-charging-discharging-schedule
    plt.figure(figsize=(10, 6))
    hours = np.arange(24)
    plt.bar(hours, ev_mod['optimal_power_schedule'], color='b', label="Power Schedule (MW)", alpha=0.8)
    plt.axhline(0, color='gray', linestyle='--')  # The dividing line between charging and discharging
    plt.title("Optimal 24-Hour Charging and Discharging Schedule")
    plt.xlabel("Time (Hour)")
    plt.ylabel("Power (MW)")
    plt.legend()
    plt.grid(True)
    plt.xticks(hours)
    plt.show(block=False)



def plot_soc(meta, ev_mod):
    hours = np.arange(25)
    soc_values = [
        meta['initial_soc']]  # soc_values = [initial_soc] This line of code means to create a list of soc_values containing the values of the initial SOC (initial_soc), with initial_soc as the first element of the list
    soc = meta['initial_soc']

    for t, power in enumerate(ev_mod['optimal_power_schedule']):
        soc -= (power * meta['delta_t']) / meta['capacity']  # Update SOC based on scheduling and capacity
        soc += ev_mod['delta_soc'][t]
        soc_values.append(soc)  # # append only adds one element to the end of the list. It can be any type of data, such as integer, string, list, etc. Using append is a direct modification of the original list, rather than creating a new list.
        print(f"soccost {ev_mod['delta_soc'][t]}")

    # Mapping SOC changes over 24 hours
    plt.figure(figsize=(10, 6))
    plt.plot(hours, soc_values[:], marker='o', linestyle='-', color='g', label="SOC (%)")
    plt.title("24-Hour SOC Variation")
    plt.xlabel("Time (Hour)")
    plt.ylabel("State of Charge (SOC)")
    plt.legend()
    plt.grid(True)
    plt.xticks(hours)
    plt.show(block=False)



def plot_gwo_scoring(network, ev_mod):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(ev_mod['gwo_best_scores']) + 1), ev_mod['gwo_best_scores'], marker='o', linestyle='-')
    plt.title("Best Score Evolution over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Best Score")
    plt.grid(True)
    plt.show(block=False)

    # Load
    hourly_load = network.loads_t.p.sum(axis=1)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(hourly_load)), hourly_load.values, color='blue', width=0.8)
    plt.title("Hourly Total Load (MW)")
    plt.xlabel("Hour")
    plt.ylabel("Load (MW)")
    plt.grid(True)
    if len(hourly_load) <= 48:
        plt.xticks(range(len(hourly_load)), [str(x) for x in hourly_load.index], rotation=45)
    plt.tight_layout()
    plt.show(block=False)

    # all generators
    hourly_generation = network.generators_t.p.sum(axis=1)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(hourly_generation)), hourly_generation.values, color='green', width=0.8)
    plt.title("Hourly Total Generation (MW)")
    plt.xlabel("Hour")
    plt.ylabel("Generation (MW)")
    plt.grid(True)
    if len(hourly_generation) <= 48:
        plt.xticks(range(len(hourly_generation)), [str(x) for x in hourly_generation.index], rotation=45)
    plt.tight_layout()
    plt.show(block=False)

    return hourly_generation, hourly_load



def plot_renew_generation(network):
    renewable_carriers = ["wind", "EMEC Tidal Generator", "EDAY Tidal Generator", "Wave Generator"]
    all_generators = network.generators.index
    wind_generators = [g for g in network.generators.index if "WindTurbine_" in g]
    tidal_generators = [g for g in all_generators if "Tidal" in g]
    wave_generators = [g for g in all_generators if "Wave" in g]
    renewable_generators = wind_generators + tidal_generators + wave_generators

    hourly_renewable = network.generators_t.p[renewable_generators].sum(axis=1)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(hourly_renewable)), hourly_renewable.values, color='orange', width=0.8)
    plt.title("Hourly Total Renewable Generation (MW)")
    plt.xlabel("Hour")
    plt.ylabel("Renewable Generation (MW)")
    plt.grid(True)
    if len(hourly_renewable) <= 48:
        plt.xticks(range(len(hourly_renewable)), [str(x) for x in hourly_renewable.index], rotation=45)
    plt.tight_layout()
    plt.show(block=False)


def plot_supply_and_demand(network, hourly_generation, hourly_load):
    supply_demand_diff = hourly_generation - hourly_load

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(supply_demand_diff)), supply_demand_diff.values, color='purple', width=0.8)
    plt.title("Hourly Supply-Demand Difference (MW)")
    plt.xlabel("Hour")
    plt.ylabel("Generation - Load (MW)")
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    slack_generators = ["Slack_generator"]
    storage_generators = network.generators[network.generators.carrier == 'battery'].index if 'battery' in network.generators.carrier.values else []
    valid_generators = [g for g in network.generators.index if g not in slack_generators and g not in storage_generators]
    hourly_generation_filtered = network.generators_t.p[valid_generators].sum(axis=1)

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(hourly_generation_filtered)), hourly_generation_filtered.values, color='green', width=0.8)
    plt.title("Hourly Total Generation (Excluding Slack and Storage)")
    plt.xlabel("Hour")
    plt.ylabel("Generation (MW)")
    plt.grid(True)
    if len(hourly_generation_filtered) <= 48:
        plt.xticks(range(len(hourly_generation_filtered)), [str(x) for x in hourly_generation_filtered.index], rotation=45)
    plt.tight_layout()
    plt.show(block=False)



def plot_voltage_box_plot(network):
    bus_voltage = network.buses_t.v_mag_pu
    df = bus_voltage.reset_index()
    df.rename(columns={'snapshot': 'index'}, inplace=True)
    bus_voltage_long = df.melt(id_vars='index', var_name='Bus', value_name='Voltage Magnitude')
    bus_voltage_long.rename(columns={'index': 'Time'}, inplace=True)

    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Bus', y='Voltage Magnitude', data=bus_voltage_long, palette="Set3")
    plt.title("Voltage Magnitude Distribution by Bus", fontsize=16)
    plt.xlabel("Bus", fontsize=14)
    plt.ylabel("Voltage Magnitude (p.u.)", fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("voltage_magnitude_boxplot.png")  # Save as a png picture
    plt.show(block=False)

    ev_storage_units = [name for name in network.storage_units.index if name.startswith("KIRKWA3A_Storage")]
    ev_power_df = network.storage_units_t.p[ev_storage_units] 
    total_ev_power = ev_power_df.sum(axis=1)
    print(ev_power_df)
    print(total_ev_power)

    return total_ev_power


def ev_charging_discharging(meta, total_ev_power):
    plt.figure(figsize=(10, 6))
    plt.plot(total_ev_power.index, total_ev_power.values, marker='s', color='orange', label='Total EV Charging Power (MW)')
    plt.xlabel('Time')
    plt.ylabel('Power (MW)')
    plt.title('Total EV Charging/Discharging Power Over Time')
    plt.grid(True)
    plt.legend()
    plt.show(block=False)
    cost_per_hour = []
    for i, t in enumerate(total_ev_power.index):
        charging_power = -min(total_ev_power[t], 0) 
        cost = charging_power * meta['charging_price'][i]   
        cost_per_hour.append(cost)

    charging_cost_series = pd.Series(cost_per_hour,
                                    index=total_ev_power.index,
                                    name='Hourly EV Charging Cost')
    total_charging_cost = charging_cost_series.sum()
    print("total ev power:", total_ev_power)
    print(charging_cost_series)
    print("Charging cost:", total_charging_cost)


def plot_line_load_ratio(network, total_ev_power):
    line_loading = pd.DataFrame(index=network.snapshots, columns=network.lines.index, data=0.0)

    for line_name in network.lines.index:
        s_nom_line = network.lines.at[line_name, "s_nom"] 
        p_flow = network.lines_t.p0[line_name]
        q_flow = network.lines_t.q0[line_name]
        flow_mva = np.sqrt(p_flow**2 + q_flow**2)  
        line_loading[line_name] = flow_mva / s_nom_line * 100 
    print("Line load rate(%): ")
    print(line_loading)
    ## transformer load rate
    transformer_loading = pd.DataFrame(index=network.snapshots, columns=network.transformers.index, data=0.0)
    for trafo_name in network.transformers.index:
        s_nom_trafo = network.transformers.at[trafo_name, "s_nom"] 
        p_flow_trafo = network.transformers_t.p0[trafo_name]
        q_flow_trafo = network.transformers_t.q0[trafo_name]
        flow_trafo_mva = np.sqrt(p_flow_trafo**2 + q_flow_trafo**2)
        transformer_loading[trafo_name] = flow_trafo_mva / s_nom_trafo * 100

    print("Transformer load rate(%): ")
    print(transformer_loading)
    wind_generators = [g for g in network.generators.index if "WindTurbine_" in g]
    print("wind_generators",wind_generators)
    wind_generation = network.generators_t.p[wind_generators].sum(axis=1)
    print("wind_generation", wind_generation)
    charging_efficiency = 0.99
    discharging_efficiency = 0.99
    wind_emission_factor = 0.011  # kg/kWh
    ev_emission_factor = 0.011
    diesel_emission_factor = 1.94   # kg/kWh
    natural_gas_factor=0.96
    co2_charging = 0.0        
    co2_discharging = 0.0      
    co2_coal_replace = 0.0     
    for t, power_mw in enumerate(total_ev_power):
        power_kwh = abs(power_mw) * 1000.0
        print("t", t)
        print("power_mw", power_mw)
        if power_mw < 0:
            if wind_generation.iloc[t] > 1e-6:
                # 11 g/kWh => 0.011 kg/kWh
                co2_charging += power_kwh * wind_emission_factor
                print("wind energy")
            else:
                co2_charging += power_kwh * natural_gas_factor
                print("natural gas energy")
                print("power_mw", power_mw)
                print("power_kwh", power_kwh)
        else:
            co2_discharging += power_kwh * ev_emission_factor * (charging_efficiency * discharging_efficiency)
            co2_coal_replace += power_kwh * diesel_emission_factor

    print("1) CO2 emissions from EV charging (kg): ", co2_charging)
    print("2) CO2 emission of EV discharge process itself (kg): ", co2_discharging)
    print("3) For example, CO2 emissions when diesel power plants are used instead of discharge (kg): ", co2_coal_replace)

    total_co2 = co2_charging + co2_discharging

    bus_voltage = network.buses_t.v_mag_pu
    time_index = np.arange(len(bus_voltage.index))
    bus_index = np.arange(len(bus_voltage.columns))

    voltage_data = bus_voltage.to_numpy()