import pandas as pd

# Load the tidal data
df = pd.read_excel("GB tidal_stream_power 2019.xlsx")

# Use Westray-South column as recommended by Shona Pennock
# This represents the active tidal site connecting at Eday
westray = df[['Date/time', 'Westray-South (200 MW)']].copy()
westray.columns = ['time', 'power_MW']

# Set datetime index
westray['time'] = pd.to_datetime(westray['time'])
westray.set_index('time', inplace=True)

# Resample from 10-minute to hourly (mean)
westray_hourly = westray.resample('h').mean()

# Normalise by max installed capacity (200 MW)
westray_hourly['power (pu)'] = westray_hourly['power_MW'] / 200.0

# Keep only the normalised column
westray_hourly = westray_hourly[['power (pu)']]

# Confirm range
print(f"Shape: {westray_hourly.shape}")
print(f"Min: {westray_hourly['power (pu)'].min():.4f}")
print(f"Max: {westray_hourly['power (pu)'].max():.4f}")
print(f"Mean CF: {westray_hourly['power (pu)'].mean():.4f}")
print(f"Date range: {westray_hourly.index[0]} to {westray_hourly.index[-1]}")
print(westray_hourly.head(10))

# Save to CSV
westray_hourly.to_csv("DATA/inputs/tidal_orkney_2019.csv")
print("\nSaved to DATA/inputs/tidal_orkney_2019.csv")