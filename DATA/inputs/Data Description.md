| Data file                                   | type | Purpose                                                      | Unit                            |
| ------------------------------------------- | ---- | ------------------------------------------------------------ | ------------------------------- |
| assets.yaml                                 | yaml | Structured data file providing data on network components/assets | n/A                             |
| Bus_Input.csv                               | csv  | Dataset that lists 73 buses, nominal bus voltage (kV), coordinates, and type. | units specified in the document |
| config.yaml                                 | yaml | Linking the bus data with local regions                      | n/A                             |
| Load_Profile.csv                            | csv  | Hourly active (MW) and reactive (MVar) power flows for the simulated 17 buses for the year of 2019 | MW and MVar                     |
| processed_ev_usage.csv                      | csv  | Charging events of 235 EVs with departure and arrival time, state-of-charge status (1 indicates fully charged and 0 indicates fully discharged), distance travelled (km), charge used for completing journey | units specified in the document |
| Transformer_input_with_type_definition.xlsx | xlsx | Transformer bus data input                                   |                                 |
| Transformer_types_definition.xlsx           | xlsx | Transformer specifications to define network object parameter | units specified in the document |
| Transmission_input.csv                      | csv  | Transmission line characteristics including resistance and reactance | units specified in the document |
| WindTurbine_{##}.csv                        | csv  | Hourly wind generator power output for one year              | MW                              |





