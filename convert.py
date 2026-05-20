import pandas as pd

for fname in [
    r"DATA\inputs\Transformer_types_definition.csv",
    r"DATA\inputs\Transformer_input_with_type_definition.csv",
]:
    df = pd.read_csv(fname, sep=';', engine='python')
    df.to_csv(fname, index=False)