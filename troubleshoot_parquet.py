# Example in a Python shell or script:
import pandas as pd
ticker = 'XYZ' # Replace with problematic ticker
df = pd.read_parquet(f'data/CSCO.parquet')
print(df.tail(10))
