import pandas as pd

# 1. Cargar el archivo ordenado
df = pd.read_parquet('recommendations_sorted.parquet')

# 2. Mostrar las primeras 20 filas
print(df.head(40))