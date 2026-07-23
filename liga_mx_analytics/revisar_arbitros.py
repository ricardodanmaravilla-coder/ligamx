import pandas as pd

archivo = 'data/historico_ligamx_completo.csv'
try:
    df = pd.read_csv(archivo)
    if 'Arbitro' in df.columns:
        print("Total de filas en el CSV:", len(df))
        print("Valores únicos en la columna Arbitro:")
        print(df['Arbitro'].value_counts(dropna=False).head(10))
    else:
        print("La columna 'Arbitro' NO existe en el CSV.")
except Exception as e:
    print("Error al leer el archivo:", e)