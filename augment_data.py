# import pandas as pd
# import numpy as np

# def augment_data(filename='results/hand_data.csv', output_filename='results/hand_data_interpolated.csv', time_step=0.002):
#     # Cargar el DataFrame original
#     df = pd.read_csv(filename)

#     # Crear un nuevo índice de tiempo con el paso de tiempo deseado
#     start_time = df['time'].iloc[0]
#     end_time = df['time'].iloc[-1]
#     new_time_index = np.arange(start_time, end_time, time_step)

#     # Configurar 'Time' como el índice del DataFrame original para la interpolación
#     df.set_index('time', inplace=True)

#     # Reindexar el DataFrame al nuevo índice de tiempo e interpolar los valores
#     df_interpolated = df.reindex(new_time_index).interpolate(method='linear').reset_index()

#     # Renombrar la columna del índice a 'Time' para mantener consistencia
#     df_interpolated.rename(columns={'index': 'time'}, inplace=True)

#     # Guardar el DataFrame interpolado en un nuevo archivo CSV
#     df_interpolated.to_csv(output_filename, index=False)
#     print(f"Datos interpolados guardados en {output_filename}")

# # Ejemplo de uso
# augment_data()
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def augment_data(filename='results/hand_data.csv', output_filename='results/hand_data_interpolated.csv', time_step=0.002):
    df = pd.read_csv(filename)

    # Crear nuevo índice de tiempo
    start_time, end_time = df['time'].iloc[0], df['time'].iloc[-1]
    new_time_index = np.arange(start_time, end_time, time_step)

    # Interpolación usando interp1d de SciPy para cada columna
    df_interpolated = pd.DataFrame({'time': new_time_index})
    for col in df.columns:
        if col != 'time':
            interp_func = interp1d(df['time'], df[col], kind='linear', fill_value="extrapolate")
            df_interpolated[col] = interp_func(new_time_index)

    df_interpolated.to_csv(output_filename, index=False)
    print(f"Datos interpolados guardados en {output_filename}")

# Ejecutar la función
augment_data()
