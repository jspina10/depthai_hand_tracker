import numpy as np
import pandas as pd

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import axes3D

# Inicialización de la figura global
# plt.ion()  # Activar el modo interactivo de Matplotlib para actualizar la gráfica.
# fig = plt.figure()
# ax_2 = fig.add_subplot(111, projection='3d')

import matplotlib
matplotlib.use("Qt5Agg")
print(matplotlib.get_backend())

import matplotlib.pyplot as plt

plt.ion()
fig = plt.figure(figsize=(9,9))
ax_2 = fig.add_subplot(111, projection='3d')

manager = plt.get_current_fig_manager()
manager.window.setGeometry(1160, 30, 760, 650)

def actualizar_grafica(arr):
    # Limpiar el gráfico actual para evitar superposiciones.
    ax_2.clear()

    # Asegurarse de que el arreglo sea de tipo numpy.ndarray.
    if not isinstance(arr, np.ndarray):
        print("El parámetro debe ser de tipo numpy.ndarray")
        return

    # Comprobar que el arreglo tenga 3 columnas (coordenadas x, y, z).
    if arr.shape[1] != 3:
        print("El arreglo debe tener exactamente 3 columnas para representar coordenadas 3D.")
        return

    # Extraer las coordenadas x, y, z.
    x = arr[:, 0]
    y = arr[:, 1]
    z = arr[:, 2]
    # Restar las coordenadas del primer punto para obtener posiciones relativas.
    x_rel = x - x[0]
    y_rel = y - y[0]
    z_rel = z - z[0]

    # Dibujar los puntos en el espacio 3D.
    ax_2.set_box_aspect([1, 1, 1])  # Relación uniforme entre los ejes
    ax_2.scatter(x_rel, y_rel, z_rel, color='blue', marker='o', s=20)  # 's' define el tamaño de los puntos.
    # ax_2.scatter(x, y, z, color='red', marker='o', s=20)  # 's' define el tamaño de los puntos.

    # Añadir los índices de los puntos.
    # for i in range(len(x_rel)):
    #     ax_2.text(x_rel[i], y_rel[i], z_rel[i], f'{i}', color='red', fontsize=10)
    # Configurar etiquetas y título.
    ax_2.set_title('Gráfica de puntos en 3D')
    ax_2.set_xlabel('Eje X')
    ax_2.set_ylabel('Eje Y')
    ax_2.set_zlabel('Eje Z')

    # Configurar los límites de los ejes para una mejor visualización.
    ax_2.set_xlim([x_rel.min() - 0.1, x_rel.max() + 0.1])
    ax_2.set_ylim([y_rel.min() - 0.1, y_rel.max() + 0.1])
    ax_2.set_zlim([z_rel.min() - 0.1, z_rel.max() + 0.1])

    # Actualizar la visualización.
    plt.draw()
    plt.pause(0.1)  # Pausar brevemente para permitir la actualización de la gráfica.


def load_hand_data():
    # Cargar los datos desde el archivo CSV
    # df = pd.read_csv("results/hand_data_interpolated.csv")

    df = pd.read_csv("results/hand_data.csv")
    
    # Extraer solo las columnas de los índices, omitiendo la columna de tiempo
    data = df.iloc[:, 1:].values  # Saltamos la primera columna ('time')

    # Redimensionar los datos a (rows, index, xyz) donde:
    # - rows es el número de filas (tiempos)
    # - index es el número de índices (21)
    # - xyz es el número de coordenadas por índice (3)
    reshaped_data = data.reshape(df.shape[0], 21, 3)
    
    return reshaped_data

data = load_hand_data()

for i in tqdm(range(len(data))):
    # print(data[i])
    actualizar_grafica(data[i])
