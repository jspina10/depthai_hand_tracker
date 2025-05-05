import pandas as pd
import numpy as np
import time

simulation_correct_points = np.array([
    [ 0.0,  0.0,  0.0],
    [ 0.01050094, -0.02501465,  0.02648499],
    [ 0.0232242 , -0.05420452,  0.04298499],
    [ 0.03334483, -0.08009647,  0.05698498],
    [ 0.03968293, -0.09528343,  0.06319997],
    [-0.01091455, -0.08092572,  0.02217801],
    [-0.01820767, -0.12220454,  0.02968801],
    [-0.01940785, -0.1470935 ,  0.03302101],
    [-0.01910385, -0.16216626,  0.03472401],
    [-0.01141782, -0.08059212,  0.00470901],
    [-0.01761261, -0.12480806,  0.00635901],
    [-0.01954348, -0.15385661,  0.00772401],
    [-0.01863315, -0.16934389,  0.00836501],
    [-0.00560389, -0.07926649, -0.009884  ],
    [-0.00691683, -0.11952255, -0.013412  ],
    [-0.00746009, -0.14431199, -0.015729  ],
    [-0.00835035, -0.1603197 , -0.01645999],
    [ 0.00344366, -0.07116528, -0.019501  ],
    [ 0.00635098, -0.10672598, -0.02478   ],
    [ 0.00861044, -0.12740118, -0.02765101],
    [ 0.0105063 , -0.14103868, -0.02900601]
])


class HandDataStorage:
    def __init__(self):
        self.hand_data = []
        self.time_data = []
        self.start_time = None
        self.total_frames = 0

    def save(self, frame, hands, bag):
        self.total_frames += 1

        if self.total_frames < 20:
            return

        if self.total_frames == 20:
            print("########")
            print("Saving frames")

        if self.start_time is None:
            self.start_time = time.time()

        current_time = round(time.time() - self.start_time, 8)

        for hand in hands:
            landmarks = hand.world_landmarks - hand.world_landmarks[0]
            landmarks = hand.world_landmarks / 1000
            self.hand_data.append(landmarks)
            self.time_data.append(current_time)
            actualizar_grafica(landmarks)

            # # Test with model denoising , but does not work good
            # # Plot with two subplots used to better visualization
            # points = hand.world_landmarks / 1000
            # adjusted_points = points - points[0]
            # R = np.array([
            #     [1,  0,  0],
            #     [0,  0,  1],
            #     [0, -1,  0]
            # ])
            # # Axis are rotated from camera frame to general frame
            # adjusted_points = adjusted_points @ R.T

            # # simulation_correct_points = np.array([-0.2022797753600513,-0.44651482753559324,1.4174861511099477,-0.19177883541118002,-0.47152947329730555,1.4439711444572425,-0.1790555762421172,-0.5007193506887352,1.4604711363963363,-0.16893494571132994,-0.5266112995759118,1.4744711299850182,-0.16259684598439286,-0.5417982571962712,1.4806861259696353,-0.21319432146706488,-0.5274405447419901,1.4396641580516942,-0.22048744759994934,-0.5687193655413809,1.4471741626869536,-0.22168762612665738,-0.593608329167631,1.4505071634543152,-0.22138362211060217,-0.6086810918607097,1.4522101632653412,-0.21369759835834248,-0.5271069454045101,1.4221951583707528,-0.21989238894763763,-0.5713228925000365,1.4238451623102584,-0.2218232565678579,-0.6003714393117181,1.4252101635420262,-0.2209129229419185,-0.6158587192996481,1.4258511629686612,-0.2078836614032952,-0.5257813151886866,1.407602154683587,-0.20919660709249455,-0.5660373734850929,1.404074155526337,-0.20973986824670543,-0.5908268139576761,1.4017571558770971,-0.21063012162632935,-0.6068345279653455,1.4010261564456807,-0.1988361158388715,-0.5176801078177101,1.397985148944163,-0.1959288003414814,-0.5532408039192818,1.3927061471095066,-0.19366933723109334,-0.5739160112037466,1.389835145681919,-0.19177347759145744,-0.5875535058042249,1.3884801444831287])
            # adjusted_points_rotated = self.rotate_points(
            #     simulation_correct_points[[0, 5, 17]],
            #     adjusted_points[[0, 5, 17]],
            #     adjusted_points
            # )
            # self.hand_data.append(adjusted_points_rotated)
            # self.time_data.append(current_time)
            # actualizar_grafica(adjusted_points_rotated)

    @staticmethod
    def rotate_points(array_1, array_2, array_3):
        """
        Calcula la matriz de rotación que alinea array_2 con array_1 y
        aplica esta rotación a array_3.

        Parámetros:
        - array_1: np.array de tamaño (N,3), puntos de referencia
        - array_2: np.array de tamaño (N,3), puntos a alinear con array_1
        - array_3: np.array de tamaño (M,3), puntos a rotar con la transformación calculada

        Retorna:
        - array_3_rotated: np.array de tamaño (M,3), los puntos rotados
        """

        # 1. Centrar los puntos respecto al primer punto (restamos el primer punto a todos)
        p1 = array_1[0]
        p2 = array_2[0]

        array_1_centered = array_1 - p1
        array_2_centered = array_2 - p2  # Es igual porque p1 == p2

        # 2. Encontrar la matriz de rotación con SVD
        H = array_2_centered.T @ array_1_centered  # Matriz de covarianza cruzada
        U, S, Vt = np.linalg.svd(H)  # Descomposición en valores singulares
        R = Vt.T @ U.T  # Matriz de rotación óptima

        # Asegurar que R es una rotación válida (evitar reflexiones)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # 3. Aplicar la rotación a array_3
        array_3_centered = array_3 - p2  # Centramos array_3 respecto a array_2
        array_3_rotated = (R @ array_3_centered.T).T + p1  # Aplicamos la rotación y desplazamos

        return array_3_rotated

    def end(self, offset=np.array([-0.20227978, -0.44651483,  1.41748615]), filename='results/hand_data.csv'):
        if not self.hand_data or not self.time_data:
            print("No hay datos para guardar.")
            return

        processed_data = []

        for i, points in enumerate(self.hand_data):
            points = np.array(points)

            # reference_point = points[0]
            adjusted_points = points - points[0]
            R = np.array([
                [1,  0,  0],
                [0,  0,  1],
                [0, -1,  0]
            ])
            # Axis are rotated from camera frame to general frame 
            adjusted_points = adjusted_points @ R.T

            # simulation_correct_points = np.array([-0.2022797753600513,-0.44651482753559324,1.4174861511099477,-0.19177883541118002,-0.47152947329730555,1.4439711444572425,-0.1790555762421172,-0.5007193506887352,1.4604711363963363,-0.16893494571132994,-0.5266112995759118,1.4744711299850182,-0.16259684598439286,-0.5417982571962712,1.4806861259696353,-0.21319432146706488,-0.5274405447419901,1.4396641580516942,-0.22048744759994934,-0.5687193655413809,1.4471741626869536,-0.22168762612665738,-0.593608329167631,1.4505071634543152,-0.22138362211060217,-0.6086810918607097,1.4522101632653412,-0.21369759835834248,-0.5271069454045101,1.4221951583707528,-0.21989238894763763,-0.5713228925000365,1.4238451623102584,-0.2218232565678579,-0.6003714393117181,1.4252101635420262,-0.2209129229419185,-0.6158587192996481,1.4258511629686612,-0.2078836614032952,-0.5257813151886866,1.407602154683587,-0.20919660709249455,-0.5660373734850929,1.404074155526337,-0.20973986824670543,-0.5908268139576761,1.4017571558770971,-0.21063012162632935,-0.6068345279653455,1.4010261564456807,-0.1988361158388715,-0.5176801078177101,1.397985148944163,-0.1959288003414814,-0.5532408039192818,1.3927061471095066,-0.19366933723109334,-0.5739160112037466,1.389835145681919,-0.19177347759145744,-0.5875535058042249,1.3884801444831287])
            adjusted_points_rotated = self.rotate_points(
                simulation_correct_points[[0, 5, 17]],
                adjusted_points[[0, 5, 17]],
                adjusted_points
            )

            adjusted_points_rotated += offset
            adjusted_points_flat = adjusted_points_rotated.flatten().tolist()
            row = [self.time_data[i]] + adjusted_points_flat
            processed_data.append(row)

        columns = ['time'] + [f'point{p}_{axis}' for p in range(21) for axis in ['x', 'y', 'z']]
        df = pd.DataFrame(processed_data, columns=columns)

        # Guardar en CSV
        df.to_csv(filename, index=False)
        print(f"Datos guardados en {filename}")


import matplotlib
matplotlib.use("Qt5Agg")
print(matplotlib.get_backend())

import matplotlib.pyplot as plt

plt.ion()
fig = plt.figure(figsize=(9,9))
ax_2 = fig.add_subplot(111, projection='3d')

manager = plt.get_current_fig_manager()
manager.window.setGeometry(1160, 30, 760, 650)

conexiones = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [5,9],[9,10],[10,11],[11,12],
    [9,13],[13,14],[14,15],[15,16],
    [13,17],[17,18],[18,19],[19,20],[0,17]
]


def actualizar_grafica(arr):
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
    # x_rel = x - x[0]
    # y_rel = y - y[0]
    # z_rel = z - z[0]

    x_rel = x
    y_rel = y
    z_rel = z

    # Dibujar los puntos en el espacio 3D.
    ax_2.set_box_aspect([1, 1, 1])  # Relación uniforme entre los ejes
    ax_2.scatter(x_rel, y_rel, z_rel, color='blue', marker='o', s=20)  # 's' define el tamaño de los puntos.
    # ax_2.scatter(x, y, z, color='green', marker='o', s=20)  # 's' define el tamaño de los puntos.

    # Añadir los índices de los puntos.
    for i in range(len(x_rel)):
        ax_2.text(x_rel[i], y_rel[i], z_rel[i], f'{i}', color='red', fontsize=10)
        # ax_2.text(x[i], y[i], z[i], f'{i}', color='purple', fontsize=10)

    for start, end in conexiones:
        ax_2.plot(
            [x_rel[start], x_rel[end]],
            [y_rel[start], y_rel[end]],
            [z_rel[start], z_rel[end]],
            color='blue'
        )
        # ax_2.plot(
        #     [x[start], x[end]],
        #     [y[start], y[end]],
        #     [z[start], z[end]],
        #     color='yellow'
        # )
    # Configurar etiquetas y título.
    ax_2.set_title('3D Extracted Landmarks')
    ax_2.set_xlabel('X [m]')
    ax_2.set_ylabel('Y [m]')
    ax_2.set_zlabel('Z [m]')

    # Configurar los límites de los ejes para una mejor visualización.
    ax_2.set_xlim([x_rel.min() - 0.05, x_rel.max() + 0.05])
    ax_2.set_ylim([y_rel.min() - 0.05, y_rel.max() + 0.05])
    ax_2.set_zlim([z_rel.min() - 0.05, z_rel.max() + 0.05])

    # Actualizar la visualización.
    plt.draw()
    plt.pause(0.1)  # Pausar brevemente para permitir la actualización de la gráfica.


# import numpy as np
# import matplotlib.pyplot as plt
# from landmark_reconstruction import LandmarkReconstruction

# # Inicialización de la figura global con dos subplots
# plt.ion()  # Activar el modo interactivo de Matplotlib para actualizar la gráfica.
# fig = plt.figure(figsize=(12, 6))

# ax_1 = fig.add_subplot(121, projection='3d')  # Subplot 1 para los puntos originales
# ax_2 = fig.add_subplot(122, projection='3d')  # Subplot 2 para los puntos reconstruidos

# conexiones = [
#     [0,1],[1,2],[2,3],[3,4],
#     [0,5],[5,6],[6,7],[7,8],
#     [5,9],[9,10],[10,11],[11,12],
#     [9,13],[13,14],[14,15],[15,16],
#     [13,17],[17,18],[18,19],[19,20],[0,17]
# ]

# landmark_reconstruction = LandmarkReconstruction()

# def actualizar_grafica(arr):
#     ax_1.clear()
#     ax_2.clear()

#     # Asegurarse de que el arreglo sea de tipo numpy.ndarray.
#     if not isinstance(arr, np.ndarray):
#         print("El parámetro debe ser de tipo numpy.ndarray")
#         return

#     # Comprobar que el arreglo tenga 3 columnas (coordenadas x, y, z).
#     if arr.shape[1] != 3:
#         print("El arreglo debe tener exactamente 3 columnas para representar coordenadas 3D.")
#         return

#     # Predicción de landmark reconstruidos
#     arr2 = arr - arr[0]
#     # arr2 = arr2 + np.array([-0.2023, -0.4465,  1.4175])
#     landmark_reconstructed = landmark_reconstruction.predict(arr2)

#     # Extraer las coordenadas x, y, z de los puntos originales
#     x = arr[:, 0]
#     y = arr[:, 1]
#     z = arr[:, 2]

#     # Extraer las coordenadas x, y, z de los puntos reconstruidos
#     x_rec = landmark_reconstructed[:, 0]
#     y_rec = landmark_reconstructed[:, 1]
#     z_rec = landmark_reconstructed[:, 2]

#     # Restar las coordenadas del primer punto para obtener posiciones relativas
#     x_rel = x - x[0]
#     y_rel = y - y[0]
#     z_rel = z - z[0]

#     x_rec_rel = x_rec - x_rec[0]
#     y_rec_rel = y_rec - y_rec[0]
#     z_rec_rel = z_rec - z_rec[0]

#     # --- GRAFICAR PUNTOS ORIGINALES EN ax_1 ---
#     ax_1.set_box_aspect([1, 1, 1])  # Relación uniforme entre los ejes
#     ax_1.scatter(x_rel, y_rel, z_rel, color='blue', marker='o', s=20)

#     # Añadir etiquetas de los puntos originales
#     for i in range(len(x_rel)):
#         ax_1.text(x_rel[i], y_rel[i], z_rel[i], f'{i}', color='red', fontsize=10)

#     # Dibujar conexiones entre los puntos originales
#     for start, end in conexiones:
#         ax_1.plot([x_rel[start], x_rel[end]], [y_rel[start], y_rel[end]], [z_rel[start], z_rel[end]], color='blue')

#     ax_1.set_title('Puntos Originales')
#     ax_1.set_xlabel('Eje X')
#     ax_1.set_ylabel('Eje Y')
#     ax_1.set_zlabel('Eje Z')

#     # Configurar los límites de los ejes para una mejor visualización
#     ax_1.set_xlim([x_rel.min() - 0.1, x_rel.max() + 0.1])
#     ax_1.set_ylim([y_rel.min() - 0.1, y_rel.max() + 0.1])
#     ax_1.set_zlim([z_rel.min() - 0.1, z_rel.max() + 0.1])

#     # --- GRAFICAR PUNTOS RECONSTRUIDOS EN ax_2 ---
#     ax_2.set_box_aspect([1, 1, 1])
#     ax_2.scatter(x_rec_rel, y_rec_rel, z_rec_rel, color='green', marker='o', s=20)

#     # Añadir etiquetas de los puntos reconstruidos
#     for i in range(len(x_rec_rel)):
#         ax_2.text(x_rec_rel[i], y_rec_rel[i], z_rec_rel[i], f'{i}', color='purple', fontsize=10)

#     # Dibujar conexiones entre los puntos reconstruidos
#     for start, end in conexiones:
#         ax_2.plot([x_rec_rel[start], x_rec_rel[end]], [y_rec_rel[start], y_rec_rel[end]], [z_rec_rel[start], z_rec_rel[end]], color='green')

#     ax_2.set_title('Puntos Reconstruidos')
#     ax_2.set_xlabel('Eje X')
#     ax_2.set_ylabel('Eje Y')
#     ax_2.set_zlabel('Eje Z')

#     # Configurar los límites de los ejes para una mejor visualización
#     ax_2.set_xlim([x_rec_rel.min() - 0.1, x_rec_rel.max() + 0.1])
#     ax_2.set_ylim([y_rec_rel.min() - 0.1, y_rec_rel.max() + 0.1])
#     ax_2.set_zlim([z_rec_rel.min() - 0.1, z_rec_rel.max() + 0.1])

#     # Actualizar la visualización
#     plt.draw()
#     plt.pause(0.1)
