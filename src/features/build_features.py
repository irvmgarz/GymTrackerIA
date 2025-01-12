import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from TemporalAbstraction import NumericalAbstraction
from sklearn.decomposition import PCA
from DataTransformation import PrincipalComponentAnalysis

# --------------------------------------------------------------
# Cargar el DataFrame
# --------------------------------------------------------------

df = pd.read_pickle("../data/02_outliers_removed.pkl")

# Convertir la columna 'epoch (ms)' al formato datetime si no lo está
df['epoch (ms)'] = pd.to_datetime(df['epoch (ms)'])

# Imputar valores faltantes
predictor_columns = list(df.columns[1:7])
for col in predictor_columns:
    df[col] = df[col].interpolate()

# Inicializar la columna 'Duration' si aún no se ha calculado
if 'Duration' not in df.columns:
    df['Duration'] = 0.0
    for s in df["Set"].unique():
        subset = df[df["Set"] == s]
        start_time = subset.iloc[0]["epoch (ms)"]
        end_time = subset.iloc[-1]["epoch (ms)"]
        duration = (end_time - start_time).total_seconds()
        df.loc[df["Set"] == s, "Duration"] = duration

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()
lowpass = LowPassFilter()
fs = 1000 / 200  # Frecuencia de muestreo
cutoff = 1.11  # Frecuencia de corte

for col in predictor_columns:
    df_lowpass = lowpass.low_pass_filter(df_lowpass, col, fs, cutoff)

# --------------------------------------------------------------
# Principal Component Analysis (PCA)
# --------------------------------------------------------------

df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()

# Determinar la varianza explicada por cada componente principal
pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

# Graficar la varianza explicada por cada componente principal
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(predictor_columns) + 1), pc_values, marker='o', linestyle='-', color='b')
plt.xlabel("Principal Component Number")
plt.ylabel("Explained Variance Ratio")
plt.title("Explained Variance by Principal Component")
plt.grid(True)
plt.show()

# Aplicar PCA para reducir a 3 componentes principales
df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

df_squared = df_pca.copy()

# Calcular la suma de cuadrados y la raíz cuadrada para los acelerómetros y giroscopios
acc_r = df_squared["Accelerometer_x"] ** 2 + df_squared["Accelerometer_y"] ** 2 + df_squared["Accelerometer_z"] ** 2
gyr_r = df_squared["Gyroscope_x"] ** 2 + df_squared["Gyroscope_y"] ** 2 + df_squared["Gyroscope_z"] ** 2

df_squared["acc_r"] = np.sqrt(acc_r)  # Magnitud del vector acelerómetro
df_squared["gyr_r"] = np.sqrt(gyr_r)  # Magnitud del vector giroscopio

# Visualizar los resultados para un conjunto específico
subset3 = df_squared[df_squared["Set"] == 15]

# Graficar las magnitudes para el Set específico
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(10, 6))
ax[0].plot(subset3["epoch (ms)"], subset3["acc_r"], label="Accelerometer Magnitude", color="purple")
ax[0].set_title("Accelerometer Magnitude")
ax[0].legend()
ax[1].plot(subset3["epoch (ms)"], subset3["gyr_r"], label="Gyroscope Magnitude", color="red")
ax[1].set_title("Gyroscope Magnitude")
ax[1].legend()
plt.show()

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal = df_squared.copy()
predictor_columns += ["acc_r", "gyr_r"]  # Agregar las magnitudes calculadas a las columnas predictoras

NumAbs = NumericalAbstraction()

# Definir el tamaño de la ventana (ws)
ws = int(1000 / 200)  # Basado en la frecuencia de muestreo

# Crear una lista para almacenar los subconjuntos procesados
df_temporal_list = []

# Iterar sobre cada Set para aplicar la abstracción temporal
for s in df_temporal["Set"].unique():
    subset = df_temporal[df_temporal["Set"] == s].copy()
    subset = NumAbs.abstract_numerical(subset, predictor_columns, ws, "mean")  # Media
    subset = NumAbs.abstract_numerical(subset, predictor_columns, ws, "std")   # Desviación estándar
    df_temporal_list.append(subset)

# Combinar todos los subconjuntos en un DataFrame final
df_temporal = pd.concat(df_temporal_list)

# Visualizar los resultados de abstracción temporal para un Set específico
subset = df_temporal[df_temporal["Set"] == 15]

# Graficar abstracciones temporales usando los nombres correctos de las columnas
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(10, 6))

# Graficar la abstracción temporal para Accelerometer_y
subset[["Accelerometer_y", "Accelerometer_ytempmeanws5", "Accelerometer_ytempstdws5"]].plot(
    ax=ax[0], title="Accelerometer Y Temporal Abstraction", legend=True
)

# Graficar la abstracción temporal para Gyroscope_y
subset[["Gyroscope_y", "Gyroscope_ytempmeanws5", "Gyroscope_ytempstdws5"]].plot(
    ax=ax[1], title="Gyroscope Y Temporal Abstraction", legend=True
)

plt.show()


print(subset.columns.tolist())