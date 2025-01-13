import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix


# --------------------------------------------------------------
# Configuración de gráficos
# --------------------------------------------------------------
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# --------------------------------------------------------------
# Crear conjunto de entrenamiento y prueba
# --------------------------------------------------------------
# Cargar el DataFrame procesado
df = pd.read_pickle("../data/03_data_feutures.pkl")  # Ajusta la ruta si es necesario
print(df.head())
# Eliminar columnas no necesarias para el entrenamiento del modelo
df_train = df.drop(["Participants", "Category", "Set"], axis=1)

# Separar características (X) y variable objetivo (y)
X = df_train.drop("Label", axis=1)  # 'Label' es la variable objetivo
y = df_train["Label"]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# --------------------------------------------------------------
# Verificar la distribución de la variable objetivo
# --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
df_train["Label"].value_counts().plot(kind="bar", color="lightblue", ax=ax, label="Total")
y_train.value_counts().plot(kind="bar", color="blue", ax=ax, label="Train")
y_test.value_counts().plot(kind="bar", color="darkblue", ax=ax, label="Test")
plt.title("Distribución de la variable objetivo en el conjunto Total, Train y Test")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Realizar Forward Feature Selection usando un Árbol de Decisión
# --------------------------------------------------------------
learner = ClassificationAlgorithms()

max_features = 3 # choose the best 10 features for model accuracy scoring 

selected_features, ordered_features, ordered_scores = learner.forward_selection(
    max_features, X_train, y_train
)

# Imprimir características seleccionadas y precisión
for i, (feature, score) in enumerate(zip(ordered_features, ordered_scores)):
    print(f"Iteración {i+1}: Feature seleccionada: {feature}, Precisión: {score:.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
plt.plot(range(0, max_features), ordered_scores)
plt.xlabel("# of features")
plt.ylabel("Accuracy Score")
plt.show()