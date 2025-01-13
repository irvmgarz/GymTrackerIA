import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from LearningAlgorithms import ClassificationAlgorithms

# --------------------------------------------------------------
# Importar conjuntos de datos y realizar preprocesamiento
# --------------------------------------------------------------

# Leer el archivo generado previamente
df = pd.read_pickle("../data/03_data_feutures.pkl")
print("Columnas disponibles en el dataset:", df.columns.tolist())

# Eliminar columnas innecesarias para el entrenamiento del modelo
X = df.drop(["Label", "Participants", "Category", "Set"], axis=1)
y = df["Label"]

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Confirmar que las divisiones son correctas
print("Distribución de clases en el conjunto de entrenamiento:")
print(y_train.value_counts())
print("\nDistribución de clases en el conjunto de prueba:")
print(y_test.value_counts())

# Instanciar el modelo
learner = ClassificationAlgorithms()

# --------------------------------------------------------------
# Definir conjuntos de características
# --------------------------------------------------------------

feature_set_1 = X_train.columns[:5].tolist()
feature_set_2 = X_train.columns[5:10].tolist()
feature_set_3 = X_train.columns[10:15].tolist()
feature_set_4 = X_train.columns[15:20].tolist()

max_features = 10
selected_features, ordered_features, ordered_scores = learner.forward_selection(
    max_features, X_train, y_train
)

# --------------------------------------------------------------
# Grid search para selección de hiperparámetros y modelos
# --------------------------------------------------------------

possible_feature_sets = [
    feature_set_1,
    feature_set_2,
    feature_set_3,
    feature_set_4,
    selected_features,
]

feature_names = [
    "Feature Set 1",
    "Feature Set 2",
    "Feature Set 3",
    "Feature Set 4",
    "Selected Features",
]

iterations = 1  # Número de iteraciones para modelos no deterministas
score_df = pd.DataFrame()

for i, f in zip(range(len(possible_feature_sets)), feature_names):
    print(f"Evaluando el conjunto de características: {f}")
    selected_train_X = X_train[possible_feature_sets[i]]
    selected_test_X = X_test[possible_feature_sets[i]]

    # Inicializar métricas
    performance_test_nn = 0
    performance_test_rf = 0

    for it in range(iterations):
        print(f"\tEntrenando red neuronal (Iteración {it + 1})...")
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.feedforward_neural_network(
            selected_train_X,
            y_train,
            selected_test_X,
            gridsearch=False,
        )
        performance_test_nn += accuracy_score(y_test, class_test_y)

        print(f"\tEntrenando Random Forest (Iteración {it + 1})...")
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.random_forest(
            selected_train_X,
            y_train,
            selected_test_X,
            gridsearch=True,
        )
        performance_test_rf += accuracy_score(y_test, class_test_y)

    # Promediar los resultados
    performance_test_nn /= iterations
    performance_test_rf /= iterations

    # Clasificadores deterministas
    print("\tEntrenando K-Nearest Neighbors (KNN)...")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.k_nearest_neighbor(
        selected_train_X,
        y_train,
        selected_test_X,
        gridsearch=True,
    )
    performance_test_knn = accuracy_score(y_test, class_test_y)

    print("\tEntrenando Decision Tree...")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        selected_train_X,
        y_train,
        selected_test_X,
        gridsearch=True,
    )
    performance_test_dt = accuracy_score(y_test, class_test_y)

    print("\tEntrenando Naive Bayes...")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(
        selected_train_X,
        y_train,
        selected_test_X,
    )
    performance_test_nb = accuracy_score(y_test, class_test_y)

    # Guardar resultados en un DataFrame
    models = ["NN", "RF", "KNN", "DT", "NB"]
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_nn,
                performance_test_rf,
                performance_test_knn,
                performance_test_dt,
                performance_test_nb,
            ],
        }
    )
    score_df = pd.concat([score_df, new_scores])

# Ordenar los resultados por precisión
score_df = score_df.sort_values(ascending=False, by="accuracy")
print("Resultados de la búsqueda de modelos y selección de hiperparámetros:")
print(score_df)

# --------------------------------------------------------------
# Visualización de resultados
# --------------------------------------------------------------
plt.figure(figsize=(10, 5))
sns.barplot(data=score_df, x="model", y="accuracy", hue="feature_set")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0.7, 1)
plt.legend(loc="lower right")
plt.xticks(rotation=45)
plt.title("Model Accuracy Comparison by Feature Set")
plt.show()

# --------------------------------------------------------------
# Matriz de confusión personalizada
# --------------------------------------------------------------
(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.random_forest(
    X_train[feature_set_4], y_train, X_test[feature_set_4], gridsearch=True
)

accuracy = accuracy_score(y_test, class_test_y)
print(f"Accuracy: {accuracy:.4f}")

classes = np.unique(y_test)
cm = confusion_matrix(y_test, class_test_y, labels=classes)

# Normalización opcional
normalize_cm = False
if normalize_cm:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 5))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j], '.2f' if normalize_cm else 'd'),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )

plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.grid(False)
plt.tight_layout()
plt.show()

# Mostrar métricas adicionales
print(classification_report(y_test, class_test_y, target_names=[str(c) for c in classes]))


# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------

# Verificar columnas disponibles
print("Columnas disponibles en el DataFrame:", df.columns.tolist())

# Asegurarse de que las columnas existen antes de operar
required_columns = ["Label", "Participants", "Category", "Set"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise KeyError(f"Faltan las columnas necesarias: {missing_columns}")

# Crear subconjunto basado en participantes
participant_df = df.drop(["Category", "Set"], axis=1)

X_train = participant_df[participant_df["Participants"] != "A"].drop("Label", axis=1)
y_train = participant_df[participant_df["Participants"] != "A"]["Label"]

X_test = participant_df[participant_df["Participants"] == "A"].drop("Label", axis=1)
y_test = participant_df[participant_df["Participants"] == "A"]["Label"]

X_train = X_train.drop("Participants", axis=1)
X_test = X_test.drop("Participants", axis=1)

# Visualización de la distribución de clases
fig, ax = plt.subplots(figsize=(10, 5))
participant_df["Label"].value_counts().plot(
    kind="bar", color="lightblue", ax=ax, label="Total"
)
y_train.value_counts().plot(kind="bar", color="blue", ax=ax, label="Train")
y_test.value_counts().plot(kind="bar", color="darkblue", ax=ax, label="Test")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------
(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.random_forest(
    X_train[feature_set_4], y_train, X_test[feature_set_4], gridsearch=True
)

accuracy = accuracy_score(y_test, class_test_y)

classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 5))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()

# --------------------------------------------------------------
# Try a simpler model with the selected features
# --------------------------------------------------------------
(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.random_forest(
    X_train[selected_features],
    y_train,
    X_test[selected_features],
    gridsearch=False,
)

accuracy = accuracy_score(y_test, class_test_y)

classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 5))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()