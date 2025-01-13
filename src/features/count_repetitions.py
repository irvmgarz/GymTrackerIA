import numpy as np
import pandas as pd
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("C:/Users/Gustavo Guzman/OneDrive/Escritorio/IA/GymTrackerIA/src/data/01_Data_Processed.pkl")

df = df[
    df["Label"] != "Rest"
]  ## since it doesn't make sense to count repetitions for rest label

# Calculating sum of squares:
Accelerometer_r = df["Accelerometer_x"] ** 2 + df["Accelerometer_y"] ** 2 + df["Accelerometer_z"] ** 2
Gyroscope_r = df["Gyroscope_x"] ** 2 + df["Gyroscope_y"] ** 2 + df["Gyroscope_z"] ** 2

df["Accelerometer_r"] = np.sqrt(Accelerometer_r)
df["Gyroscope_r"] = np.sqrt(Gyroscope_r)

# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------
bench_df = df[df["Label"] == "bench"]
ohp_df = df[df["Label"] == "ohp"]
squat_df = df[df["Label"] == "squat"]
dead_df = df[df["Label"] == "dead"]
row_df = df[df["Label"] == "row"]

# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------
fs = 1000 / 200
LowPass = LowPassFilter()

# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------

def counts_reps(dataset, cutoff=0.4, order=10, column="Accelerometer_r"):
    data = LowPass.low_pass_filter(
        dataset, col=column, sampling_frequency=fs, cutoff_frequency=cutoff, order=order
    )
    indexes = argrelextrema(data[column + "_lowpass"].values, np.greater)
    peaks = data.iloc[indexes]

    # Gráficos deshabilitados
    # fig, ax = plt.subplots()
    # plt.plot(data[column + "_lowpass"])
    # plt.scatter(x=peaks.index, y=peaks[column + "_lowpass"], c="r", s=80)
    # plt.ylabel(f"{column}_lowpass")
    # exercise = data["Label"].iloc[0].title()  # Corrige el acceso a Label
    # plt.title(f"{exercise}: {len(peaks)} Reps")
    # plt.show()

    return len(peaks)

# Usar la función corregida
counts_reps(bench_df, cutoff=0.4)
counts_reps(squat_df, cutoff=0.35)
counts_reps(row_df, cutoff=0.6, column="Gyroscope_r")
counts_reps(ohp_df, cutoff=0.35)
counts_reps(dead_df, cutoff=0.4)

# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------
df["reps"] = df["Category"].apply(lambda x: 5 if x == "heavy" else 10)

rep_df = df.groupby(["Label", "Category", "Set"])["reps"].max().reset_index()
rep_df["pred_reps"] = 0

for s in df.Set.unique():
    subset = df[df["Set"] == s]
    column = "Accelerometer_r"
    cutoff = 0.4

    if subset.iloc[0]["Label"] == "squat":
        cutoff = 0.35

    if subset.iloc[0]["Label"] == "row":
        cutoff = 0.6
        column = "Gyroscope_r"

    if subset.iloc[0]["Label"] == "ohp":
        cutoff = 0.35

    reps = counts_reps(subset, cutoff=cutoff, column=column)
    rep_df.loc[rep_df["Set"] == s, "pred_reps"] = reps

# Mostrar solo la tabla rep_df
print(rep_df)

# --------------------------------------------------------------
# Resultados Evaluados
# --------------------------------------------------------------
error = round(mean_absolute_error(rep_df["reps"], rep_df["pred_reps"]), 2)
print("MAE: ", error)

# Gráficos deshabilitados
# rep_df.groupby(["Label", "Category"])[["reps", "pred_reps"]].mean().plot(kind="bar")
# plt.show()
