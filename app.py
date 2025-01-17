import os
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, send_file, send_from_directory
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use('Agg')  # Solución para evitar errores relacionados con GUI
import matplotlib.pyplot as plt

class LowPassFilter:
    def low_pass_filter(self, dataset, col, sampling_frequency, cutoff_frequency, order):
        nyquist = 0.5 * sampling_frequency
        normal_cutoff = cutoff_frequency / nyquist
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        dataset[col + "_lowpass"] = filtfilt(b, a, dataset[col])
        return dataset

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def upload_file():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Cargar y procesar el archivo CSV
        df = pd.read_csv(filepath)
        df = df.applymap(lambda x: str(x).replace('\n', '').strip() if isinstance(x, str) else x)

        # Procesamiento de los datos
        df = df[df["Label"] != "Rest"]
        Accelerometer_r = df["Accelerometer_x"] ** 2 + df["Accelerometer_y"] ** 2 + df["Accelerometer_z"] ** 2
        Gyroscope_r = df["Gyroscope_x"] ** 2 + df["Gyroscope_y"] ** 2 + df["Gyroscope_z"] ** 2
        df["Accelerometer_r"] = np.sqrt(Accelerometer_r)
        df["Gyroscope_r"] = np.sqrt(Gyroscope_r)

        def counts_reps(dataset, cutoff=0.4, order=10, column="Accelerometer_r"):
            LowPass = LowPassFilter()
            fs = 1000 / 200
            data = LowPass.low_pass_filter(
                dataset, col=column, sampling_frequency=fs, cutoff_frequency=cutoff, order=order
            )
            indexes = argrelextrema(data[column + "_lowpass"].values, np.greater)
            peaks = data.iloc[indexes]
            return len(peaks)

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

        error = round(mean_absolute_error(rep_df["reps"], rep_df["pred_reps"]), 2)

        # Verificar y convertir columnas a numéricas
        rep_df["reps"] = pd.to_numeric(rep_df["reps"], errors="coerce")
        rep_df["pred_reps"] = pd.to_numeric(rep_df["pred_reps"], errors="coerce")

        # Calcular  de repeticiones por ejercicio
        rep_df_grouped = rep_df.groupby("Label")[["reps", "pred_reps"]].mean()

        # Generar gráfico de  de repeticiones por ejercicio
        plt.figure(figsize=(10, 6))
        plt.bar(rep_df_grouped.index, rep_df_grouped["pred_reps"], label="Repeticiones Predichas ", alpha=0.7, color='blue')
        plt.bar(rep_df_grouped.index, rep_df_grouped["reps"], label="Repeticiones Reales ", alpha=0.7, color='orange')
        plt.xlabel("Ejercicio")
        plt.ylabel("Repeticiones ")
        plt.title("Repeticiones Reales vs Predichas por Ejercicio")
        plt.legend()
        graph_path = os.path.join(app.config['UPLOAD_FOLDER'], "reps_comparison_avg.png")
        plt.savefig(graph_path)
        plt.close()

        return render_template('results.html', error=error, graph_filename="reps_comparison_avg.png")

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

@app.route('/uploads/<filename>')
def serve_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
