import os
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, send_file
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error
from scipy.signal import butter, filtfilt

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

        # Limpieza de datos (eliminar \n y espacios en blanco)
        df = df.applymap(lambda x: str(x).replace('\n', '').strip() if isinstance(x, str) else x)

        # Aquí comienza tu lógica
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

        # Limpieza exhaustiva para evitar problemas
        rep_df = rep_df.replace({r'\n': '', r'\r': ''}, regex=True)
        rep_df = rep_df.applymap(lambda x: str(x).strip() if isinstance(x, str) else x)

        # Verificar si hay caracteres no deseados en el DataFrame (depuración)
        print(rep_df.head())  # Esto imprime los primeros registros para verificar

        error = round(mean_absolute_error(rep_df["reps"], rep_df["pred_reps"]), 2)
        output_file = os.path.join(app.config['UPLOAD_FOLDER'], "results.csv")
        rep_df.to_csv(output_file, index=False)

        return render_template('results.html', tables=[rep_df.to_html(classes='data', escape=False)], error=error, filename="results.csv")

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
