import numpy as np

class FourierTransformation:
    # Transformación rápida de Fourier
    def find_fft_transformation(self, data, sampling_rate):
        transformation = np.fft.rfft(data, len(data))
        return transformation.real, transformation.imag

    # Abstracción de frecuencia
    def abstract_frequency(self, data_table, cols, window_size, sampling_rate):
        # Frecuencias a analizar
        freqs = np.round((np.fft.rfftfreq(int(window_size)) * sampling_rate), 3)

        for col in cols:
            data_table[col + "_max_freq"] = np.nan
            data_table[col + "_freq_weighted"] = np.nan
            data_table[col + "_pse"] = np.nan
            for freq in freqs:
                data_table[col + "freq" + str(freq) + "Hz_ws" + str(window_size)] = np.nan

        # Iterar sobre las filas del DataFrame
        for i in range(window_size, len(data_table.index)):
            for col in cols:
                real_ampl, imag_ampl = self.find_fft_transformation(
                    data_table[col].iloc[i - window_size : min(i + 1, len(data_table.index))],
                    sampling_rate,
                )
                # Calcular amplitudes
                for j in range(0, len(freqs)):
                    data_table.loc[
                        i, col + "freq" + str(freqs[j]) + "Hz_ws" + str(window_size)
                    ] = real_ampl[j]

                # Frecuencia máxima
                data_table.loc[i, col + "_max_freq"] = freqs[np.argmax(real_ampl)]

                # Frecuencia ponderada
                data_table.loc[i, col + "_freq_weighted"] = float(
                    np.sum(freqs * real_ampl)
                ) / np.sum(real_ampl)

                # Cálculo de Power Spectral Density (PSD)
                PSD = np.divide(np.square(real_ampl), float(len(real_ampl)))
                PSD_pdf = np.divide(PSD, np.sum(PSD))
                PSD_pdf = np.where(PSD_pdf > 0, PSD_pdf, 1e-12)  # Evitar ceros
                data_table.loc[i, col + "_pse"] = -np.sum(np.log(PSD_pdf) * PSD_pdf)

        return data_table
