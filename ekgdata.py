import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

# %% Objekt-Welt

# Klasse EKG-Data für Peakfinder, die uns ermöglicht peaks zu finden

class EKGdata:

## Konstruktor der Klasse soll die Daten einlesen

    def __init__(self, ekg_dict):
        self.ekg_dict = ekg_dict
        self.id = ekg_dict["id"]
        self.date = ekg_dict["date"]
        self.data = ekg_dict["result_link"]
        self.df = pd.read_csv(self.data, sep='\t', header=None, names=['Messwerte in mV','Zeit in ms',])
    
    @staticmethod
    def load_by_id(ekg_list, target_id):
        for ekg in ekg_list:
            if int(ekg["id"]) == int(target_id):  # <- sicherer Vergleich
                return ekg
        return None

    
    
    def find_peaks(self, threshold, respacing_factor=5):
        """
        A function to find the peaks in a series
        Args:
            - series (pd.Series): The series to find the peaks in
            - threshold (float): The threshold for the peaks
            - respacing_factor (int): The factor to respace the series
        Returns:
            - peaks (list): A list of the indices of the peaks
        """
        # Respace the series
        series_df = self.df["Messwerte in mV"]

        series = series_df.iloc[::respacing_factor]
        
        # Filter the series
        series = series[series>threshold]


        peaks = []
        last = 0
        current = 0
        next = 0

        for index, row in series.items():
            last = current
            current = next
            next = row

            if last < current and current > next and current > threshold:
                peaks.append(index-respacing_factor)

        return peaks

    @staticmethod
    def estimate_hr(peaks):
        if len(peaks) < 2:
            return 0  # Nicht genug Daten für Berechnung

        rr_intervals_ms = [peaks[i+1] - peaks[i] for i in range(len(peaks)-1)]
        mean_rr_ms = sum(rr_intervals_ms) / len(rr_intervals_ms)
        mean_rr_sec = mean_rr_ms / 1000.0
        hr = 60 / mean_rr_sec
        return hr
    
    def find_bradykardie(self):
        """
        Find episodes of bradycardia in the EKG data.
        Bradycardia is defined as a heart rate below 60 bpm.
        """
        # Get the peaks
        peaks = self.find_peaks(threshold=340)
        # Get the heart rate
        hr = self.estimate_hr(peaks)
        if hr < 60:
            return True
        return False
    
    def find_tachykardie(self):
        """
        Find episodes of tachycardia in the EKG data.
        Tachycardia is defined as a heart rate above 100 bpm.
        """
        # Get the peaks
        peaks = self.find_peaks(threshold=340)
        # Get the heart rate
        hr = self.estimate_hr(peaks)
        if hr > 100:
            return True
        return False
    
    def find_atrial_fibrillation(self):
        peaks = self.find_peaks(threshold=340)
        if len(peaks) < 3:
            return False
        
        rr_intervals = [peaks[i+1] - peaks[i] for i in range(len(peaks)-1)]

        rr_std = np.std(rr_intervals)

        # Schwellenwert empirisch wählen (z.B. >80 ms hohe Varianz = AFib-Verdacht)
        if rr_std > 80:
            return True
        return False
    
    def detect_st_elevation(self, threshold=0.1, st_offset_ms=60, baseline_offset_ms=120):
        peaks = self.find_peaks(threshold=340)
        st_shifts = []

        for peak_index in peaks:
            # Hole Zeit in ms des Peaks
            peak_time = self.df["Zeit in ms"].iloc[peak_index]

            # ST-Zeitpunkt ca. 60 ms nach R-Peak
            st_time = peak_time + st_offset_ms
            baseline_time = peak_time - baseline_offset_ms

            # Finde den Index, der der gewünschten Zeit am nächsten kommt
            try:
                st_idx = (self.df["Zeit in ms"] - st_time).abs().idxmin()
                baseline_idx = (self.df["Zeit in ms"] - baseline_time).abs().idxmin()

                st_value = self.df["Messwerte in mV"].iloc[st_idx]
                baseline_value = self.df["Messwerte in mV"].iloc[baseline_idx]

                shift = st_value - baseline_value
                st_shifts.append(shift)
            except:
                continue

        # Mittelwert der ST-Abweichung
        avg_shift = np.mean(st_shifts)

        print(f"Durchschnittliche ST-Abweichung: {avg_shift:.3f} mV")

        if avg_shift > threshold:
            return "ST-Hebung"
        elif avg_shift < -threshold:
            return "ST-Senkung"
        else:
            return "ST normal"
        
    def find_extrasystoles(self, threshold_ratio=0.7):
        peaks = self.find_peaks(threshold=340)
        rr_intervals = [peaks[i+1] - peaks[i] for i in range(len(peaks)-1)]

        if len(rr_intervals) < 3:
            return []

        # Berechne durchschnittliches RR-Intervall (ohne Ausreißer)
        mean_rr = np.median(rr_intervals)

        extrasystoles = []

        for i in range(1, len(rr_intervals)):
            if rr_intervals[i] < threshold_ratio * mean_rr:
                extrasystoles.append((peaks[i], rr_intervals[i]))

        return extrasystoles

    
    def plot_time_series(self, peaks, visible_range): # Visible Range soll im main.py eingegeben werden und dort umgesetzt werden
        start, end = visible_range
        self.df = pd.read_csv(self.data, sep='\t', header=None, names=['Messwerte in mV','Zeit in ms',])
        # Erstellte einen Line Plot, der ersten 2000 Werte mit der Zeit aus der x-Achse
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.df["Zeit in ms"].iloc[start:end],
            y=self.df["Messwerte in mV"].iloc[start:end],
            mode='lines',
            name='EKG Daten',
            line=dict(color='blue', width=2)
        ))

        peak_indices = [i for i in peaks if start <= i < end]
        fig.add_trace(go.Scatter(
            x=self.df["Zeit in ms"].iloc[peak_indices],
            y=self.df["Messwerte in mV"].iloc[peak_indices],
            mode='markers',
            name='Peaks',
            marker=dict(color='red', size=8, symbol='circle')
        ))
        extras = self.find_extrasystoles()
        visible_extras = [peak for peak in extras if start <= peak[0] < end]
        fig.add_trace(go.Scatter(
            x=[self.df["Zeit in ms"].iloc[peak[0]] for peak in visible_extras],
            y=[self.df["Messwerte in mV"].iloc[peak[0]] for peak in visible_extras],
            mode='markers',
            name='Extrasystolen',
            marker=dict(color='orange', size=10, symbol='x')
        ))

        fig.update_layout(
            title='EKG Zeitreihe mit Peaks',
            xaxis_title='Zeit in ms',
            yaxis_title='Messwerte in mV',
            template='plotly_white'
        )
        return fig
    

if __name__ == "__main__":
    #print("This is a module with some functions to read the EKG data")
    file = open("data/person_db.json")
    person_data = json.load(file)
    ekg_dict = person_data[0]["ekg_tests"][1]
    #print(ekg_dict)
    ekg = EKGdata(ekg_dict)
    #print(ekg.df.head())
    id = 1
    ekg_list = [person_data[0]["ekg_tests"][0], person_data[0]["ekg_tests"][1], person_data[1]["ekg_tests"][0], person_data[2]["ekg_tests"][0]]
    #print(EKGdata.load_by_id(ekg_list, id))
    all_data = EKGdata.load_by_id(ekg_list, id)
    threshold = 340
    peaks = ekg.find_peaks(threshold)
    #print(ekg.find_peaks(threshold))
    #print(EKGdata.estimate_hr(peaks))
    #print(EKGdata.plot_time_series(EKGdata, peaks))
    # print(ekg.find_bradykardie())
    # print(ekg.find_tachykardie())
    # print(ekg.find_atrial_fibrillation())
    # print(ekg.detect_st_elevation())
    print(ekg.find_extrasystoles())
    visible_range = (5000, 10000)  # Beispiel für sichtbaren Bereich
    print(ekg.plot_time_series(peaks, visible_range).show())