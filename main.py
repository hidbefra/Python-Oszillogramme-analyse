# vibe coding mit gemini :)
# Python 3.13.0 (tags/v3.13.0:60403a5, Oct  7 2024, 09:38:07) [MSC v.1941 64 bit (AMD64)] on win32
# pip install pandas matplotlib PyQt5 scipy

import pandas as pd
import numpy as np
import glob
import os
import matplotlib
matplotlib.use('Qt5Agg') # Ändert das Backend, um Tkinter-Probleme zu umgehen
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.optimize import minimize
from PyQt5.QtWidgets import QComboBox, QWidget, QHBoxLayout, QCompleter
from PyQt5.QtCore import Qt
from typing import Dict, Tuple

class PlotManager:
    """
    Verwaltet mehrere PlotNavigator-Instanzen, um sie synchron zu halten.
    """
    def __init__(self, navigators):
        self.navigators = navigators
        # Annahme: Alle Navigatoren arbeiten auf demselben Set von Dateien
        self.files = navigators[0].files if navigators else []
        self.current_index = 0

    def next(self, event):
        """Wechselt zur nächsten Datei in allen verwalteten Plots."""
        self.current_index = (self.current_index + 1) % len(self.files)
        self.update_all_plots()

    def prev(self, event):
        """Wechselt zur vorherigen Datei in allen verwalteten Plots."""
        self.current_index = (self.current_index - 1) % len(self.files)
        self.update_all_plots()

        # Rufe den Callback auf, um die ComboBox zu aktualisieren
        if hasattr(self, 'update_combo_callback') and callable(self.update_combo_callback):
            self.update_combo_callback()

    def update_all_plots(self):
        """Aktualisiert alle Plots, um die Daten des aktuellen Index anzuzeigen."""
        for nav in self.navigators:
            nav.current_index = self.current_index
            nav.update_plot()

        # Rufe den Callback auf, um die ComboBox zu aktualisieren, falls vorhanden
        # Dies ist nützlich für die direkte Auswahl über das Dropdown
        if hasattr(self, 'update_combo_callback') and callable(self.update_combo_callback):
            self.update_combo_callback()

class PlotNavigator:
    """
    Eine Klasse zur Verwaltung der interaktiven Navigation zwischen den Plots
    verschiedener CSV-Dateien.
    """
    def __init__(self, combined_df, ax1, ax2, ax4, time_col, volt_col, curr_col, features, global_limits):
        self.combined_df = combined_df
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax4 = ax4
        self.time_col = time_col
        self.volt_col = volt_col
        self.curr_col = curr_col
        self.power_col = 'Schaltleistung [W]'
        self.global_limits = global_limits
        self.features = features
        self.table = None
        self.simplified_volt_col = 'vereinfachte Spannung'
        self.files = self.combined_df['source_file'].unique()
        self.current_index = 0
        self._setup_axes()

        # Erstelle die Achsen für die Feature-Tabelle nur einmal.
        # [left, bottom, width, height]
        self.ax_table = plt.axes([0.125, 0.05, 0.2, 0.075], frame_on=False)
        self.ax_table.xaxis.set_visible(False)
        self.ax_table.yaxis.set_visible(False)

    def _setup_axes(self, target_value_for_one=20.0):
        """Konfiguriert die statischen Eigenschaften der Achsen einmalig."""
        volt_color = 'tab:blue'
        curr_color = 'tab:red'
        power_color = 'tab:purple'

        # Konfiguration für die erste Achse (Spannung, links)
        self.ax1.set_xlabel('Zeit [s]')
        self.ax1.set_ylabel('Spannung [V]', color=volt_color)
        self.ax1.tick_params(axis='y', labelcolor=volt_color)
        self.ax1.yaxis.set_ticks_position('left')
        self.ax1.grid(True)
        self.ax1.set_ylim(self.global_limits['volt'])

        # Konfiguration für die zweite Achse (Strom, rechts)
        self.ax2.set_ylabel('Strom [A]', color=curr_color)
        self.ax2.tick_params(axis='y', labelcolor=curr_color)
        self.ax2.yaxis.set_ticks_position('right')
        self.ax2.patch.set_visible(False)  # Mache den Hintergrund von ax2 transparent
        self.ax2.set_ylim(self.global_limits['curr'])

        # Konfiguration für die vierte Achse (Schaltleistung, rechts ganz außen)
        self.ax4.set_ylabel(self.power_col, color=power_color)
        self.ax4.tick_params(axis='y', labelcolor=power_color)
        self.ax4.spines['right'].set_position(('outward', 80)) # Positioniere die Achse noch weiter rechts
        self.ax4.patch.set_visible(False)
        self.ax4.set_ylim(self.global_limits['power'])

    def update_plot(self):
        """Löscht nur die Linien und zeichnet den Plot für den aktuellen Index neu."""


        # Entferne alte Linien, anstatt die ganzen Achsen zu löschen
        for line in self.ax1.lines:
            line.remove()
        # Die Linien von ax2, ax3, ax4 sind in ax1.lines enthalten, da sie twinx() sind.
        # Ein explizites Löschen ist nicht nötig und kann zu Fehlern führen.
        # Wir löschen sie trotzdem, um sicherzugehen, falls sich das Verhalten ändert.
        for line in self.ax2.lines:
            line.remove()
        for line in self.ax4.lines:
            line.remove()
        
        current_file = self.files[self.current_index]
        df_to_plot = self.combined_df[self.combined_df['source_file'] == current_file]

        # Überspringe das Plotten, wenn die Zeitspalte für diese Gruppe nur NaNs enthält
        if df_to_plot[self.time_col].isnull().all():
            self.ax1.set_title(f'Daten für {self.time_col} nicht verfügbar: {current_file}')
            return

        # Zeichne die neuen Daten
        line1 = self.ax1.plot(df_to_plot[self.time_col], df_to_plot[self.volt_col], color='tab:blue', label='Spannung [V]')
        line2 = self.ax2.plot(df_to_plot[self.time_col], df_to_plot[self.curr_col], color='tab:red', label='Strom [A]')
        line4 = self.ax4.plot(df_to_plot[self.time_col], df_to_plot[self.power_col], color='tab:purple', label=self.power_col, linestyle=':')

        # Hole die extrahierten Features für die aktuelle Datei
        current_features = self.features.get(current_file, {})
        
        # Erstelle die Daten für die Tabelle
        table_data = [
            [f"{value}"] for value in current_features.values()
        ]
        row_labels = list(current_features.keys())

        if table_data:
            # Leere die bestehenden Achsen der Tabelle und zeichne sie neu.
            self.ax_table.cla()
            self.table = self.ax_table.table(cellText=table_data, rowLabels=row_labels, loc='center')
            
        # Titel und Legende nur für den oberen Plot (ax1) setzen
        if self.ax1.get_figure().axes[0] == self.ax1:
            self.ax1.set_title(f'Oszillogramm von: {current_file} ({self.current_index + 1}/{len(self.files)})')
            # Platziere die Legende außerhalb des Plot-Bereichs
            self.ax1.legend(
                handles=line1 + line2 + line4, # line3 (vereinfachte Spannung) entfernt
                loc='lower center',
                bbox_to_anchor=(0.5, 1.05), # Positioniere sie über dem Plot
                ncol=3, frameon=False)
        plt.draw()

    def next(self, event):
        self.current_index = (self.current_index + 1) % len(self.files)
        self.update_plot()

    def prev(self, event):
        self.current_index = (self.current_index - 1) % len(self.files)
        self.update_plot()

def import_data(data_folder):
    """
    Sucht, lädt und kombiniert alle CSV-Dateien aus dem angegebenen Ordner.
    Gibt einen kombinierten DataFrame zurück oder None bei einem Fehler.
    """
    all_csv_files = glob.glob(os.path.join(data_folder, '*.csv'))
    if not all_csv_files:
        print(f"Keine CSV-Dateien im Ordner '{data_folder}' gefunden.")
        return None

    # Sortiere die Dateien, um eine konsistente Reihenfolge zu gewährleisten
    all_csv_files.sort()

    # Wähle nur jede 20. Datei aus
    csv_files_to_load = all_csv_files[::20]

    if not csv_files_to_load:
        print(f"Nach dem Filtern (jede 20. Datei) wurden keine Dateien zum Laden ausgewählt aus insgesamt {len(all_csv_files)} Dateien.")
        return None

    print(f"{len(all_csv_files)} CSV-Dateien gefunden. Lade {len(csv_files_to_load)} davon (jede 20.)...")

    list_of_dfs = []
    for file in csv_files_to_load:
        try:
            df = pd.read_csv(file, skiprows=11, header=None, names=['Zeit[s]', 'Strom[A]', 'Spannung[V]'], usecols=[0, 1, 2])
            df['source_file'] = os.path.basename(file)
            list_of_dfs.append(df)
        except Exception as e:
            print(f"Fehler beim Lesen der Datei {file}: {e}")

    if not list_of_dfs:
        print("Konnte keine Daten aus den CSV-Dateien laden.")
        return None

    return pd.concat(list_of_dfs, ignore_index=True)

def process_data(df):
    """
    Führt grundlegende Analysen und Verarbeitungen auf dem DataFrame durch.
    """
    print("\n--- Datenübersicht ---")
    print("Erste 5 Zeilen der kombinierten Daten:")
    print(df.head())

    print("\nStatistische Zusammenfassung der numerischen Spalten:")
    print(df.describe())

def shift_time_axis(df, time_col, curr_col, threshold=5.0):
    """
    Verschiebt die Zeitachse für jede Messung (jede Datei) so,
    dass der Nullpunkt (t=0) an der Stelle liegt, an der der Strom
    erstmals den Schwellenwert (absolut) überschreitet.

    Gibt einen neuen DataFrame mit der verschobenen Zeitachse zurück.
    """
    print(f"\n--- Zeitachse anpassen ---")
    print(f"Verschiebe den Zeit-Nullpunkt zum ersten Ereignis, bei dem |Strom| > {threshold}A.")

    processed_groups = []
    for file_name, group in df.groupby('source_file'):
        group_copy = group.copy()
        # Finde den ersten Index, wo der absolute Stromwert den Schwellenwert überschreitet
        trigger_index = group_copy[abs(group_copy[curr_col]) > threshold].first_valid_index()

        if trigger_index is not None:
            # Hole den Zeitwert an diesem Index
            t_zero = group_copy.loc[trigger_index, time_col]
            # Verschiebe die gesamte Zeitspalte für diese Gruppe
            group_copy[time_col] = group_copy[time_col] - t_zero
        else:
            print(f"Warnung: In '{file_name}' wurde der Schwellenwert von {threshold}A nie überschritten. Zeitachse nicht verschoben.")
        processed_groups.append(group_copy)
    return pd.concat(processed_groups, ignore_index=True)

def optimize_and_correct_voltage(df, time_col, volt_col, curr_col, time_range=(0.01, 0.03)):
    """
    Optimiert Widerstand und Offset, um die Spannung in einem Zeitfenster zu minimieren,
    und wendet die Korrektur auf den gesamten Datensatz an.
    """
    print(f"\n--- Spannungskorrektur durchführen ---")
    print(f"Optimiere R und Offset, um die Spannung im Zeitbereich {time_range}s zu minimieren.")

    processed_groups = []
    for file_name, group in df.groupby('source_file'):
        group_copy = group.copy()
        # 1. Daten für die Optimierung im Zeitfenster filtern
        opt_window = group_copy[(group_copy[time_col] >= time_range[0]) & (group_copy[time_col] <= time_range[1])]

        if opt_window.empty:
            print(f"Warnung: Für '{file_name}' keine Daten im Optimierungsfenster {time_range}s gefunden. Spannung nicht korrigiert.")

        # 2. Fehlerfunktion definieren, die wir minimieren wollen
        # params wird ein Array [Widerstand, Offset] sein
        def error_func(params, voltage, current):
            resistance, offset = params
            corrected_voltage = voltage + current * resistance + offset
            # Wir minimieren die Summe der Quadrate der Abweichungen von Null
            return np.sum(corrected_voltage**2)

        # 3. Optimierung durchführen
        initial_guess = [0.0, 0.0] # Startwerte für [Widerstand, Offset]
        result = minimize(
            fun=error_func,
            x0=initial_guess,
            args=(opt_window[volt_col], opt_window[curr_col]),
            method='Nelder-Mead' # Ein einfacher und robuster Algorithmus
        )

        # 4. Optimale Parameter extrahieren und Korrektur anwenden
        optimal_resistance, optimal_offset = result.x
        print(f"  - Datei: {file_name}, Opt. R={optimal_resistance:.4f} Ohm, Opt. Offset={optimal_offset:.4f} V")

        # Wende die Korrektur auf die *gesamte* Gruppe an
        group_copy[volt_col] = group_copy[volt_col] + group_copy[curr_col] * optimal_resistance + optimal_offset
        processed_groups.append(group_copy)
    return pd.concat(processed_groups, ignore_index=True)

def add_simplified_voltage_column(df, volt_col):
    """
    Fügt eine neue Spalte 'vereinfachte Spannung' hinzu.
    Der Wert ist 1, wenn die Spannung > 1V oder < -1V ist, ansonsten 0.
    """
    new_col_name = 'vereinfachte Spannung'
    print(f"\n--- Spalte '{new_col_name}' hinzufügen ---")

    # Bedingung: Spannung > 1 ODER Spannung < -1
    condition = (df[volt_col] > 1) | (df[volt_col] < -1)

    # np.where ist effizient: wenn Bedingung wahr, dann 1, sonst 0
    df[new_col_name] = np.where(condition, 1, 0)
    print(f"Spalte '{new_col_name}' wurde hinzugefügt.")
    return df

def add_power_column(df, volt_col, curr_col):
    """
    Fügt eine neue Spalte 'Schaltleistung [W]' hinzu.
    Die Leistung wird nur berechnet, wenn der Betrag des Stroms > 0.3A und der Betrag der Spannung > 0.3V ist, sonst ist sie 0.
    """
    new_col_name = 'Schaltleistung [W]'
    curr_threshold = 0.3
    volt_threshold = 0.8
    print(f"\n--- Spalte '{new_col_name}' hinzufügen (nur wenn |Strom| > {curr_threshold}A und |Spannung| > {volt_threshold}V) ---")

    # Berechne die Leistung nur, wenn Strom und Spannung über ihren jeweiligen Schwellenwerten liegen, sonst 0
    condition = (df[curr_col].abs() > curr_threshold) & (df[volt_col].abs() > volt_threshold)
    df[new_col_name] = np.where(condition, df[volt_col] * df[curr_col], 0)
    print(f"Spalte '{new_col_name}' wurde hinzugefügt.")
    return df

def extract_features(df, time_col, opener_time_col, simplified_volt_col, power_col, volt_col, prellen_time_range=(0.0, 0.02), ausschalt_time_range=(0.0, 0.02)):
    """
    Extrahiert spezifische Features (Prellen, Schaltarbeit, Ausschaltarbeit) für jede Datei.
    """
    print(f"\n--- Extrahiere Features ---")
    print(f"  - 'Prellen' und 'Schaltarbeit' im Zeitbereich {prellen_time_range}s (auf '{time_col}')")
    print(f"  - 'Ausschaltarbeit' im Zeitbereich {ausschalt_time_range}s (auf '{opener_time_col}')")
    print(f"  - 'Prelldauer' im Zeitbereich {prellen_time_range}s (auf '{time_col}')")
    print(f"  - 'Lichtbogendauer' im Zeitbereich {ausschalt_time_range}s (auf '{opener_time_col}')")

    features = {}

    # Gruppiere nach Datei und wende die Analyse an
    for file_name, group in df.groupby('source_file'):
        file_features = {}

        # --- Features beim Einschalten ---
        prellen_window = group[(group[time_col] >= prellen_time_range[0]) & (group[time_col] <= prellen_time_range[1])]

        if prellen_window.empty:
            file_features['Prellen'] = 0
            file_features['Prelldauer [s]'] = 0.0
            file_features['Schaltarbeit [Ws]'] = 0.0
        else:
            # Prellen (Anzahl der 1->0 Wechsel)
            transitions = prellen_window[simplified_volt_col].diff()
            count_1_to_0 = (transitions == -1).sum()
            file_features['Prellen'] = count_1_to_0

            # Prelldauer (Zeit des letzten 1->0 Wechsels)
            last_bounce_indices = prellen_window.index[transitions == -1]
            if not last_bounce_indices.empty:
                file_features['Prelldauer [s]'] = round(prellen_window.loc[last_bounce_indices[-1], time_col], 5)
            else:
                file_features['Prelldauer [s]'] = 0.0


            # Schaltarbeit (Integral der Leistung über die Zeit)
            if len(prellen_window) > 1:
                switching_work = np.trapezoid(y=prellen_window[power_col].abs(), x=prellen_window[time_col])
            else:
                switching_work = 0.0
            # Runde auf 4 signifikante Stellen
            if switching_work != 0:
                switching_work = round(switching_work, 4 - 1 - int(np.floor(np.log10(abs(switching_work)))))
            file_features['Schaltarbeit [Ws]'] = switching_work

        # --- Features beim Ausschalten ---
        ausschalt_window = group[(group[opener_time_col] >= ausschalt_time_range[0]) & (group[opener_time_col] <= ausschalt_time_range[1])]

        if ausschalt_window.empty or ausschalt_window[opener_time_col].isnull().all():
            turn_off_work = 0.0
            arc_duration = 0.0
        else:
            # Ausschaltarbeit
            if len(ausschalt_window) > 1:
                turn_off_work = np.trapezoid(y=ausschalt_window[power_col].abs(), x=ausschalt_window[opener_time_col])
            else:
                turn_off_work = 0.0
            
            # Lichtbogendauer (erster Nulldurchgang der Spannung)
            # Finde, wo sich das Vorzeichen ändert (Produkt ist negativ)
            zero_crossings = ausschalt_window.index[
                (ausschalt_window[volt_col].shift(-1) * ausschalt_window[volt_col]) < 0
            ]
            if not zero_crossings.empty:
                arc_duration = round(ausschalt_window.loc[zero_crossings[0], opener_time_col], 5)
            else:
                arc_duration = 0.0

        # Runde auf 4 signifikante Stellen
        if turn_off_work != 0:
            turn_off_work = round(turn_off_work, 4 - 1 - int(np.floor(np.log10(abs(turn_off_work)))))
        file_features['AusSchaltarbeit [Ws]'] = turn_off_work
        file_features['Lichtbogendauer [s]'] = arc_duration

        features[file_name] = file_features
        print(
            f"  - Datei: {file_name}, "
            f"Prellen: {file_features.get('Prellen', 0)}, "
            f"Prelldauer: {file_features.get('Prelldauer [s]', 0.0):.5f} s, "
            f"Schaltarbeit: {file_features.get('Schaltarbeit [Ws]', 0.0):.4f} Ws, "
            f"Ausschaltarbeit: {file_features.get('AusSchaltarbeit [Ws]', 0.0):.4f} Ws, "
            f"Lichtbogendauer: {file_features.get('Lichtbogendauer [s]', 0.0):.5f} s"
        )

    return features

def add_opener_time_column(df, time_col, volt_col, search_start_time=0.4, voltage_threshold=1.0):
    """
    Fügt eine neue Spalte 'öffner Zeit [s]' hinzu.
    Der Nullpunkt wird an die Stelle gesetzt, an der die absolute Spannung
    nach einer Startzeit zum ersten Mal einen Schwellenwert überschreitet.
    """
    new_col_name = 'öffner Zeit [s]'
    print(f"\n--- Spalte '{new_col_name}' hinzufügen ---")
    print(f"Suche nach |{volt_col}| > {voltage_threshold}V ab t={search_start_time}s.")

    processed_groups = []
    for file_name, group in df.groupby('source_file'):
        group_copy = group.copy()
        # Filtere die Gruppe für den Suchbereich
        search_window = group_copy[group_copy[time_col] >= search_start_time]

        # Finde den ersten Index, an dem die Bedingung erfüllt ist
        trigger_index = search_window[search_window[volt_col].abs() > voltage_threshold].first_valid_index()

        if trigger_index is not None:
            # Hole den Zeitwert am Trigger-Punkt
            t_zero_opener = group_copy.loc[trigger_index, time_col]
            # Berechne die neue Zeitspalte für die gesamte Gruppe
            group_copy[new_col_name] = group_copy[time_col] - t_zero_opener
        else:
            # Wenn das Ereignis nicht gefunden wird, fülle die Spalte mit NaN
            print(f"Warnung: In '{file_name}' wurde der Schwellenwert von |{voltage_threshold}|V nach {search_start_time}s nicht gefunden. '{new_col_name}' wird mit NaN gefüllt.")
            group_copy[new_col_name] = np.nan
        processed_groups.append(group_copy)
    return pd.concat(processed_groups, ignore_index=True)

def calculate_and_align_global_limits(df: pd.DataFrame, time_col: str, opener_time_col: str, volt_col: str, curr_col: str, power_col: str) -> Dict[str, Tuple[float, float]]:
    """
    Berechnet die globalen, symmetrischen Achsen-Limits für alle Plots.
    """
    print("\n--- Berechne globale Achsen-Limits ---")
    
    opener_time_limits = (df[opener_time_col].min(), df[opener_time_col].max())
    
    def get_symmetric_limits(series: pd.Series, padding: float = 1.05) -> Tuple[float, float]:
        """Berechnet symmetrische Limits um 0."""
        max_abs = series.abs().max() * padding
        if pd.isna(max_abs) or max_abs == 0: # pragma: no cover
            return -1, 1
        return -max_abs, max_abs

    limits = {
        'volt': (-815, 815), #get_symmetric_limits(df[volt_col]),
        # Die Zeitachse wird nicht symmetrisch um 0 benötigt.
        'time': (df[time_col].min(), df[time_col].max()),
        'opener_time': opener_time_limits if not np.any(pd.isna(opener_time_limits)) else (df[time_col].min(), df[time_col].max()),
        'curr': (-70, 70), #get_symmetric_limits(df[curr_col]),
        'power': (-200, 200) #get_symmetric_limits(df[power_col])
    }
    print(f"  - Spannungs-Limits: ({limits['volt'][0]:.2f}, {limits['volt'][1]:.2f}) V")
    print(f"  - Strom-Limits: ({limits['curr'][0]:.2f}, {limits['curr'][1]:.2f}) A")
    print(f"  - Leistungs-Limits: ({limits['power'][0]:.2f}, {limits['power'][1]:.2f}) W")
    
    return limits

def create_feature_summary_plot(features: Dict):
    """
    Erstellt einen separaten Plot, der alle extrahierten Features über alle Dateien zusammenfasst.
    """
    if not features:
        print("Keine Features zum Plotten vorhanden.")
        return

    # Konvertiere das Feature-Dictionary in einen DataFrame für einfaches Plotten
    features_df = pd.DataFrame.from_dict(features, orient='index')

    # Konvertiere Spalten, die numerisch sein sollten, aber als Strings gespeichert sind
    for col in features_df.columns:
        # Konvertiere alle Spalten, die keine reinen Strings sein sollen, in numerische Werte
        features_df[col] = pd.to_numeric(features_df[col], errors='coerce')

    num_features = len(features_df.columns)
    fig, axes = plt.subplots(nrows=num_features, ncols=1, figsize=(15, 5 * num_features), sharex=True)
    fig.canvas.manager.set_window_title('Feature-Zusammenfassung')
    fig.suptitle('Zusammenfassung der extrahierten Features', fontsize=16)

    # Stelle sicher, dass 'axes' immer ein Array ist, auch bei nur einem Feature
    if num_features == 1:
        axes = [axes]

    for i, feature_name in enumerate(features_df.columns):
        ax = axes[i]

        # Wenn das Feature 'Schaltarbeit' oder 'Ausschaltarbeit' ist, erstelle einen Boxplot
        if any(keyword in feature_name.lower() for keyword in ['arbeit', 'prellen', 'dauer']):
            group_size = 25
            data_series = features_df[feature_name].dropna()

            # Erstelle Gruppen von 50 Messungen
            grouped_data = [data_series[i:i + group_size] for i in range(0, len(data_series), group_size)]
            
            # Erstelle Labels für die x-Achse, die den Bereich der Messungen anzeigen
            labels = [f"{i*group_size+1}-{(i+1)*group_size}" for i in range(len(grouped_data))]
            
            ax.boxplot(grouped_data, tick_labels=labels)
            ax.set_title(f"Boxplot für: {feature_name}")

        # Für alle anderen Features, behalte den Linienplot bei
        else:
            ax.plot(features_df.index, features_df[feature_name], marker='o', linestyle='-')
            ax.set_title(f"Verlauf für: {feature_name}")

        ax.set_ylabel(feature_name)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.tick_params(axis='x', labelrotation=90)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Platz für Titel und rotierte Labels

def create_interactive_plot(df, time_col, opener_time_col, volt_col, curr_col, features, global_limits):
    """
    Erstellt und zeigt einen interaktiven Plot mit Navigationsbuttons an.
    Erstellt zwei synchronisierte Fenster für unterschiedliche Zeitachsen.
    """
    # Erstelle eine Figur mit zwei Subplots übereinander.
    # `sharex=False`, da die Zeitachsen unterschiedliche Skalen haben.
    fig, (ax1, ax1_2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    fig.canvas.manager.set_window_title('Analyse-Plot')
    # Passe das Layout an, um Platz für Buttons, Legenden und Achsenbeschriftungen zu schaffen
    fig.subplots_adjust(bottom=0.15, right=0.85, top=0.92, hspace=0.4)

    # --- Setup für den oberen Plot (Haupt-Zeitachse) ---
    ax2_top = ax1.twinx()
    ax4_top = ax1.twinx()

    # Erstelle den Navigator für den oberen Plot
    # ax3 wird auf None gesetzt, um die vereinfachte Spannung nicht zu plotten
    navigator1 = PlotNavigator(df, ax1, ax2_top, ax4_top, time_col, volt_col, curr_col, features, global_limits)
    ax1.set_xlim(global_limits['time'])

    # --- Setup für den unteren Plot (Öffner-Zeitachse) ---
    ax2_bottom = ax1_2.twinx()
    ax4_bottom = ax1_2.twinx()

    # Erstelle den Navigator für den unteren Plot
    navigator2 = PlotNavigator(df, ax1_2, ax2_bottom, ax4_bottom, opener_time_col, volt_col, curr_col, features, global_limits)
    ax1_2.set_xlim(global_limits['opener_time'])
    ax1_2.set_xlabel(opener_time_col) # Überschreibe das Standard-Label

    # --- Manager und Buttons ---
    # Erstelle einen Manager, der beide Navigatoren synchronisiert
    plot_manager = PlotManager([navigator1, navigator2])

    # --- Dropdown-Menü (ComboBox) mit Qt hinzufügen ---
    # Hole die Werkzeugleiste des Qt-Fensters
    toolbar = fig.canvas.manager.toolbar
    
    # Erstelle die ComboBox
    combo_box = QComboBox()
    # Setze eine Mindestbreite, damit die Box größer ist und mehr Text anzeigt
    combo_box.setMinimumWidth(350)
    combo_box.addItems(plot_manager.files)
    combo_box.setEditable(True) # Erlaubt das Tippen
    combo_box.setInsertPolicy(QComboBox.NoInsert)
    combo_box.completer().setCompletionMode(QCompleter.PopupCompletion)
    combo_box.completer().setFilterMode(Qt.MatchContains) # Filtert bei Eingabe
    combo_box.completer().setCaseSensitivity(Qt.CaseInsensitive) # Groß-/Kleinschreibung ignorieren

    # Funktion, die aufgerufen wird, wenn ein Element ausgewählt wird
    def on_combo_select(index):
        if plot_manager.current_index != index:
            plot_manager.current_index = index
            plot_manager.update_all_plots()

    # Verbinde die Auswahlfunktion mit dem Signal der ComboBox
    combo_box.activated.connect(on_combo_select)

    # Füge die ComboBox zur Werkzeugleiste hinzu
    # Wir erstellen ein temporäres Widget, um die ComboBox hinzuzufügen
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.addWidget(combo_box)
    toolbar.addWidget(container)

    # Füge die Buttons am unteren Rand der Figur hinzu
    ax_prev = fig.add_axes([0.7, 0.02, 0.1, 0.05])
    ax_next = fig.add_axes([0.81, 0.02, 0.1, 0.05])

    b_prev = Button(ax_prev, 'Zurück')
    b_next = Button(ax_next, 'Weiter')

    # Verbinde die Buttons mit dem PlotManager
    b_prev.on_clicked(plot_manager.prev)
    b_next.on_clicked(plot_manager.next)

    # Zeichne beide Plots initial
    plot_manager.update_all_plots()

    # Funktion, um die ComboBox zu aktualisieren, wenn Buttons geklickt werden
    def update_combo_on_nav():
        combo_box.setCurrentIndex(plot_manager.current_index)
    plot_manager.update_combo_callback = update_combo_on_nav

    # Gib die Button-Objekte zurück, damit sie nicht vom Garbage Collector entfernt werden
    return b_prev, b_next

def analyze_oscilloscope_data(data_folder):
    """
    Lädt alle CSV-Dateien aus einem Ordner, kombiniert sie, führt eine
    Basisanalyse durch und visualisiert das erste Signal.
    """
    # Annahme: Deine CSVs haben Spalten namens 'Time' und 'Voltage'.
    # Passe diese Namen bei Bedarf an.
    time_col = 'Zeit[s]'
    volt_col = 'Spannung[V]'
    curr_col = 'Strom[A]'
    opener_time_col = 'öffner Zeit [s]'

    # 1. Daten importieren
    combined_df = import_data(data_folder)
    if combined_df is None:
        return

    # 2. Zeitachse verschieben (damit das Optimierungsfenster konsistent ist)
    shifted_df = shift_time_axis(combined_df, time_col, curr_col, threshold=5.0)

    # 3. Spannungswerte optimieren und korrigieren
    corrected_df = optimize_and_correct_voltage(shifted_df, time_col, volt_col, curr_col)

    # 4. Spalte "vereinfachte Spannung" hinzufügen
    final_df = add_simplified_voltage_column(corrected_df, volt_col)

    # 5. Spalte "öffner Zeit [s]" hinzufügen
    final_df = add_opener_time_column(final_df, time_col, volt_col)

    # 6. Spalte "Schaltleistung [W]" hinzufügen
    final_df = add_power_column(final_df, volt_col, curr_col)

    # 7. Spezifische Features für jede Datei extrahieren
    features = extract_features(final_df, time_col, opener_time_col, 'vereinfachte Spannung', 'Schaltleistung [W]', volt_col)

    # 8. Filtere Messungen basierend auf den extrahierten Features
    print("\n--- Filtere Messungen mit hoher Schalt-/Ausschaltarbeit ---")
    work_threshold = 100.0
    files_to_remove = set()
    for file_name, feature_values in features.items():
        # Konvertiere die String-Werte in Floats für den Vergleich
        schaltarbeit = float(feature_values.get('Schaltarbeit [Ws]', 0.0))
        ausschaltarbeit = float(feature_values.get('AusSchaltarbeit [Ws]', 0.0))
        if schaltarbeit > work_threshold or ausschaltarbeit > work_threshold:
            files_to_remove.add(file_name)

    if files_to_remove:
        print(f"Entferne {len(files_to_remove)} Messungen mit Arbeit > {work_threshold} [Ws]:")
        for file_name in sorted(list(files_to_remove)):
            print(f"  - {file_name}")
        
        # Filtere das 'features'-Dictionary und den 'final_df'-DataFrame
        features = {k: v for k, v in features.items() if k not in files_to_remove}
        final_df = final_df[~final_df['source_file'].isin(files_to_remove)].copy()
    else:
        print(f"Keine Messungen überschreiten den Schwellenwert von {work_threshold} Ws. Alle Daten werden beibehalten.")

    # 9. Globale, ausgerichtete Achsen-Limits für alle Plots berechnen
    global_limits = calculate_and_align_global_limits(final_df, time_col, opener_time_col, volt_col, curr_col, 'Schaltleistung [W]')

    # 10. Allgemeine Daten verarbeiten/analysieren (mit den finalen Werten)
    process_data(final_df)

    # 11. Feature-Zusammenfassungs-Plot erstellen
    create_feature_summary_plot(features)

    # 12. Interaktiven Analyse-Plot erstellen und Referenzen auf die Buttons halten
    button_references = None
    if all(col in final_df.columns for col in [time_col, volt_col, curr_col]):
        button_references = create_interactive_plot(final_df, time_col, opener_time_col, volt_col, curr_col, features, global_limits)

    # 12. Alle erstellten Plots gleichzeitig anzeigen
    plt.show()

def main():
    """Hauptfunktion, die die Analyse startet."""
    data_directory = 'data'
    if not os.path.exists(data_directory) or not os.listdir(data_directory):
        os.makedirs(data_directory, exist_ok=True)
        print(f"Der Ordner '{data_directory}' ist leer. Bitte lege deine CSV-Dateien dort ab und starte das Skript erneut.")
        return
    analyze_oscilloscope_data(data_directory)

if __name__ == "__main__":
    main()