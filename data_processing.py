# data_processing.py
"""
Modul für den Import, die Verarbeitung und die Feature-Extraktion von Oszillogrammdaten.
"""
import glob
import os
import numpy as np
import pandas as pd
from typing import Tuple
from scipy.optimize import minimize

from config_loader import settings as config


def import_data(data_folder: str):
    """
    Sucht, lädt und kombiniert alle CSV-Dateien aus dem konfigurierten Ordner.
    Gibt einen kombinierten DataFrame zurück oder None bei einem Fehler.
    """
    all_csv_files = glob.glob(os.path.join(data_folder, '*.csv'))
    if not all_csv_files:
        print(f"Keine CSV-Dateien im Ordner '{data_folder}' gefunden.")
        return None

    all_csv_files.sort()
    sampling_rate = config.IMPORT_CONFIG["file_sampling_rate"]
    csv_files_to_load = all_csv_files[::sampling_rate]

    if not csv_files_to_load:
        print(f"Nach dem Filtern (jede {sampling_rate}. Datei) wurden keine Dateien zum Laden ausgewählt.")
        return None

    print(f"{len(all_csv_files)} CSV-Dateien gefunden. Lade {len(csv_files_to_load)} davon...")

    list_of_dfs = []
    for file in csv_files_to_load:
        try:
            df = pd.read_csv(
                file,
                skiprows=config.IMPORT_CONFIG["skiprows"],
                header=config.IMPORT_CONFIG["header"],
                names=[config.COLUMN_NAMES["time"], config.COLUMN_NAMES["current"], config.COLUMN_NAMES["voltage"]],
                usecols=config.IMPORT_CONFIG["usecols"]
            )
            df[config.COLUMN_NAMES["source_file"]] = os.path.basename(file)
            list_of_dfs.append(df)
        except Exception as e:
            print(f"Fehler beim Lesen der Datei {file}: {e}")

    if not list_of_dfs:
        print("Konnte keine Daten aus den CSV-Dateien laden.")
        return None

    return pd.concat(list_of_dfs, ignore_index=True)


def shift_time_axis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verschiebt die Zeitachse für jede Messung so, dass t=0 an der Stelle liegt,
    an der der Strom erstmals den Schwellenwert überschreitet.
    """
    time_col = config.COLUMN_NAMES["time"]
    curr_col = config.COLUMN_NAMES["current"]
    threshold = config.ANALYSIS_CONFIG["time_shift_current_threshold"]
    
    print(f"\n--- Zeitachse anpassen (t=0 bei |Strom| > {threshold}A) ---")

    processed_groups = []
    for file_name, group in df.groupby(config.COLUMN_NAMES["source_file"]):
        group_copy = group.copy()
        trigger_index = group_copy[abs(group_copy[curr_col]) > threshold].first_valid_index()

        if trigger_index is not None:
            t_zero = group_copy.loc[trigger_index, time_col]
            group_copy[time_col] = group_copy[time_col] - t_zero
        else:
            print(f"Warnung: In '{file_name}' wurde der Strom-Schwellenwert nie überschritten. Zeitachse nicht verschoben.")
        processed_groups.append(group_copy)
    return pd.concat(processed_groups, ignore_index=True)


def optimize_and_correct_voltage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimiert R und Offset zur Minimierung der Spannung in einem Zeitfenster
    und wendet die Korrektur auf den gesamten Datensatz an.
    """
    time_col = config.COLUMN_NAMES["time"]
    volt_col = config.COLUMN_NAMES["voltage"]
    curr_col = config.COLUMN_NAMES["current"]
    time_range = config.ANALYSIS_CONFIG["voltage_correction_time_range"]
    
    print(f"\n--- Spannungskorrektur durchführen (im Zeitbereich {time_range}s) ---")

    processed_groups = []
    for file_name, group in df.groupby(config.COLUMN_NAMES["source_file"]):
        group_copy = group.copy()
        opt_window = group_copy[(group_copy[time_col] >= time_range[0]) & (group_copy[time_col] <= time_range[1])]

        if opt_window.empty:
            print(f"Warnung: Für '{file_name}' keine Daten im Optimierungsfenster gefunden. Spannung nicht korrigiert.")
            processed_groups.append(group_copy)
            continue

        def error_func(params, voltage, current):
            resistance, offset = params
            corrected_voltage = voltage + current * resistance + offset
            return np.sum(corrected_voltage**2)

        initial_guess = [0.0, 0.0]
        result = minimize(
            fun=error_func,
            x0=initial_guess,
            args=(opt_window[volt_col], opt_window[curr_col]),
            method='Nelder-Mead'
        )

        optimal_resistance, optimal_offset = result.x
        print(f"  - Datei: {file_name}, Opt. R={optimal_resistance:.4f} Ohm, Opt. Offset={optimal_offset:.4f} V")

        group_copy[volt_col] = group_copy[volt_col] + group_copy[curr_col] * optimal_resistance + optimal_offset
        processed_groups.append(group_copy)
    return pd.concat(processed_groups, ignore_index=True)


def add_simplified_voltage_column(df: pd.DataFrame) -> pd.DataFrame:
    """Fügt eine binarisierte Spannungsspalte hinzu."""
    volt_col = config.COLUMN_NAMES["voltage"]
    new_col_name = config.COLUMN_NAMES["simplified_voltage"]
    threshold = config.ANALYSIS_CONFIG["simplified_voltage_threshold"]
    
    print(f"\n--- Spalte '{new_col_name}' hinzufügen (|Spannung| > {threshold}V) ---")
    condition = df[volt_col].abs() > threshold
    df[new_col_name] = np.where(condition, 1, 0)
    return df


def add_power_column(df: pd.DataFrame) -> pd.DataFrame:
    """Fügt eine Spalte für die Schaltleistung hinzu."""
    volt_col = config.COLUMN_NAMES["voltage"]
    curr_col = config.COLUMN_NAMES["current"]
    new_col_name = config.COLUMN_NAMES["power"]
    curr_threshold = config.ANALYSIS_CONFIG["power_calculation_current_threshold"]
    volt_threshold = config.ANALYSIS_CONFIG["power_calculation_voltage_threshold"]
    
    print(f"\n--- Spalte '{new_col_name}' hinzufügen ---")
    condition = (df[curr_col].abs() > curr_threshold) & (df[volt_col].abs() > volt_threshold)
    df[new_col_name] = np.where(condition, df[volt_col] * df[curr_col], 0)
    return df


def add_opener_time_column(df: pd.DataFrame) -> pd.DataFrame:
    """Fügt eine Zeitachse für den Öffnungsvorgang hinzu."""
    time_col = config.COLUMN_NAMES["time"]
    volt_col = config.COLUMN_NAMES["voltage"]
    new_col_name = config.COLUMN_NAMES["opener_time"]
    search_start_time = config.ANALYSIS_CONFIG["opener_time_search_start"]
    voltage_threshold = config.ANALYSIS_CONFIG["opener_time_voltage_threshold"]
    
    print(f"\n--- Spalte '{new_col_name}' hinzufügen (Suche ab t={search_start_time}s) ---")

    processed_groups = []
    for file_name, group in df.groupby(config.COLUMN_NAMES["source_file"]):
        group_copy = group.copy()
        search_window = group_copy[group_copy[time_col] >= search_start_time]
        trigger_index = search_window[search_window[volt_col].abs() > voltage_threshold].first_valid_index()

        if trigger_index is not None:
            t_zero_opener = group_copy.loc[trigger_index, time_col]
            group_copy[new_col_name] = group_copy[time_col] - t_zero_opener
        else:
            print(f"Warnung: In '{file_name}' wurde der Öffner-Schwellenwert nicht gefunden. '{new_col_name}' wird mit NaN gefüllt.")
            group_copy[new_col_name] = np.nan
        processed_groups.append(group_copy)
    return pd.concat(processed_groups, ignore_index=True)


def extract_features(df: pd.DataFrame) -> dict:
    """Extrahiert die definierten Features für jede Messung."""
    time_col = config.COLUMN_NAMES["time"]
    opener_time_col = config.COLUMN_NAMES["opener_time"]
    simplified_volt_col = config.COLUMN_NAMES["simplified_voltage"]
    power_col = config.COLUMN_NAMES["power"]
    volt_col = config.COLUMN_NAMES["voltage"]
    prellen_time_range = config.FEATURE_CONFIG["prellen_time_range"]
    ausschalt_time_range = config.FEATURE_CONFIG["ausschalt_time_range"]

    print(f"\n--- Extrahiere Features ---")
    features = {}

    for file_name, group in df.groupby(config.COLUMN_NAMES["source_file"]):
        file_features = {}

        # --- Features beim Einschalten ---
        prellen_window = group[(group[time_col] >= prellen_time_range[0]) & (group[time_col] <= prellen_time_range[1])]
        if prellen_window.empty:
            file_features.update({'Prellen': 0, 'Prelldauer [s]': 0.0, 'Schaltarbeit [Ws]': 0.0})
        else:
            transitions = prellen_window[simplified_volt_col].diff()
            file_features['Prellen'] = (transitions == -1).sum()

            last_bounce_indices = prellen_window.index[transitions == -1]
            file_features['Prelldauer [s]'] = round(prellen_window.loc[last_bounce_indices[-1], time_col], 5) if not last_bounce_indices.empty else 0.0

            switching_work = np.trapezoid(y=prellen_window[power_col].abs(), x=prellen_window[time_col]) if len(prellen_window) > 1 else 0.0
            file_features['Schaltarbeit [Ws]'] = switching_work

        # --- Features beim Ausschalten ---
        ausschalt_window = group[(group[opener_time_col] >= ausschalt_time_range[0]) & (group[opener_time_col] <= ausschalt_time_range[1])].dropna(subset=[opener_time_col])
        if ausschalt_window.empty:
            file_features.update({'AusSchaltarbeit [Ws]': 0.0, 'Lichtbogendauer [s]': 0.0})
        else:
            turn_off_work = np.trapezoid(y=ausschalt_window[power_col].abs(), x=ausschalt_window[opener_time_col]) if len(ausschalt_window) > 1 else 0.0
            file_features['AusSchaltarbeit [Ws]'] = turn_off_work

            zero_crossings = ausschalt_window.index[(ausschalt_window[volt_col].shift(-1) * ausschalt_window[volt_col]) < 0]
            file_features['Lichtbogendauer [s]'] = round(ausschalt_window.loc[zero_crossings[0], opener_time_col], 5) if not zero_crossings.empty else 0.0

        # Runde die Arbeits-Features auf 4 signifikante Stellen
        for key in ['Schaltarbeit [Ws]', 'AusSchaltarbeit [Ws]']:
            val = file_features[key]
            if val != 0:
                file_features[key] = round(val, 4 - 1 - int(np.floor(np.log10(abs(val)))))

        features[file_name] = file_features
        print(
            f"  - Datei: {file_name}, "
            f"Prellen: {file_features.get('Prellen', 0)}, "
            f"Prelldauer: {file_features.get('Prelldauer [s]', 0.0):.5f}s, "
            f"Schaltarbeit: {file_features.get('Schaltarbeit [Ws]', 0.0):.4f}Ws, "
            f"Ausschaltarbeit: {file_features.get('AusSchaltarbeit [Ws]', 0.0):.4f}Ws, "
            f"Lichtbogendauer: {file_features.get('Lichtbogendauer [s]', 0.0):.5f}s"
        )
    return features


def filter_data_by_features(df: pd.DataFrame, features: dict) -> Tuple[pd.DataFrame, dict]:
    """
    Filtert den DataFrame und das Feature-Dictionary basierend auf einem Schwellenwert für die Arbeit.
    """
    work_threshold = config.FILTER_CONFIG["max_work_threshold"]
    print(f"\n--- Filtere Messungen mit Arbeit > {work_threshold} Ws ---")
    
    files_to_remove = set()
    for file_name, feature_values in features.items():
        schaltarbeit = float(feature_values.get('Schaltarbeit [Ws]', 0.0))
        ausschaltarbeit = float(feature_values.get('AusSchaltarbeit [Ws]', 0.0))
        if schaltarbeit > work_threshold or ausschaltarbeit > work_threshold:
            files_to_remove.add(file_name)

    if not files_to_remove:
        print("Keine Messungen überschreiten den Schwellenwert. Alle Daten werden beibehalten.")
        return df, features

    print(f"Entferne {len(files_to_remove)} Messungen:")
    for file_name in sorted(list(files_to_remove)):
        print(f"  - {file_name}")
    
    filtered_features = {k: v for k, v in features.items() if k not in files_to_remove}
    filtered_df = df[~df[config.COLUMN_NAMES["source_file"]].isin(files_to_remove)].copy()
    
    return filtered_df, filtered_features


def run_processing_pipeline(data_folder: str):
    """
    Führt die gesamte Datenverarbeitungspipeline aus.
    """
    # 1. Daten importieren
    df = import_data(data_folder)
    if df is None:
        return None, None

    # 2. Verarbeitungsschritte
    df = shift_time_axis(df)
    df = optimize_and_correct_voltage(df)
    df = add_simplified_voltage_column(df)
    df = add_opener_time_column(df)
    df = add_power_column(df)

    # 3. Feature-Extraktion
    features = extract_features(df)

    # 4. Daten filtern
    df, features = filter_data_by_features(df, features)

    print("\n--- Datenübersicht nach Verarbeitung ---")
    print("Erste 5 Zeilen der finalen Daten:")
    print(df.head())
    print("\nStatistische Zusammenfassung der numerischen Spalten:")
    print(df.describe())

    return df, features