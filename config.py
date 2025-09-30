# config.py
"""
Zentrale Konfigurationsdatei für das Oszillogramm-Analyse-Tool.
Alle Einstellungen, Schwellenwerte und Parameter sind hier definiert.
"""

# --- Datenimport-Einstellungen ---
IMPORT_CONFIG = {
    "data_folder": "data",
    "skiprows": 11,
    "header": None,
    "usecols": [0, 1, 2],
    "file_sampling_rate": 1,  # Jede n-te Datei laden (1 für alle)
}

# --- Spaltennamen ---
# Definiert die Namen der Spalten, wie sie im Skript verwendet werden.
COLUMN_NAMES = {
    "time": "Zeit[s]",
    "voltage": "Spannung[V]",
    "current": "Strom[A]",
    "simplified_voltage": "vereinfachte Spannung",
    "power": "Schaltleistung [W]",
    "opener_time": "öffner Zeit [s]",
    "source_file": "source_file",
}

# --- Analyse-Parameter ---
ANALYSIS_CONFIG = {
    # Zeitachsen-Verschiebung: Schwellenwert für Strom, um t=0 zu finden
    "time_shift_current_threshold": 5.0,

    # Spannungskorrektur: Zeitfenster für die Optimierung
    "voltage_correction_time_range": (0.01, 0.03),

    # Vereinfachte Spannung: Spannungsschwelle für die Binarisierung
    "simplified_voltage_threshold": 1.0,

    # Schaltleistung: Schwellenwerte, ab denen die Leistung berechnet wird
    "power_calculation_current_threshold": 0.3,
    "power_calculation_voltage_threshold": 0.8,

    # Öffner-Zeit: Startzeit und Spannungsschwelle für die Suche nach dem Öffnungsereignis
    "opener_time_search_start": 0.4,
    "opener_time_voltage_threshold": 1.0,
}

# --- Feature-Extraktion-Parameter ---
FEATURE_CONFIG = {
    # Zeitfenster für die Analyse von Prellen und Schaltarbeit
    "prellen_time_range": (0.0, 0.02),

    # Zeitfenster für die Analyse der Ausschaltarbeit und Lichtbogendauer
    "ausschalt_time_range": (0.0, 0.02),
}

# --- Datenfilterung ---
FILTER_CONFIG = {
    # Messungen mit einer Schalt- oder Ausschaltarbeit über diesem Wert werden entfernt
    "max_work_threshold": 100.0,
}

# --- Plot-Einstellungen ---
PLOT_CONFIG = {
    # Feste Y-Achsen-Limits für die Plots.
    # Setze auf None, um automatische Limits zu verwenden.
    "y_axis_limits": {
        "voltage": (-815, 815),
        "current": (-70, 70),
        "power": (-200, 200),
    },
    # Symmetrisches Padding, falls keine festen Limits gesetzt sind
    "symmetric_padding": 1.05,
}

# --- Zusammenfassungs-Plot ---
SUMMARY_PLOT_CONFIG = {
    # Anzahl der Messungen, die in einem Boxplot zusammengefasst werden
    "boxplot_group_size": 25,
}