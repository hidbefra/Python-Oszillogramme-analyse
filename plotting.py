# plotting.py
"""
Modul für die interaktive Visualisierung der Oszillogrammdaten und Features.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Button
from PyQt5.QtWidgets import QComboBox, QWidget, QHBoxLayout, QCompleter
from PyQt5.QtCore import Qt
from typing import Dict, Tuple, List

import config


class PlotManager:
    """Verwaltet mehrere PlotNavigator-Instanzen, um sie synchron zu halten."""
    def __init__(self, navigators: list):
        self.navigators = navigators
        self.files = navigators[0].files if navigators else []
        self.current_index = 0
        self.update_combo_callback = None

    def _navigate(self, step: int):
        """Interne Navigationslogik."""
        if not self.files:
            return
        self.current_index = (self.current_index + step) % len(self.files)
        self.update_all_plots()

    def next(self, event):
        """Wechselt zur nächsten Datei."""
        self._navigate(1)

    def prev(self, event):
        """Wechselt zur vorherigen Datei."""
        self._navigate(-1)

    def update_all_plots(self):
        """Aktualisiert alle Plots, um die Daten des aktuellen Index anzuzeigen."""
        for nav in self.navigators:
            nav.current_index = self.current_index
            nav.update_plot()
        if self.update_combo_callback:
            self.update_combo_callback()


class PlotNavigator:
    """Verwaltet die interaktive Navigation für einen einzelnen Plot."""
    def __init__(self, combined_df: pd.DataFrame, ax1: plt.Axes, ax2: plt.Axes, ax4: plt.Axes, time_col: str, features: dict, global_limits: dict):
        self.combined_df = combined_df
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax4 = ax4
        self.time_col = time_col
        self.volt_col = config.COLUMN_NAMES["voltage"]
        self.curr_col = config.COLUMN_NAMES["current"]
        self.power_col = config.COLUMN_NAMES["power"]
        self.global_limits = global_limits
        self.features = features
        self.files = self.combined_df[config.COLUMN_NAMES["source_file"]].unique()
        self.current_index = 0
        self._setup_axes()

        self.ax_table = plt.axes([0.125, 0.05, 0.2, 0.075], frame_on=False)
        self.ax_table.xaxis.set_visible(False)
        self.ax_table.yaxis.set_visible(False)

    def _setup_axes(self):
        """Konfiguriert die statischen Eigenschaften der Achsen."""
        volt_color, curr_color, power_color = 'tab:blue', 'tab:red', 'tab:purple'

        self.ax1.set_xlabel('Zeit [s]')
        self.ax1.set_ylabel('Spannung [V]', color=volt_color)
        self.ax1.tick_params(axis='y', labelcolor=volt_color)
        self.ax1.grid(True)
        self.ax1.set_ylim(self.global_limits['volt'])

        self.ax2.set_ylabel('Strom [A]', color=curr_color)
        self.ax2.tick_params(axis='y', labelcolor=curr_color)
        self.ax2.patch.set_visible(False)
        self.ax2.set_ylim(self.global_limits['curr'])

        self.ax4.set_ylabel(self.power_col, color=power_color)
        self.ax4.tick_params(axis='y', labelcolor=power_color)
        self.ax4.spines['right'].set_position(('outward', 80))
        self.ax4.patch.set_visible(False)
        self.ax4.set_ylim(self.global_limits['power'])

    def update_plot(self):
        """Zeichnet den Plot für den aktuellen Index neu."""
        for ax in [self.ax1, self.ax2, self.ax4]:
            for line in ax.lines:
                line.remove()

        current_file = self.files[self.current_index]
        df_to_plot = self.combined_df[self.combined_df[config.COLUMN_NAMES["source_file"]] == current_file]

        if df_to_plot[self.time_col].isnull().all():
            self.ax1.set_title(f'Daten für {self.time_col} nicht verfügbar: {current_file}')
            return

        line1 = self.ax1.plot(df_to_plot[self.time_col], df_to_plot[self.volt_col], color='tab:blue', label='Spannung [V]')
        line2 = self.ax2.plot(df_to_plot[self.time_col], df_to_plot[self.curr_col], color='tab:red', label='Strom [A]')
        line4 = self.ax4.plot(df_to_plot[self.time_col], df_to_plot[self.power_col], color='tab:purple', label=self.power_col, linestyle=':')

        current_features = self.features.get(current_file, {})
        table_data = [[f"{v:.4g}"] for v in current_features.values()]
        row_labels = list(current_features.keys())

        self.ax_table.cla()
        if table_data:
            self.ax_table.table(cellText=table_data, rowLabels=row_labels, loc='center')

        if self.ax1.get_figure().axes[0] == self.ax1:
            self.ax1.set_title(f'Oszillogramm: {current_file} ({self.current_index + 1}/{len(self.files)})')
            self.ax1.legend(handles=line1 + line2 + line4, loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=False)
        
        plt.draw()


def calculate_global_plot_limits(df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """Berechnet die globalen, symmetrischen Achsen-Limits für alle Plots."""
    print("\n--- Berechne globale Achsen-Limits ---")
    time_col, opener_time_col = config.COLUMN_NAMES["time"], config.COLUMN_NAMES["opener_time"]
    volt_col, curr_col, power_col = config.COLUMN_NAMES["voltage"], config.COLUMN_NAMES["current"], config.COLUMN_NAMES["power"]

    def get_symmetric_limits(series: pd.Series) -> Tuple[float, float]:
        padding = config.PLOT_CONFIG["symmetric_padding"]
        max_abs = series.abs().max() * padding
        return (-max_abs, max_abs) if pd.notna(max_abs) and max_abs > 0 else (-1, 1)

    limits = {
        'volt': config.PLOT_CONFIG["y_axis_limits"]["voltage"] or get_symmetric_limits(df[volt_col]),
        'curr': config.PLOT_CONFIG["y_axis_limits"]["current"] or get_symmetric_limits(df[curr_col]),
        'power': config.PLOT_CONFIG["y_axis_limits"]["power"] or get_symmetric_limits(df[power_col]),
        'time': (df[time_col].min(), df[time_col].max()),
        'opener_time': (df[opener_time_col].min(), df[opener_time_col].max())
    }
    
    if np.any(pd.isna(limits['opener_time'])):
        limits['opener_time'] = limits['time']

    print(f"  - Spannungs-Limits: {limits['volt']}")
    print(f"  - Strom-Limits: {limits['curr']}")
    print(f"  - Leistungs-Limits: {limits['power']}")
    return limits


def create_feature_summary_plot(features: Dict):
    """Erstellt einen zusammenfassenden Plot für alle extrahierten Features."""
    if not features:
        print("Keine Features zum Plotten vorhanden.")
        return

    features_df = pd.DataFrame.from_dict(features, orient='index').apply(pd.to_numeric, errors='coerce')
    num_features = len(features_df.columns)
    fig, axes = plt.subplots(nrows=num_features, ncols=1, figsize=(15, 5 * num_features), sharex=True)
    fig.canvas.manager.set_window_title('Feature-Zusammenfassung')
    fig.suptitle('Zusammenfassung der extrahierten Features', fontsize=16)
    axes = [axes] if num_features == 1 else axes

    for i, feature_name in enumerate(features_df.columns):
        ax = axes[i]
        data_series = features_df[feature_name].dropna()
        
        if any(keyword in feature_name.lower() for keyword in ['arbeit', 'prellen', 'dauer']):
            group_size = config.SUMMARY_PLOT_CONFIG["boxplot_group_size"]
            grouped_data = [data_series[i:i + group_size] for i in range(0, len(data_series), group_size)]
            labels = [f"{i*group_size+1}-{(i+1)*group_size}" for i in range(len(grouped_data))]
            ax.boxplot(grouped_data, tick_labels=labels)
            ax.set_title(f"Boxplot für: {feature_name}")
        else:
            ax.plot(features_df.index, features_df[feature_name], marker='o', linestyle='-')
            ax.set_title(f"Verlauf für: {feature_name}")

        ax.set_ylabel(feature_name)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.tick_params(axis='x', labelrotation=90)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


def create_interactive_plot(df: pd.DataFrame, features: dict, global_limits: dict):
    """Erstellt und zeigt einen interaktiven Plot mit zwei synchronisierten Zeitachsen an."""
    time_col = config.COLUMN_NAMES["time"]
    opener_time_col = config.COLUMN_NAMES["opener_time"]

    fig, (ax1_top, ax1_bottom) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    fig.canvas.manager.set_window_title('Analyse-Plot')
    fig.subplots_adjust(bottom=0.15, right=0.85, top=0.92, hspace=0.4)

    # --- Setup für den oberen Plot (Haupt-Zeitachse) ---
    ax2_top, ax4_top = ax1_top.twinx(), ax1_top.twinx()
    navigator1 = PlotNavigator(df, ax1_top, ax2_top, ax4_top, time_col, features, global_limits)
    ax1_top.set_xlim(global_limits['time'])

    # --- Setup für den unteren Plot (Öffner-Zeitachse) ---
    ax2_bottom, ax4_bottom = ax1_bottom.twinx(), ax1_bottom.twinx()
    navigator2 = PlotNavigator(df, ax1_bottom, ax2_bottom, ax4_bottom, opener_time_col, features, global_limits)
    ax1_bottom.set_xlim(global_limits['opener_time'])
    ax1_bottom.set_xlabel(opener_time_col)

    # --- Manager und UI-Elemente ---
    plot_manager = PlotManager([navigator1, navigator2])

    # --- Dropdown-Menü (ComboBox) ---
    toolbar = fig.canvas.manager.toolbar
    combo_box = QComboBox()
    combo_box.setMinimumWidth(350)
    combo_box.addItems(plot_manager.files)
    combo_box.setEditable(True)
    combo_box.setInsertPolicy(QComboBox.NoInsert)
    combo_box.completer().setCompletionMode(QCompleter.PopupCompletion)
    combo_box.completer().setFilterMode(Qt.MatchContains)
    combo_box.completer().setCaseSensitivity(Qt.CaseInsensitive)

    def on_combo_select(index):
        if plot_manager.current_index != index:
            plot_manager.current_index = index
            plot_manager.update_all_plots()

    combo_box.activated.connect(on_combo_select)
    
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.addWidget(combo_box)
    toolbar.addWidget(container)

    # --- Navigations-Buttons ---
    ax_prev = fig.add_axes([0.7, 0.02, 0.1, 0.05])
    ax_next = fig.add_axes([0.81, 0.02, 0.1, 0.05])
    b_prev = Button(ax_prev, 'Zurück')
    b_next = Button(ax_next, 'Weiter')
    b_prev.on_clicked(plot_manager.prev)
    b_next.on_clicked(plot_manager.next)

    # --- Callbacks verbinden ---
    def update_combo_on_nav():
        combo_box.setCurrentIndex(plot_manager.current_index)
    plot_manager.update_combo_callback = update_combo_on_nav

    # Initiales Plotten
    plot_manager.update_all_plots()

    # Referenzen zurückgeben, damit sie nicht vom Garbage Collector gelöscht werden
    return b_prev, b_next


def show_plots():
    """Zeigt alle erstellten Matplotlib-Fenster an."""
    print("\n--- Plot-Fenster werden angezeigt. Die Anwendung läuft weiter. ---")
    plt.show(block=False)