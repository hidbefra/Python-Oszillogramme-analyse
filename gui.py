# gui.py
"""
Haupt-GUI-Modul für das Oszillogramm-Analyse-Tool.
Ermöglicht die Auswahl von Ordnern und das Starten von Analysen in separaten Threads.
"""
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QFileDialog, QLabel, QLineEdit, QHBoxLayout, QTextEdit
)
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from PyQt5.QtGui import QTextCursor

import data_processing
import plotting 
from config_loader import settings as config


class Stream(QObject):
    """Leitet Text von einem Stream (wie stdout) an ein PyQt-Signal weiter."""
    newText = pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))

    def flush(self):
        pass # Nötig für die Stream-Schnittstelle


class AnalysisWorker(QThread):
    """
    Worker-Thread, der die Datenverarbeitung und Plot-Erstellung ausführt.
    Sendet Signale, um die GUI über den Fortschritt und das Ergebnis zu informieren.
    """
    analysis_complete = pyqtSignal(object, object, str) # df, features, data_folder
    error_signal = pyqtSignal(str)

    def __init__(self, data_folder: str):
        super().__init__()
        self.data_folder = data_folder

    def run(self):
        """Führt die Analyse-Pipeline aus."""
        try:
            print(f"\n--- Starte Analyse für Ordner: {self.data_folder} ---")
            # 1. Datenverarbeitungspipeline ausführen
            final_df, features = data_processing.run_processing_pipeline(self.data_folder)

            if final_df is None or final_df.empty:
                self.error_signal.emit(f"Keine Daten im Ordner '{self.data_folder}' gefunden oder verarbeitet.")
                return

            # Sende die Ergebnisse an den Haupt-Thread, um die Plots zu erstellen
            self.analysis_complete.emit(final_df, features, self.data_folder)
        except Exception as e:
            self.error_signal.emit(f"Ein Fehler ist aufgetreten: {e}")


class MainWindow(QWidget):
    """Hauptfenster der Anwendung."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Oszillogramm Analyse Tool")
        self.setGeometry(100, 100, 800, 600)
        self.threads = []
        self.plot_button_references = [] # Referenzen auf Plot-Buttons halten

        # Layout und Widgets
        layout = QVBoxLayout(self)
        folder_layout = QHBoxLayout()

        self.folder_path_edit = QLineEdit(config.IMPORT_CONFIG["data_folder"])
        folder_layout.addWidget(self.folder_path_edit)

        self.select_folder_btn = QPushButton("Ordner wählen...")
        self.select_folder_btn.clicked.connect(self.select_folder)
        folder_layout.addWidget(self.select_folder_btn)

        layout.addLayout(folder_layout)

        self.start_analysis_btn = QPushButton("Analyse starten")
        self.start_analysis_btn.clicked.connect(self.start_analysis)
        layout.addWidget(self.start_analysis_btn)

        self.status_label = QLabel("Bereit. Bitte einen Ordner wählen und Analyse starten.")
        layout.addWidget(self.status_label)

        # Log-Konsole hinzufügen
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        layout.addWidget(self.log_console)

        # stdout und stderr umleiten
        self.redirect_streams()

    def redirect_streams(self):
        """Leitet print-Anweisungen und Fehler in die Log-Konsole um."""
        # stdout
        sys.stdout = Stream(newText=self.on_new_log_text)
        # stderr
        sys.stderr = Stream(newText=self.on_new_log_text)

    def on_new_log_text(self, text):
        """Fügt Text zur Log-Konsole hinzu und scrollt nach unten."""
        self.log_console.moveCursor(QTextCursor.End)
        self.log_console.insertPlainText(text)

    def select_folder(self):
        """Öffnet einen Dialog zur Ordnerauswahl."""
        folder = QFileDialog.getExistingDirectory(self, "Datenordner auswählen", self.folder_path_edit.text())
        if folder:
            self.folder_path_edit.setText(folder)

    def start_analysis(self):
        """Startet einen neuen Analyse-Thread."""
        data_folder = self.folder_path_edit.text()
        self.status_label.setText(f"Starte Analyse für '{data_folder}'...")
        
        worker = AnalysisWorker(data_folder)
        worker.analysis_complete.connect(self.on_analysis_complete)
        worker.error_signal.connect(lambda msg: self.status_label.setText(f"Fehler: {msg}"))
        
        # Wenn der Thread fertig ist (egal ob erfolgreich oder nicht), entfernen wir ihn aus der Liste
        worker.finished.connect(lambda: self.threads.remove(worker))
        
        self.threads.append(worker) # Referenz halten, damit der Thread nicht gelöscht wird
        worker.start()

    def on_analysis_complete(self, final_df, features, data_folder):
        """
        Slot, der aufgerufen wird, wenn eine Analyse abgeschlossen ist.
        Erstellt die Plots im Haupt-Thread.
        """
        self.status_label.setText(f"Analyse für '{data_folder}' abgeschlossen. Erstelle Plots...")
        
        # 1. Plots erstellen (jetzt im sicheren Haupt-Thread)
        global_limits = plotting.calculate_global_plot_limits(final_df)
        plotting.create_feature_summary_plot(features)
        
        b_prev, b_next = plotting.create_interactive_plot(final_df, features, global_limits)
        self.plot_button_references.extend([b_prev, b_next]) # Referenzen speichern

        # 2. Alle erstellten Plots anzeigen (nicht-blockierend)
        plotting.show_plots()
        self.status_label.setText(f"Analyse für '{data_folder}' abgeschlossen. Plots werden angezeigt.")