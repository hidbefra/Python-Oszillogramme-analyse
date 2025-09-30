# config_loader.py
"""
Modul zum dynamischen Laden der Konfigurationsdatei (config.py).
Dies ermöglicht es, die Konfiguration zur Laufzeit zu ändern,
auch wenn die Anwendung als .exe-Datei kompiliert ist.
"""
import importlib.util
import os
import sys

# Globale Variable, die die geladene Konfiguration enthalten wird.
# Wir initialisieren sie mit einem leeren Objekt, falls die Datei nicht gefunden wird.
class Config:
    pass

settings = Config()

def get_config_path():
    """
    Ermittelt den Pfad zur config.py.
    Wenn die Anwendung als .exe läuft (gefroren), suchen wir neben der .exe.
    Ansonsten suchen wir im aktuellen Arbeitsverzeichnis.
    """
    if getattr(sys, 'frozen', False):
        # Wir sind in einer .exe-Datei (PyInstaller)
        application_path = os.path.dirname(sys.executable)
    else:
        # Wir laufen als normales .py-Skript
        application_path = os.path.dirname(os.path.abspath(__file__))
    
    return os.path.join(application_path, 'config.py')

def load_config():
    """
    Lädt die Konfiguration aus der config.py-Datei.
    """
    config_path = get_config_path()
    
    if not os.path.exists(config_path):
        print(f"WARNUNG: Konfigurationsdatei nicht gefunden unter: {config_path}")
        print("Es werden Standardwerte verwendet, was zu Fehlern führen kann.")
        return

    try:
        # Lade die config.py-Datei als Modul
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)

        # Kopiere alle globalen Variablen aus dem geladenen Modul in unser 'settings'-Objekt
        for key in dir(config_module):
            if not key.startswith("__"):
                setattr(settings, key, getattr(config_module, key))
        
        print(f"Konfiguration erfolgreich von {config_path} geladen.")

    except Exception as e:
        print(f"FEHLER beim Laden der Konfiguration von {config_path}: {e}")
        print("Die Anwendung wird möglicherweise nicht korrekt funktionieren.")

# Lade die Konfiguration, sobald dieses Modul importiert wird.
load_config()
