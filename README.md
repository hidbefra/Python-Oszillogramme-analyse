# Oszillogramm-Analyse-Tool

Dieses Projekt wurde in Zusammenarbeit mit Gemini Code Assist entwickelt.

Dieses Python-Skript dient zur automatisierten Analyse und Visualisierung von Oszillogrammdaten, die im CSV-Format vorliegen. Es lädt mehrere Messungen aus einem Verzeichnis, führt eine Reihe von Datenbereinigungs- und Verarbeitungsschritten durch, extrahiert relevante physikalische Kenngrößen und stellt die Ergebnisse in interaktiven Plots dar.

## Kernfunktionen

-   **Batch-Verarbeitung:** Lädt und verarbeitet alle CSV-Dateien aus einem `data`-Verzeichnis.
-   **Datenbereinigung und -transformation:**
    -   **Zeitachsen-Synchronisation:** Verschiebt die Zeitachse jeder Messung, sodass `t=0` dem Zeitpunkt der ersten signifikanten Spannungsänderung entspricht.
    -   **Spannungskorrektur:** Optimiert und korrigiert die Spannungsmessung, um parasitäre Widerstände und Offsets zu kompensieren.
    -   **Feature-Engineering:** Fügt abgeleitete Spalten hinzu, wie z. B. eine vereinfachte binäre Spannung, die Schaltleistung und eine zweite Zeitachse für den Ausschaltvorgang (`öffner Zeit`).
-   **Extraktion von Kenngrößen:** Berechnet für jede Messung automatisch:
    -   **Prellen:** Die Anzahl der Kontaktunterbrechungen beim Einschalten.
    -   **Prelldauer, Schaltarbeit (Ws):** Energie und Dauer während des Einschaltvorgangs.
    -   **Lichtbogendauer, Ausschaltarbeit (Ws):** Energie und Dauer während des Ausschaltvorgangs.
-   **Datenfilterung:** Schließt Messungen mit unrealistisch hohen Energiewerten (z.B. > 100 Ws) von der Analyse aus.
-   **Interaktive Visualisierung:**
    -   Ein Hauptfenster (`Analyse-Plot`) mit zwei synchronisierten Plots, die die Messung auf zwei verschiedenen Zeitachsen darstellen (Einschalt- und Ausschaltvorgang).
    -   Anzeige von Spannung, Strom, vereinfachter Spannung und Leistung auf vier separaten Y-Achsen.
    -   Navigations-Buttons (`Weiter`/`Zurück`) und ein durchsuchbares Dropdown-Menü zum schnellen Wechseln zwischen den Messungen.
    -   Anzeige der extrahierten Kenngrößen direkt im Plot.
-   **Zusammenfassungs-Plot:** Erstellt ein separates Fenster (`Feature-Zusammenfassung`) mit Boxplots, das die Verteilung der extrahierten Kenngrößen über alle Messungen hinweg darstellt.
-   **Flexible Konfiguration:** Alle Analyseparameter sind in einer externen `config.py`-Datei ausgelagert und können zur Laufzeit angepasst werden.
-   **Standalone Anwendung:** Kann als `.exe`-Datei für Windows ohne Python-Installation ausgeführt werden.

## Anforderungen

Stelle sicher, dass die folgenden Python-Bibliotheken installiert sind:

```bash
pip install pandas matplotlib PyQt5 scipy numpy
```

## Verwendung

1.  **Daten vorbereiten:** Erstelle einen Ordner namens `data` im selben Verzeichnis wie das Skript. Lege alle zu analysierenden `.csv`-Dateien in diesem Ordner ab.
2.  **Konfiguration anpassen (optional):** Öffne die `config.py` und passe die Analyseparameter nach Bedarf an.
3.  **Skript ausführen:** Führe das Skript über die Kommandozeile aus:
    ```bash
    python main.py
    ```
4.  **Analyse:** Die GUI startet. Wähle den Datenordner aus und klicke auf "Analyse starten". Nach der Verarbeitung werden die Plot-Fenster geöffnet.

## Datenverarbeitung im Detail

Das Skript führt die folgenden Analyseschritte in der angegebenen Reihenfolge durch:

1.  **Datenimport:** Lädt die CSV-Dateien (überspringt die ersten 11 Zeilen) und kombiniert sie in einem einzigen DataFrame.
2.  **Zeitachsen-Verschiebung:** Findet den Zeitpunkt, an dem die Spannungsänderung `|ΔU| > 20 V` überschreitet, und setzt diesen als neuen Nullpunkt `t=0`.
3.  **Spannungskorrektur:** Minimiert die Spannung im Zeitfenster `[0.01s, 0.03s]` durch eine Optimierung von Widerstand und Offset, um Messfehler zu korrigieren.
4.  **Hinzufügen von Spalten:**
    -   `vereinfachte Spannung`: `1` wenn `|U| > 1V`, sonst `0`.
    -   `öffner Zeit [s]`: Eine neue Zeitachse, deren Nullpunkt auf den Beginn des Ausschaltvorgangs gelegt wird (erster Spannungsanstieg nach `t=0.4s`).
    -   `Schaltleistung [W]`: `U * I`, berechnet nur in den relevanten Schaltfenstern und wenn Spannungs- und Stromschwellenwerte überschritten werden.
5.  **Feature-Extraktion:** Berechnet Prellen, Schalt- und Ausschaltarbeit in definierten Zeitfenstern.
6.  **Datenfilterung:** Messungen, deren Schalt- oder Ausschaltarbeit einen Schwellenwert (standardmäßig 100 Ws) überschreiten, werden von der weiteren Analyse und Visualisierung ausgeschlossen.
7.  **Visualisierung:** Erstellt die interaktiven Plots auf Basis der bereinigten und angereicherten Daten.

## Erstellen einer neuen Version (Release)

Das Projekt enthält ein `build.bat`-Skript, um den Prozess der Erstellung einer neuen Release-Version zu automatisieren.

1.  **Versionsnummer erhöhen:** Öffne die `config.py` und erhöhe die `VERSION`-Variable (z.B. von `"1.0.0"` auf `"1.1.0"`).
2.  **Änderungen committen:** Speichere deine Änderungen und committe sie mit Git.
    ```bash
    git add .
    git commit -m "Release v1.1.0"
    ```
3.  **Build-Skript ausführen:** Führe die `build.bat` aus. Das Skript wird:
    -   Die `.exe`-Datei mit PyInstaller erstellen.
    -   Ein `.zip`-Archiv im `release`-Ordner erstellen, das die `OszillogrammAnalyseTool.exe` und die `config.py` enthält.
4.  **Release auf GitHub erstellen:**
    -   Pushe deine Commits und erstelle einen neuen Tag, der zur Version passt:
      ```bash
      git tag v1.1.0
      git push origin main --tags
      ```
    -   Gehe zur "Releases"-Seite deines GitHub-Repositories und erstelle einen neuen Release aus dem soeben erstellten Tag.
    -   Lade die generierte `.zip`-Datei als Release-Asset hoch.

<img width="1171" height="417" alt="grafik" src="https://github.com/user-attachments/assets/93797a70-0283-4ba1-ac99-ac444542ba29" />

<img width="1677" height="877" alt="grafik" src="https://github.com/user-attachments/assets/76fcaa15-b026-4d28-8fdb-f0b26557450f" />
