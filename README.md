# Oszillogramm-Analyse-Tool

vibe coding mit gemini

Dieses Python-Skript dient zur automatisierten Analyse und Visualisierung von Oszillogrammdaten, die im CSV-Format vorliegen. Es lädt mehrere Messungen aus einem Verzeichnis, führt eine Reihe von Datenbereinigungs- und Verarbeitungsschritten durch, extrahiert relevante physikalische Kenngrößen und stellt die Ergebnisse in interaktiven Plots dar.

## Kernfunktionen

-   **Batch-Verarbeitung:** Lädt und verarbeitet alle CSV-Dateien aus einem `data`-Verzeichnis.
-   **Datenbereinigung und -transformation:**
    -   **Zeitachsen-Synchronisation:** Verschiebt die Zeitachse jeder Messung, sodass `t=0` dem Zeitpunkt entspricht, an dem der Strom erstmals einen Schwellenwert überschreitet.
    -   **Spannungskorrektur:** Optimiert und korrigiert die Spannungsmessung, um parasitäre Widerstände und Offsets zu kompensieren.
    -   **Feature-Engineering:** Fügt abgeleitete Spalten hinzu, wie z. B. eine vereinfachte binäre Spannung, die Schaltleistung und eine zweite Zeitachse für den Ausschaltvorgang (`öffner Zeit`).
-   **Extraktion von Kenngrößen:** Berechnet für jede Messung automatisch:
    -   **Prellen:** Die Anzahl der Kontaktunterbrechungen beim Einschalten.
    -   **Schaltarbeit (Ws):** Die Energie, die während des Einschaltvorgangs umgesetzt wird.
    -   **Ausschaltarbeit (Ws):** Die Energie, die während des Ausschaltvorgangs umgesetzt wird.
-   **Datenfilterung:** Schließt Messungen mit unrealistisch hohen Energiewerten (z.B. > 100 Ws) von der Analyse aus.
-   **Interaktive Visualisierung:**
    -   Ein Hauptfenster (`Analyse-Plot`) mit zwei synchronisierten Plots, die die Messung auf zwei verschiedenen Zeitachsen darstellen (Einschalt- und Ausschaltvorgang).
    -   Anzeige von Spannung, Strom, vereinfachter Spannung und Leistung auf vier separaten Y-Achsen.
    -   Navigations-Buttons (`Weiter`/`Zurück`) und ein durchsuchbares Dropdown-Menü zum schnellen Wechseln zwischen den Messungen.
    -   Anzeige der extrahierten Kenngrößen direkt im Plot.
-   **Zusammenfassungs-Plot:** Erstellt ein separates Fenster (`Feature-Zusammenfassung`) mit Boxplots, das die Verteilung der extrahierten Kenngrößen über alle Messungen hinweg darstellt.

## Anforderungen

Für die Ausführung als Python-Skript, stelle sicher, dass die folgenden Bibliotheken installiert sind:

```bash
pip install -r requirements.txt
```

## Verwendung

1.  **Daten vorbereiten:** Erstelle einen Ordner namens `data` im selben Verzeichnis wie das Skript. Lege alle zu analysierenden `.csv`-Dateien in diesem Ordner ab.
2.  **Skript ausführen:** Führe das Skript über die Kommandozeile aus:
    ```bash
    python main.py
    ```
3.  **Analyse:** Das Skript verarbeitet alle Dateien und öffnet anschließend zwei Fenster:
    -   `Analyse-Plot`: Der interaktive Plot zur Untersuchung der einzelnen Oszillogramme.
    -   `Feature-Zusammenfassung`: Der Plot mit der statistischen Übersicht der Kenngrößen.

## Datenverarbeitung im Detail

Das Skript führt die folgenden Analyseschritte in der angegebenen Reihenfolge durch:

1.  **Datenimport:** Lädt die CSV-Dateien (überspringt die ersten 11 Zeilen) und kombiniert sie in einem einzigen DataFrame.
2.  **Zeitachsen-Verschiebung:** Findet den Zeitpunkt, an dem der Strom `|I| > 5.0 A` überschreitet, und setzt diesen als neuen Nullpunkt `t=0`.
3.  **Spannungskorrektur:** Minimiert die Spannung im Zeitfenster `[0.01s, 0.03s]` durch eine Optimierung von Widerstand und Offset, um Messfehler zu korrigieren.
4.  **Hinzufügen von Spalten:**
    -   `vereinfachte Spannung`: `1` wenn `|U| > 1V`, sonst `0`.
    -   `öffner Zeit [s]`: Eine neue Zeitachse, deren Nullpunkt auf den Beginn des Ausschaltvorgangs gelegt wird (erster Spannungsanstieg nach `t=0.4s`).
    -   `Schaltleistung W`: `U * I`, aber nur wenn `|I| > 0.3 A`.
5.  **Feature-Extraktion:** Berechnet Prellen, Schalt- und Ausschaltarbeit in definierten Zeitfenstern.
6.  **Datenfilterung:** Messungen, deren Schalt- oder Ausschaltarbeit einen Schwellenwert (standardmäßig 100 Ws) überschreiten, werden von der weiteren Analyse und Visualisierung ausgeschlossen.
7.  **Visualisierung:** Erstellt die interaktiven Plots auf Basis der bereinigten und angereicherten Daten.

<img width="1171" height="417" alt="grafik" src="https://github.com/user-attachments/assets/93797a70-0283-4ba1-ac99-ac444542ba29" />

<img width="1677" height="877" alt="grafik" src="https://github.com/user-attachments/assets/76fcaa15-b026-4d28-8fdb-f0b26557450f" />
