@echo off
echo "Erstelle die .exe-Datei für das OszillogrammAnalyseTool..."

pyinstaller --name "OszillogrammAnalyseTool" --windowed --onefile --icon="icon.ico" main.py

echo "Build abgeschlossen. Die .exe befindet sich im 'dist'-Ordner."
pause
