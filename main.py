# vibe coding mit gemini :)
# Python 3.13.0 (tags/v3.13.0:60403a5, Oct  7 2024, 09:38:07) [MSC v.1941 64 bit (AMD64)] on win32
# pip install pandas matplotlib PyQt5 scipy
import sys
import matplotlib
matplotlib.use('Qt5Agg') # Ändert das Backend, um Tkinter-Probleme zu umgehen

from PyQt5.QtWidgets import QApplication
from gui import MainWindow

def main():
    """
    Hauptfunktion, die die GUI-Anwendung startet.
    """
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()