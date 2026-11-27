"""Punto de entrada de la aplicacion PyQt5 para el proyecto UNSW-NB15."""

import sys

from PyQt5.QtWidgets import QApplication

from src.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    ventana = MainWindow()
    ventana.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())