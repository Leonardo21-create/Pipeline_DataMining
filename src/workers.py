"""
Modulo de concurrencia para PyQt5.

Ejecuta tareas pesadas (carga de CSV, preprocesamiento y entrenamiento)
en un QThread secundario sin acoplar el backend a la interfaz grafica.
"""

from typing import Any, Callable

from PyQt5.QtCore import QObject, QThread, pyqtSignal


class TaskWorker(QObject):
    """Worker generico que invoca una funcion Python dentro de un hilo secundario."""

    terminado = pyqtSignal(object)
    error = pyqtSignal(str)
    finalizado = pyqtSignal()

    def __init__(self, funcion: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._funcion = funcion
        self._args = args
        self._kwargs = kwargs

    def run(self) -> None:
        try:
            resultado = self._funcion(*self._args, **self._kwargs)
            self.terminado.emit(resultado)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.finalizado.emit()


def iniciar_tarea_en_hilo(
    funcion: Callable[..., Any],
    on_terminado: Callable[[Any], None],
    on_error: Callable[[str], None],
    on_finalizado: Callable[[], None],
    *args: Any,
    **kwargs: Any,
) -> tuple[QThread, TaskWorker]:
    """
    Crea, enlaza e inicia un QThread con limpieza automatica de memoria.
    El llamador debe conservar la referencia al par (thread, worker) mientras corre.
    """
    hilo = QThread()
    worker = TaskWorker(funcion, *args, **kwargs)
    worker.moveToThread(hilo)

    hilo.started.connect(worker.run)
    worker.terminado.connect(on_terminado)
    worker.error.connect(on_error)

    worker.finalizado.connect(hilo.quit)
    worker.finalizado.connect(worker.deleteLater)
    hilo.finished.connect(hilo.deleteLater)
    hilo.finished.connect(on_finalizado)

    hilo.start()
    return hilo, worker