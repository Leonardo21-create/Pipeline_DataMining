"""
Controlador principal de la interfaz PyQt5 (metodologia CRISP-DM).

Carga `ui/main_window.ui` en tiempo de ejecucion, coordina el estado entre
pestanas, despacha tareas pesadas via `src/workers.py` y renderiza las figuras
de `src/plotting.py` sin logica de negocio ni de modelado acoplada.
"""

from pathlib import Path
from typing import Any

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import pandas as pd
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import (
    QFileDialog,
    QHeaderView,
    QLayout,
    QMainWindow,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
)

from src.modeling import entrenar_y_evaluar
from src.plotting import (
    graficar_curva_roc,
    graficar_distribucion_clases,
    graficar_matriz_confusion,
    graficar_outliers_p99,
)
from src.preprocessing import (
    cargar_e_integrar_datos,
    dividir_y_escalar,
    ejecutar_demo_joins,
    limpiar_datos,
    transformar_datos,
)
from src.workers import iniciar_tarea_en_hilo

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_UI = RAIZ_PROYECTO / "ui" / "main_window.ui"

TEXTO_NEGOCIO_HTML = """
<h2>Proyecto de Mineria de Datos: Deteccion de Intrusiones en Red (UNSW-NB15)</h2>
<p><b>Metodologia:</b> CRISP-DM &nbsp;|&nbsp; <b>Enfoque:</b> Clasificacion supervisada binaria (Normal vs. Ataque)</p>
<hr>
<h3>1. Contexto y Problema</h3>
<p>Las redes empresariales procesan gran cantidad de trafico donde los sistemas  basados en reglas fijas
dejan pasar amenazas   y saturan al equipo de IT con falsas alarmas.</p>

<h3>2. Objetivo del Negocio</h3>
<p>Proteger la seguridad de la red  mediante la deteccion confiable
de trafico malicioso, reduciendo el riesgo de intrusiones  y disminuyendo la carga operativa.</p>

<h3>3. Objetivo de Mineria de Datos</h3>
<p>Entrenar y comparar modelos capaces de clasificar cada conexion de red del dataset <b>UNSW-NB15</b> en
<b>Normal (0)</b> o <b>Ataque (1)</b> a partir de sus caracteristicas.</p>

<h3>4. Criterios de Exito</h3>
<ul>
  <li><b>Negocio:</b> Identificar la gran mayoria de ataques reales manteniendo bajo control las falsas alarmas.</li>
  <li><b>Tecnico:</b> Priorizar <b>Sensibilidad </b> para minimizar ataques no detectados, equilibrada con
  <b>Precision</b>, <b>Especificidad</b>, <b>F1-Score</b> y <b>AUC-ROC</b> , validando con
  <i>Hold-out</i> y <i>Stratified K-Fold</i> sin fuga de datos.</li>
</ul>
"""

COLUMNAS_HISTORIAL: tuple[str, ...] = (
    "#",
    "Modelo",
    "Validacion",
    "Exactitud",
    "Precision",
    "Sensibilidad",
    "Especificidad",
    "F1-Score",
    "AUC-ROC",
    "Tiempo (s)",
)

ETIQUETAS_METRICAS: tuple[tuple[str, str], ...] = (
    ("exactitud", "Exactitud (Accuracy)"),
    ("precision", "Precision"),
    ("sensibilidad", "Sensibilidad (Recall)"),
    ("especificidad", "Especificidad"),
    ("f1_score", "F1-Score"),
    ("auc_roc", "AUC-ROC"),
)


def _tarea_cargar_y_joins(ruta_train: str, ruta_test: str) -> dict[str, Any]:
    """Tarea compuesta ejecutada en QThread para la Pestana 2."""
    df_integrado, resumen = cargar_e_integrar_datos(ruta_train, ruta_test, retornar_resumen=True)
    joins = ejecutar_demo_joins(df_integrado)
    return {"df": df_integrado, "resumen": resumen, "joins": joins}


def _tarea_preprocesar(
    df_integrado: pd.DataFrame,
    metodo_escalado: str,
    test_size: float,
) -> dict[str, Any]:
    """Tarea compuesta ejecutada en QThread para la Pestana 3."""
    df_limpio, res_limpieza = limpiar_datos(df_integrado, retornar_resumen=True)
    X_encoded, y, res_transformacion = transformar_datos(
        df_limpio, columna_objetivo="label", retornar_resumen=True
    )
    X_train, X_test, y_train, y_test, scaler, res_escalado = dividir_y_escalar(
        X_encoded,
        y,
        metodo=metodo_escalado,
        test_size=test_size,
        random_state=42,
        retornar_resumen=True,
    )
    return {
        "X_encoded": X_encoded,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "metodo_escalado": metodo_escalado,
        "res_limpieza": res_limpieza,
        "res_transformacion": res_transformacion,
        "res_escalado": res_escalado,
    }


class MainWindow(QMainWindow):
    """Ventana principal de la aplicacion CRISP-DM para UNSW-NB15."""

    def __init__(self) -> None:
        super().__init__()
        if not RUTA_UI.exists():
            raise FileNotFoundError(f"No se encontro el archivo de interfaz: {RUTA_UI}")

        uic.loadUi(str(RUTA_UI), self)

        # Estado en memoria compartido entre pestanas
        self._df_integrado: pd.DataFrame | None = None
        self._joins_demo: dict[str, dict[str, Any]] | None = None
        self._datos_prep: dict[str, Any] | None = None
        self._historial_resultados: list[dict[str, Any]] = []

        # Referencias vivas del hilo activo (evita recoleccion prematura por GC)
        self._hilo_activo: Any = None
        self._worker_activo: Any = None

        self._inicializar_vista()
        self._conectar_senales()

    def _inicializar_vista(self) -> None:
        """Configura contenido inicial, rutas absolutas y bloqueo secuencial de pestanas."""
        self.txt_negocio.setHtml(TEXTO_NEGOCIO_HTML)

        ruta_train_def = RAIZ_PROYECTO / "data" / "UNSW_NB15_training-set.csv"
        ruta_test_def = RAIZ_PROYECTO / "data" / "UNSW_NB15_testing-set.csv"
        self.txt_ruta_train.setText(str(ruta_train_def))
        self.txt_ruta_test.setText(str(ruta_test_def))

        # Bloquear fases posteriores hasta completar la anterior
        self.tab_crisp.setTabEnabled(2, False)
        self.tab_crisp.setTabEnabled(3, False)
        self.tab_crisp.setTabEnabled(4, False)

        # Configurar encabezados de tablas fijas
        self.tbl_historial.setColumnCount(len(COLUMNAS_HISTORIAL))
        self.tbl_historial.setHorizontalHeaderLabels(list(COLUMNAS_HISTORIAL))
        self.tbl_historial.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        self.tbl_metricas_actual.setColumnCount(3)
        self.tbl_metricas_actual.setHorizontalHeaderLabels(["Metrica", "Valor", "K-Fold (Media ± Std)"])
        self.tbl_metricas_actual.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self._actualizar_controles_modelo()
        self._actualizar_controles_validacion()

    def _conectar_senales(self) -> None:
        """Enlaza todos los eventos de la vista con sus slots del controlador."""
        self.btn_buscar_train.clicked.connect(lambda: self._seleccionar_csv(self.txt_ruta_train))
        self.btn_buscar_test.clicked.connect(lambda: self._seleccionar_csv(self.txt_ruta_test))
        self.btn_cargar.clicked.connect(self._on_cargar_datos)
        self.cmb_join.currentTextChanged.connect(self._on_cambiar_join)

        self.btn_preprocesar.clicked.connect(self._on_preprocesar)

        self.cmb_modelo.currentIndexChanged.connect(self._actualizar_controles_modelo)
        self.cmb_validacion.currentIndexChanged.connect(self._actualizar_controles_validacion)
        self.btn_entrenar.clicked.connect(self._on_entrenar)

        self.tbl_historial.itemSelectionChanged.connect(self._on_seleccionar_corrida)
        self.btn_limpiar_historial.clicked.connect(self._on_limpiar_historial)

    # ------------------------------------------------------------------
    # Utilidades de UI, tablas, graficas y control de hilos
    # ------------------------------------------------------------------
    def _set_ocupado(self, ocupado: bool, mensaje: str) -> None:
        """Bloquea botones de accion y activa la barra de progreso indeterminada."""
        self.lbl_estado.setText(mensaje)
        self.progress_bar.setRange(0, 0 if ocupado else 100)
        self.progress_bar.setValue(0)

        for boton in (
            self.btn_cargar,
            self.btn_buscar_train,
            self.btn_buscar_test,
            self.btn_preprocesar,
            self.btn_entrenar,
            self.btn_limpiar_historial,
        ):
            boton.setEnabled(not ocupado)

    def _lanzar_tarea(self, funcion: Any, on_ok: Any, mensaje: str, *args: Any, **kwargs: Any) -> None:
        """Despacha una funcion en QThread evitando ejecuciones concurrentes."""
        if self._hilo_activo is not None:
            QMessageBox.information(self, "Proceso en curso", "Espera a que termine la tarea actual.")
            return

        self._set_ocupado(True, mensaje)

        def _al_finalizar() -> None:
            self._hilo_activo = None
            self._worker_activo = None
            self._set_ocupado(False, "Listo.")

        def _al_error(detalle: str) -> None:
            QMessageBox.critical(self, "Error en ejecucion", detalle)

        self._hilo_activo, self._worker_activo = iniciar_tarea_en_hilo(
            funcion,
            on_ok,
            _al_error,
            _al_finalizar,
            *args,
            **kwargs,
        )

    @staticmethod
    def _limpiar_layout(layout: QLayout) -> None:
        """Destruye widgets previos de un layout contenedor para evitar fugas de memoria."""
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _montar_figura(self, layout: QLayout, figura: Figure) -> None:
        """Reemplaza el lienzo anterior por un nuevo FigureCanvasQTAgg."""
        self._limpiar_layout(layout)
        canvas = FigureCanvasQTAgg(figura)
        layout.addWidget(canvas)
        canvas.draw()

    @staticmethod
    def _llenar_tabla_df(tabla: QTableWidget, df: pd.DataFrame) -> None:
        """Puebla un QTableWidget a partir de un DataFrame pequeno (preview o muestra)."""
        tabla.clearContents()
        tabla.setRowCount(len(df))
        tabla.setColumnCount(len(df.columns))
        tabla.setHorizontalHeaderLabels([str(c) for c in df.columns])

        for fila_idx, (_, fila) in enumerate(df.iterrows()):
            for col_idx, valor in enumerate(fila):
                texto = "NaN" if pd.isna(valor) else str(valor)
                item = QTableWidgetItem(texto)
                tabla.setItem(fila_idx, col_idx, item)

        tabla.resizeColumnsToContents()

    def _seleccionar_csv(self, campo_texto: Any) -> None:
        ruta, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar archivo CSV",
            str(RAIZ_PROYECTO / "data"),
            "Archivos CSV (*.csv);;Todos los archivos (*)",
        )
        if ruta:
            campo_texto.setText(ruta)

    # ------------------------------------------------------------------
    # Pestana 2: Entendimiento e Integracion
    # ------------------------------------------------------------------
    def _on_cargar_datos(self) -> None:
        ruta_train = self.txt_ruta_train.text().strip()
        ruta_test = self.txt_ruta_test.text().strip()
        if not ruta_train or not ruta_test:
            QMessageBox.warning(self, "Rutas incompletas", "Especifica los archivos de entrenamiento y prueba.")
            return

        self._lanzar_tarea(
            _tarea_cargar_y_joins,
            self._on_datos_cargados,
            "Cargando CSV, integrando registros y calculando joins...",
            ruta_train,
            ruta_test,
        )

    def _on_datos_cargados(self, payload: dict[str, Any]) -> None:
        self._df_integrado = payload["df"]
        self._joins_demo = payload["joins"]
        res = payload["resumen"]

        # Reiniciar fases dependientes para evitar estado obsoleto
        self._datos_prep = None
        self._on_limpiar_historial()
        self.txt_resumen_prep.clear()
        self.tbl_outliers.setRowCount(0)
        self._limpiar_layout(self.layout_canvas_outliers)
        self.tab_crisp.setTabEnabled(3, False)
        self.tab_crisp.setTabEnabled(4, False)

        self.lbl_resumen_datos.setText(
            f"Train: {res['filas_train']:,} | Test: {res['filas_test']:,} | "
            f"Integrado: {res['filas_total']:,} filas x {res['columnas_total']} cols | "
            f"IDs duplicados corregidos: {res['ids_duplicados_antes']:,} -> {res['ids_duplicados_despues']} | "
            f"Nulos estandar: {res['nulos_estandar']}"
        )

        self._llenar_tabla_df(self.tbl_preview, res["preview"])
        fig_clases = graficar_distribucion_clases(res["distribucion_clases"])
        self._montar_figura(self.layout_canvas_clases, fig_clases)

        self.cmb_join.setEnabled(True)
        self._on_cambiar_join(self.cmb_join.currentText())

        self.tab_crisp.setTabEnabled(2, True)

    def _on_cambiar_join(self, tipo_join: str) -> None:
        if not self._joins_demo or tipo_join not in self._joins_demo:
            return

        info = self._joins_demo[tipo_join]
        self.lbl_join_resumen.setText(
            f"{tipo_join.upper()} -> Filas: {info['filas']:,} | "
            f"Nulos en nivel_severidad: {info['nulos_severidad']:,} | "
            f"Nulos en id: {info['nulos_id']:,}"
        )

        # Mostrar filas sin coincidencia si existen; si es INNER (0 nulos), mostrar preview
        muestra = info["muestra_sin_coincidencia"]
        if muestra.empty:
            muestra = info["muestra_preview"].head(5)
        self._llenar_tabla_df(self.tbl_join_muestra, muestra)

    # ------------------------------------------------------------------
    # Pestana 3: Preprocesamiento
    # ------------------------------------------------------------------
    def _on_preprocesar(self) -> None:
        if self._df_integrado is None:
            QMessageBox.warning(self, "Sin datos", "Primero carga e integra los datos en la Pestana 2.")
            return

        metodo_escalado = self.cmb_escalador.currentText()
        test_size = round(float(self.spn_test_size.value()), 2)

        self._lanzar_tarea(
            _tarea_preprocesar,
            self._on_preprocesamiento_listo,
            f"Limpiando, codificando y escalando ({metodo_escalado})...",
            self._df_integrado,
            metodo_escalado,
            test_size,
        )

    def _on_preprocesamiento_listo(self, payload: dict[str, Any]) -> None:
        self._datos_prep = payload
        r_lim = payload["res_limpieza"]
        r_tr = payload["res_transformacion"]
        r_esc = payload["res_escalado"]

        # Limpiar historial previo porque cambio la configuracion de escalado/particion
        self._on_limpiar_historial()
        self.tab_crisp.setTabEnabled(4, False)

        resumen_txt = (
            f"[LIMPIEZA]\n"
            f"- Filas antes: {r_lim['filas_antes']:,} | Despues: {r_lim['filas_despues']:,} "
            f"(Duplicados eliminados: {r_lim['duplicados_eliminados']:,})\n"
            f"- Valores ocultos '-' en service imputados como 'unknown': {r_lim['nulos_ocultos_service']:,} "
            f"| Nulos restantes: {r_lim['nulos_restantes']}\n\n"
            f"[TRANSFORMACION]\n"
            f"- Memoria: {r_tr['memoria_antes_mb']} MB -> {r_tr['memoria_despues_mb']} MB "
            f"(ahorro: {r_tr['ahorro_memoria_pct']}%)\n"
            f"- Nuevas variables: {', '.join(r_tr['nuevas_variables'])}\n"
            f"- One-Hot Encoding: {r_tr['columnas_antes_ohe']} -> {r_tr['columnas_despues_ohe']} columnas\n\n"
            f"[ESCALADO Y HOLD-OUT]\n"
            f"- Escalador: {r_esc['metodo']} (ajustado solo en X_train)\n"
            f"- X_train: {r_esc['shape_train']} | X_test: {r_esc['shape_test']} "
            f"(nulos: {r_esc['nulos_train'] + r_esc['nulos_test']})"
        )
        self.txt_resumen_prep.setPlainText(resumen_txt)

        # Poblar tabla IQR / p99
        outliers = r_lim["resumen_outliers"]
        cols_tabla = ["Variable", "Outliers IQR", "% IQR", "Max Antes", "Tope p99", "Max Despues"]
        self.tbl_outliers.clearContents()
        self.tbl_outliers.setColumnCount(len(cols_tabla))
        self.tbl_outliers.setHorizontalHeaderLabels(cols_tabla)
        self.tbl_outliers.setRowCount(len(outliers))

        for row_idx, (var, stats) in enumerate(outliers.items()):
            valores = [
                var,
                f"{stats['outliers_iqr']:,}",
                f"{stats['porcentaje_iqr']:.2f}%",
                f"{stats['max_antes']:,.2f}",
                f"{stats['limite_p99']:,.2f}",
                f"{stats['max_despues']:,.2f}",
            ]
            for col_idx, val in enumerate(valores):
                self.tbl_outliers.setItem(row_idx, col_idx, QTableWidgetItem(val))
        self.tbl_outliers.resizeColumnsToContents()

        fig_outliers = graficar_outliers_p99(r_lim["datos_boxplot"])
        self._montar_figura(self.layout_canvas_outliers, fig_outliers)

        self.tab_crisp.setTabEnabled(3, True)

    # ------------------------------------------------------------------
    # Pestana 4: Modelado
    # ------------------------------------------------------------------
    def _actualizar_controles_modelo(self) -> None:
        es_logistica = self.cmb_modelo.currentIndex() == 0
        for widget in (self.lbl_c, self.spn_c, self.lbl_max_iter, self.spn_max_iter):
            widget.setVisible(es_logistica)
        for widget in (self.lbl_clusters, self.spn_clusters):
            widget.setVisible(not es_logistica)

    def _actualizar_controles_validacion(self) -> None:
        es_kfold = self.cmb_validacion.currentIndex() == 1
        self.lbl_folds.setEnabled(es_kfold)
        self.spn_folds.setEnabled(es_kfold)

    def _on_entrenar(self) -> None:
        if self._datos_prep is None:
            QMessageBox.warning(self, "Falta preprocesamiento", "Ejecuta primero la Pestana 3.")
            return

        modelo_clave = "logistic_regression" if self.cmb_modelo.currentIndex() == 0 else "kmeans"
        validacion = "holdout" if self.cmb_validacion.currentIndex() == 0 else "stratified_kfold"

        if modelo_clave == "logistic_regression":
            hiperparametros: dict[str, Any] = {
                "C": float(self.spn_c.value()),
                "max_iter": int(self.spn_max_iter.value()),
            }
        else:
            hiperparametros = {
                "n_clusters": int(self.spn_clusters.value()),
                "n_init": 10,
            }

        kwargs: dict[str, Any] = {
            "modelo_clave": modelo_clave,
            "validacion": validacion,
            "hiperparametros": hiperparametros,
            "random_state": 42,
        }

        if validacion == "holdout":
            kwargs.update({
                "X_train": self._datos_prep["X_train"],
                "X_test": self._datos_prep["X_test"],
                "y_train": self._datos_prep["y_train"],
                "y_test": self._datos_prep["y_test"],
            })
        else:
            kwargs.update({
                "X_encoded": self._datos_prep["X_encoded"],
                "y": self._datos_prep["y"],
                "metodo_escalado": self._datos_prep["metodo_escalado"],
                "n_splits": int(self.spn_folds.value()),
            })

        self._lanzar_tarea(
            entrenar_y_evaluar,
            self._on_modelo_entrenado,
            f"Entrenando {self.cmb_modelo.currentText()} ({self.cmb_validacion.currentText()})...",
            **kwargs,
        )

    def _on_modelo_entrenado(self, resultado: dict[str, Any]) -> None:
        self._historial_resultados.append(resultado)
        m = resultado["metricas"]
        auc_txt = f"{m['auc_roc']:.4f}" if m["auc_roc"] is not None else "N/A"

        self.txt_log_modelo.appendPlainText(
            f"[OK] {resultado['modelo_nombre']} | {resultado['validacion']} | "
            f"Acc={m['exactitud']:.4f}, Recall={m['sensibilidad']:.4f}, "
            f"Spec={m['especificidad']:.4f}, F1={m['f1_score']:.4f}, AUC={auc_txt} "
            f"({resultado['tiempo_seg']}s)"
        )

        # Agregar fila a tabla comparativa
        row_idx = self.tbl_historial.rowCount()
        self.tbl_historial.insertRow(row_idx)
        valores_fila = [
            str(row_idx + 1),
            resultado["modelo_nombre"],
            resultado["validacion"],
            f"{m['exactitud']:.4f}",
            f"{m['precision']:.4f}",
            f"{m['sensibilidad']:.4f}",
            f"{m['especificidad']:.4f}",
            f"{m['f1_score']:.4f}",
            auc_txt,
            f"{resultado['tiempo_seg']:.2f}",
        ]
        for col_idx, val in enumerate(valores_fila):
            item = QTableWidgetItem(val)
            if col_idx not in (1, 2):
                item.setTextAlignment(Qt.AlignCenter)
            self.tbl_historial.setItem(row_idx, col_idx, item)

        # Actualizar curva ROC comparativa con todas las corridas acumuladas
        fig_roc = graficar_curva_roc(self._historial_resultados)
        self._montar_figura(self.layout_canvas_roc, fig_roc)

        self.tab_crisp.setTabEnabled(4, True)
        self.tbl_historial.selectRow(row_idx)
        self._mostrar_detalle_corrida(row_idx)
        self.tab_crisp.setCurrentIndex(4)

    # ------------------------------------------------------------------
    # Pestana 5: Evaluacion y Comparacion
    # ------------------------------------------------------------------
    def _on_seleccionar_corrida(self) -> None:
        fila = self.tbl_historial.currentRow()
        if 0 <= fila < len(self._historial_resultados):
            self._mostrar_detalle_corrida(fila)

    def _mostrar_detalle_corrida(self, indice: int) -> None:
        res = self._historial_resultados[indice]
        metricas = res["metricas"]
        por_fold = res["metricas_por_fold"]

        self.tbl_metricas_actual.setRowCount(len(ETIQUETAS_METRICAS))
        for row_idx, (clave, nombre) in enumerate(ETIQUETAS_METRICAS):
            val = metricas[clave]
            val_txt = f"{val:.4f}" if val is not None else "N/A"

            if por_fold and por_fold.get(clave) is not None:
                fold_info = por_fold[clave]
                fold_txt = f"{fold_info['media']:.4f} ± {fold_info['std']:.4f}"
            else:
                fold_txt = "-"

            self.tbl_metricas_actual.setItem(row_idx, 0, QTableWidgetItem(nombre))
            self.tbl_metricas_actual.setItem(row_idx, 1, QTableWidgetItem(val_txt))
            self.tbl_metricas_actual.setItem(row_idx, 2, QTableWidgetItem(fold_txt))

        titulo_cm = f"{res['modelo_nombre']} ({res['validacion']})"
        fig_cm = graficar_matriz_confusion(res["matriz_confusion"], titulo=titulo_cm)
        self._montar_figura(self.layout_canvas_matriz, fig_cm)

    def _on_limpiar_historial(self) -> None:
        self._historial_resultados.clear()
        self.tbl_historial.setRowCount(0)
        self.tbl_metricas_actual.setRowCount(0)
        self._limpiar_layout(self.layout_canvas_matriz)
        self._limpiar_layout(self.layout_canvas_roc)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Evita cerrar la ventana mientras un QThread secundario sigue ejecutandose."""
        if self._hilo_activo is not None and self._hilo_activo.isRunning():
            QMessageBox.warning(
                self,
                "Proceso activo",
                "Espera a que finalice la ejecucion en segundo plano antes de cerrar.",
            )
            event.ignore()
            return
        event.accept()