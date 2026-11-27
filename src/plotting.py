"""
Modulo de visualizacion orientado a objetos para PyQt5 (dataset UNSW-NB15).

Todas las funciones son puras: reciben datos/resultados y devuelven un objeto
`matplotlib.figure.Figure` listo para incrustarse en `FigureCanvasQTAgg`.
Prohibido el uso de `matplotlib.pyplot` para evitar estado global y fugas de RAM.
"""

from typing import Any

from matplotlib.figure import Figure
import numpy as np
import seaborn as sns

COLOR_NORMAL = "#2E86AB"
COLOR_ATAQUE = "#D64550"
COLOR_ANTES = "#E07A5F"
COLOR_DESPUES = "#3D85C6"


def _validar_matriz_confusion(cm: np.ndarray) -> np.ndarray:
    """Valida que la matriz de confusion sea un arreglo numerico 2x2."""
    matriz = np.asarray(cm)
    if matriz.shape != (2, 2):
        raise ValueError(f"Se esperaba una matriz de confusion 2x2, se recibio {matriz.shape}.")
    return matriz


def graficar_distribucion_clases(
    distribucion_clases: dict[int, int],
    titulo: str = "Distribucion de Clases (Normal vs. Ataque)",
) -> Figure:
    """
    Pestana 2 (EDA): Grafica de barras con conteo absoluto y porcentaje
    para trafico Normal (0) y Ataque (1).
    """
    conteo_normal = int(distribucion_clases.get(0, 0))
    conteo_ataque = int(distribucion_clases.get(1, 0))
    total = conteo_normal + conteo_ataque
    if total == 0:
        raise ValueError("distribucion_clases no contiene registros para graficar.")

    etiquetas = ["Normal (0)", "Ataque (1)"]
    valores = [conteo_normal, conteo_ataque]
    colores = [COLOR_NORMAL, COLOR_ATAQUE]

    with sns.axes_style("whitegrid"):
        fig = Figure(figsize=(6.0, 4.0), dpi=100)
        ax = fig.add_subplot(111)

        barras = ax.bar(etiquetas, valores, color=colores, width=0.55, edgecolor="#222222", linewidth=0.7)
        ax.set_title(titulo, fontsize=11, fontweight="bold", pad=10)
        ax.set_ylabel("Cantidad de registros", fontsize=9)
        ax.set_ylim(0, max(valores) * 1.18)

        for barra, val in zip(barras, valores):
            pct = (val / total) * 100.0
            ax.text(
                barra.get_x() + barra.get_width() / 2.0,
                val + (max(valores) * 0.025),
                f"{val:,}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        fig.tight_layout()
    return fig


def graficar_outliers_p99(
    datos_boxplot: dict[str, dict[str, np.ndarray]],
) -> Figure:
    """
    Pestana 3 (Limpieza): Cuadricula 2x2 de boxplots horizontales comparando
    las variables clave antes y despues del tope en el percentil 99.
    """
    variables = [col for col in ("dur", "sbytes") if col in datos_boxplot]
    if not variables:
        raise ValueError("datos_boxplot debe contener al menos 'dur' o 'sbytes'.")

    n_filas = len(variables)
    flier_style = {"marker": ".", "markersize": 2, "alpha": 0.35, "markerfacecolor": "#555555"}

    with sns.axes_style("whitegrid"):
        fig = Figure(figsize=(9.0, 3.2 * n_filas), dpi=100)
        axes = fig.subplots(n_filas, 2, squeeze=False)

        for fila_idx, col in enumerate(variables):
            serie_antes = np.asarray(datos_boxplot[col]["antes"], dtype=np.float32)
            serie_despues = np.asarray(datos_boxplot[col]["despues"], dtype=np.float32)

            ax_antes = axes[fila_idx, 0]
            sns.boxplot(x=serie_antes, ax=ax_antes, color=COLOR_ANTES, flierprops=flier_style)
            ax_antes.set_title(f"{col} - Antes del tope (p99)", fontsize=10, fontweight="bold")
            ax_antes.set_xlabel("")

            ax_despues = axes[fila_idx, 1]
            sns.boxplot(x=serie_despues, ax=ax_despues, color=COLOR_DESPUES, flierprops=flier_style)
            ax_despues.set_title(f"{col} - Despues del tope (p99)", fontsize=10, fontweight="bold")
            ax_despues.set_xlabel("")

        fig.tight_layout()
    return fig


def graficar_matriz_confusion(
    cm: np.ndarray,
    titulo: str = "Matriz de Confusion",
) -> Figure:
    """
    Pestana 5 (Evaluacion): Mapa de calor 2x2 con etiquetas explicitas
    (Normal / Ataque), conteo absoluto y porcentaje sobre el total.
    """
    matriz = _validar_matriz_confusion(cm)
    total = int(matriz.sum())
    etiquetas_clases = ["Normal (0)", "Ataque (1)"]

    anotaciones = np.empty((2, 2), dtype=object)
    nombres_celda = (("VN", "FP"), ("FN", "VP"))
    for i in range(2):
        for j in range(2):
            val = int(matriz[i, j])
            pct = (val / total) * 100.0 if total > 0 else 0.0
            anotaciones[i, j] = f"{nombres_celda[i][j]}\n{val:,}\n({pct:.1f}%)"

    with sns.axes_style("white"):
        fig = Figure(figsize=(5.5, 4.5), dpi=100)
        ax = fig.add_subplot(111)

        sns.heatmap(
            matriz,
            annot=anotaciones,
            fmt="",
            cmap="Blues",
            cbar=True,
            square=True,
            linewidths=0.8,
            linecolor="#CCCCCC",
            xticklabels=etiquetas_clases,
            yticklabels=etiquetas_clases,
            ax=ax,
        )
        ax.set_title(titulo, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Prediccion del modelo", fontsize=9, fontweight="bold")
        ax.set_ylabel("Valor real", fontsize=9, fontweight="bold")

        fig.tight_layout()
    return fig


def graficar_curva_roc(
    historial_resultados: list[dict[str, Any]],
    titulo: str = "Curva ROC Comparativa",
) -> Figure:
    """
    Pestana 5 (Evaluacion): Traza la curva ROC de cada corrida que cuente con
    probabilidades calibradas. Si solo existen modelos sin probabilidades
    (como K-Means), renderiza un mensaje informativo limpio.
    """
    corridas_con_roc = [
        res for res in historial_resultados
        if res.get("roc_datos") is not None
    ]

    with sns.axes_style("whitegrid"):
        fig = Figure(figsize=(5.8, 4.5), dpi=100)
        ax = fig.add_subplot(111)

        if not corridas_con_roc:
            ax.text(
                0.5,
                0.5,
                "Curva ROC no disponible para los modelos actuales.\n"
                "(K-Means es no supervisado y no genera probabilidades calibradas)",
                ha="center",
                va="center",
                fontsize=9.5,
                color="#444444",
                transform=ax.transAxes,
            )
            ax.set_title(titulo, fontsize=11, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.tight_layout()
            return fig

        for res in corridas_con_roc:
            roc = res["roc_datos"]
            etiqueta = f"{res['modelo_nombre']} [{res['validacion']}] (AUC = {roc['auc']:.4f})"
            ax.plot(roc["fpr"], roc["tpr"], linewidth=2.0, label=etiqueta)

        ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#888888", linewidth=1.2, label="Azar (AUC = 0.5000)")
        ax.set_title(titulo, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Tasa de Falsos Positivos (1 - Especificidad)", fontsize=9)
        ax.set_ylabel("Tasa de Verdaderos Positivos (Sensibilidad)", fontsize=9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="lower right", fontsize=8, frameon=True)

        fig.tight_layout()
    return fig


if __name__ == "__main__":
    # Smoke test headless: valida construccion y renderizado real sin abrir ventanas ni usar pyplot
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    rng = np.random.default_rng(42)

    fig_clases = graficar_distribucion_clases({0: 93000, 1: 164673})
    fig_outliers = graficar_outliers_p99({
        "dur": {"antes": rng.exponential(scale=5.0, size=1000), "despues": rng.exponential(scale=1.0, size=1000)},
        "sbytes": {"antes": rng.exponential(scale=5000.0, size=1000), "despues": rng.exponential(scale=800.0, size=1000)},
    })
    fig_cm = graficar_matriz_confusion(np.array([[13523, 3621], [1003, 14402]]), "Regresion Logistica")
    fig_roc_vacia = graficar_curva_roc([{"modelo_nombre": "K-Means", "validacion": "holdout", "roc_datos": None}])
    fig_roc_ok = graficar_curva_roc([{
        "modelo_nombre": "Regresion Logistica",
        "validacion": "holdout",
        "roc_datos": {"fpr": np.array([0.0, 0.2, 1.0]), "tpr": np.array([0.0, 0.93, 1.0]), "auc": 0.9536},
    }])

    for nombre, figura in (
        ("distribucion_clases", fig_clases),
        ("outliers_p99", fig_outliers),
        ("matriz_confusion", fig_cm),
        ("roc_sin_proba", fig_roc_vacia),
        ("roc_con_proba", fig_roc_ok),
    ):
        FigureCanvasAgg(figura).draw()
        print(f"[OK] {nombre}: renderizada correctamente ({figura.get_size_inches()})")