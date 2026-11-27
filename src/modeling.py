"""
Modulo de modelado y evaluacion binaria para el dataset UNSW-NB15.

Soporta Regresion Logistica y K-Means (mapeado por clase mayoritaria en train)
bajo estrategias Hold-out y Stratified K-Fold sin fuga de datos.
Sin dependencias de PyQt5 ni Matplotlib.
"""

from pathlib import Path
import sys
import time
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

MODELOS_SOPORTADOS: dict[str, str] = {
    "logistic_regression": "Regresion Logistica",
    "kmeans": "K-Means (No Supervisado)",
}


class KMeansBinaryClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper compatible con Scikit-learn para evaluar K-Means en clasificacion binaria.
    Mapea cada cluster a la clase mayoritaria usando exclusivamente y_train en fit().
    """

    def __init__(
        self,
        n_clusters: int = 2,
        n_init: int = 10,
        random_state: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.random_state = random_state

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> "KMeansBinaryClassifier":
        y_arr = np.asarray(y, dtype=np.int8)
        self.classes_ = np.array([0, 1], dtype=np.int8)

        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            random_state=self.random_state,
        )
        clusters = self.kmeans_.fit_predict(X)

        # Asignar a cada cluster la clase mayoritaria observada solo en train
        self.cluster_to_class_: dict[int, int] = {}
        clase_global = int(np.bincount(y_arr).argmax())

        for cluster_id in range(self.n_clusters):
            mascara = clusters == cluster_id
            if np.any(mascara):
                self.cluster_to_class_[cluster_id] = int(np.bincount(y_arr[mascara]).argmax())
            else:
                self.cluster_to_class_[cluster_id] = clase_global

        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "kmeans_"):
            raise RuntimeError("El modelo KMeansBinaryClassifier debe entrenarse antes de predecir.")

        clusters = self.kmeans_.predict(X)
        return np.array([self.cluster_to_class_[int(c)] for c in clusters], dtype=np.int8)


def crear_modelo(
    modelo_clave: Literal["logistic_regression", "kmeans"],
    hiperparametros: dict[str, Any] | None = None,
    random_state: int = 42,
) -> tuple[BaseEstimator, dict[str, Any]]:
    """Fabrica de modelos con validacion estricta de parametros."""
    params = dict(hiperparametros or {})

    if modelo_clave == "logistic_regression":
        c_val = float(params.get("C", 1.0))
        max_iter = int(params.get("max_iter", 1000))
        if c_val <= 0:
            raise ValueError("El parametro C de Regresion Logistica debe ser mayor a 0.")
        if max_iter < 50:
            raise ValueError("max_iter debe ser al menos 50.")

        config_final = {"C": c_val, "max_iter": max_iter, "random_state": random_state}
        return LogisticRegression(**config_final), config_final

    if modelo_clave == "kmeans":
        n_clusters = int(params.get("n_clusters", 2))
        n_init = int(params.get("n_init", 10))
        if n_clusters < 2:
            raise ValueError("n_clusters debe ser al menos 2.")

        config_final = {"n_clusters": n_clusters, "n_init": n_init, "random_state": random_state}
        return KMeansBinaryClassifier(**config_final), config_final

    raise ValueError(
        f"Modelo no soportado: '{modelo_clave}'. Opciones validas: {list(MODELOS_SOPORTADOS)}"
    )


def _crear_escalador(metodo_escalado: Literal["standard", "minmax"]) -> StandardScaler | MinMaxScaler:
    if metodo_escalado == "standard":
        return StandardScaler()
    if metodo_escalado == "minmax":
        return MinMaxScaler()
    raise ValueError("metodo_escalado debe ser 'standard' o 'minmax'.")


def calcular_metricas(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> tuple[dict[str, float | None], np.ndarray, dict[str, Any] | None]:
    """
    Calcula las 6 metricas requeridas (incluyendo especificidad),
    la matriz de confusion 2x2 fija en orden [[TN, FP], [FN, TP]]
    y los datos de curva ROC si existen probabilidades.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int8)
    y_pred_arr = np.asarray(y_pred, dtype=np.int8)

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    especificidad = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    auc_roc: float | None = None
    roc_datos: dict[str, Any] | None = None
    if y_prob is not None and len(np.unique(y_true_arr)) == 2:
        auc_roc = round(float(roc_auc_score(y_true_arr, y_prob)), 4)
        fpr, tpr, _ = roc_curve(y_true_arr, y_prob)
        roc_datos = {"fpr": fpr, "tpr": tpr, "auc": auc_roc}

    metricas: dict[str, float | None] = {
        "exactitud": round(float(accuracy_score(y_true_arr, y_pred_arr)), 4),
        "precision": round(float(precision_score(y_true_arr, y_pred_arr, zero_division=0)), 4),
        "sensibilidad": round(float(recall_score(y_true_arr, y_pred_arr, zero_division=0)), 4),
        "especificidad": round(especificidad, 4),
        "f1_score": round(float(f1_score(y_true_arr, y_pred_arr, zero_division=0)), 4),
        "auc_roc": auc_roc,
    }

    return metricas, cm, roc_datos


def _evaluar_holdout(
    modelo_clave: Literal["logistic_regression", "kmeans"],
    X_train: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    y_test: pd.Series | np.ndarray,
    hiperparametros: dict[str, Any] | None,
    random_state: int,
) -> dict[str, Any]:
    modelo, config_final = crear_modelo(modelo_clave, hiperparametros, random_state)

    inicio = time.perf_counter()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo, "predict_proba") else None
    tiempo_seg = round(time.perf_counter() - inicio, 2)

    metricas, cm, roc_datos = calcular_metricas(y_test, y_pred, y_prob)

    return {
        "modelo_clave": modelo_clave,
        "modelo_nombre": MODELOS_SOPORTADOS[modelo_clave],
        "validacion": "holdout",
        "hiperparametros": config_final,
        "metricas": metricas,
        "metricas_por_fold": None,
        "matriz_confusion": cm,
        "roc_datos": roc_datos,
        "tiempo_seg": tiempo_seg,
    }


def _evaluar_stratified_kfold(
    modelo_clave: Literal["logistic_regression", "kmeans"],
    X_encoded: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    metodo_escalado: Literal["standard", "minmax"],
    n_splits: int,
    hiperparametros: dict[str, Any] | None,
    random_state: int,
) -> dict[str, Any]:
    if n_splits < 2 or n_splits > 10:
        raise ValueError("n_splits para Stratified K-Fold debe estar entre 2 y 10.")

    X_arr = np.asarray(X_encoded, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.int8)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_pred = np.zeros(len(y_arr), dtype=np.int8)
    usa_proba = modelo_clave == "logistic_regression"
    oof_prob = np.zeros(len(y_arr), dtype=np.float64) if usa_proba else None

    historial_folds: dict[str, list[float]] = {
        "exactitud": [],
        "precision": [],
        "sensibilidad": [],
        "especificidad": [],
        "f1_score": [],
        "auc_roc": [],
    }

    config_final: dict[str, Any] = {}
    inicio = time.perf_counter()

    for idx_train, idx_val in skf.split(X_arr, y_arr):
        modelo, config_final = crear_modelo(modelo_clave, hiperparametros, random_state)
        pipeline = Pipeline([
            ("scaler", _crear_escalador(metodo_escalado)),
            ("model", modelo),
        ])

        pipeline.fit(X_arr[idx_train], y_arr[idx_train])
        pred_fold = pipeline.predict(X_arr[idx_val])
        oof_pred[idx_val] = pred_fold

        prob_fold: np.ndarray | None = None
        if usa_proba and oof_prob is not None:
            prob_fold = pipeline.predict_proba(X_arr[idx_val])[:, 1]
            oof_prob[idx_val] = prob_fold

        metricas_fold, _, _ = calcular_metricas(y_arr[idx_val], pred_fold, prob_fold)
        for clave, valor in metricas_fold.items():
            if valor is not None:
                historial_folds[clave].append(valor)

    tiempo_seg = round(time.perf_counter() - inicio, 2)

    # Matriz de confusion y curva ROC globales a partir de predicciones fuera de muestra
    metricas_globales, cm_global, roc_datos = calcular_metricas(y_arr, oof_pred, oof_prob)

    resumen_folds: dict[str, dict[str, float] | None] = {}
    for clave, valores in historial_folds.items():
        if valores:
            resumen_folds[clave] = {
                "media": round(float(np.mean(valores)), 4),
                "std": round(float(np.std(valores)), 4),
            }
        else:
            resumen_folds[clave] = None

    return {
        "modelo_clave": modelo_clave,
        "modelo_nombre": MODELOS_SOPORTADOS[modelo_clave],
        "validacion": f"stratified_kfold (k={n_splits})",
        "hiperparametros": config_final,
        "metricas": metricas_globales,
        "metricas_por_fold": resumen_folds,
        "matriz_confusion": cm_global,
        "roc_datos": roc_datos,
        "tiempo_seg": tiempo_seg,
    }


def entrenar_y_evaluar(
    modelo_clave: Literal["logistic_regression", "kmeans"],
    validacion: Literal["holdout", "stratified_kfold"] = "holdout",
    *,
    X_train: pd.DataFrame | np.ndarray | None = None,
    X_test: pd.DataFrame | np.ndarray | None = None,
    y_train: pd.Series | np.ndarray | None = None,
    y_test: pd.Series | np.ndarray | None = None,
    X_encoded: pd.DataFrame | np.ndarray | None = None,
    y: pd.Series | np.ndarray | None = None,
    metodo_escalado: Literal["standard", "minmax"] = "standard",
    n_splits: int = 5,
    hiperparametros: dict[str, Any] | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Punto de entrada unico para PyQt5 (Pestana 4 -> Pestana 5).

    - Si validacion == 'holdout': requiere X_train, X_test, y_train, y_test (ya escalados).
    - Si validacion == 'stratified_kfold': requiere X_encoded y y (SIN escalar).
    """
    if validacion == "holdout":
        if any(dato is None for dato in (X_train, X_test, y_train, y_test)):
            raise ValueError("Hold-out requiere X_train, X_test, y_train y y_test.")
        return _evaluar_holdout(
            modelo_clave=modelo_clave,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            hiperparametros=hiperparametros,
            random_state=random_state,
        )

    if validacion == "stratified_kfold":
        if X_encoded is None or y is None:
            raise ValueError("Stratified K-Fold requiere X_encoded y y sin escalar.")
        return _evaluar_stratified_kfold(
            modelo_clave=modelo_clave,
            X_encoded=X_encoded,
            y=y,
            metodo_escalado=metodo_escalado,
            n_splits=n_splits,
            hiperparametros=hiperparametros,
            random_state=random_state,
        )

    raise ValueError("validacion debe ser 'holdout' o 'stratified_kfold'.")


if __name__ == "__main__":
    # Permite ejecutar `python src\modeling.py` desde la raiz del proyecto
    raiz_proyecto = str(Path(__file__).resolve().parent.parent)
    if raiz_proyecto not in sys.path:
        sys.path.insert(0, raiz_proyecto)

    from src.preprocessing import ejecutar_pipeline

    datos = ejecutar_pipeline(metodo_escalado="standard")

    for clave in ("logistic_regression", "kmeans"):
        res_ho = entrenar_y_evaluar(
            modelo_clave=clave,
            validacion="holdout",
            X_train=datos["X_train"],
            X_test=datos["X_test"],
            y_train=datos["y_train"],
            y_test=datos["y_test"],
        )
        print(f"\n[{res_ho['modelo_nombre']} | {res_ho['validacion']}] ({res_ho['tiempo_seg']}s)")
        print("Metricas:", res_ho["metricas"])
        print("Matriz de confusion:\n", res_ho["matriz_confusion"])

    res_kf = entrenar_y_evaluar(
        modelo_clave="logistic_regression",
        validacion="stratified_kfold",
        X_encoded=datos["X_encoded"],
        y=datos["y"],
        metodo_escalado="standard",
        n_splits=3,
    )
    print(f"\n[{res_kf['modelo_nombre']} | {res_kf['validacion']}] ({res_kf['tiempo_seg']}s)")
    print("Metricas globales OOF:", res_kf["metricas"])
    print("Resumen por fold (F1):", res_kf["metricas_por_fold"]["f1_score"])