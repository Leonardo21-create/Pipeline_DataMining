"""
Modulo de preprocesamiento para el dataset UNSW-NB15.

Disenado para ejecutarse paso a paso desde una interfaz PyQt5 (CRISP-DM)
o de forma completa mediante `ejecutar_pipeline`. Sin dependencias de GUI ni graficas.
"""

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Constantes inmutables de esquema y tratamiento
COLS_OUTLIERS: tuple[str, ...] = ("dur", "sbytes", "dbytes", "rate", "sload", "dload")
COLS_BOXPLOT_DEMO: tuple[str, ...] = ("dur", "sbytes")
COLS_CATEGORICAS_OHE: tuple[str, ...] = ("proto", "service", "state")
COLS_REQUERIDAS: tuple[str, ...] = (
    "id", "dur", "proto", "service", "state",
    "spkts", "dpkts", "sbytes", "dbytes",
    "rate", "sload", "dload", "attack_cat", "label",
)

CATALOGO_SEVERIDAD_DEMO: tuple[tuple[str, str], ...] = (
    ("Normal", "Ninguna"),
    ("Reconnaissance", "Baja"),
    ("Analysis", "Baja"),
    ("Fuzzers", "Media"),
    ("DoS", "Alta"),
    ("Generic", "Alta"),
    ("Exploits", "Alta"),
    ("Shellcode", "Critica"),
    ("Backdoor", "Critica"),
    ("ZeroDay", "Critica"),  # Categoria inexistente en UNSW-NB15 para evidenciar right/outer
)


def validar_esquema(df: pd.DataFrame, columnas_requeridas: tuple[str, ...] = COLS_REQUERIDAS) -> None:
    """Valida que el DataFrame contenga las columnas minimas antes de procesar."""
    faltantes = [col for col in columnas_requeridas if col not in df.columns]
    if faltantes:
        raise ValueError(f"Esquema invalido. Faltan columnas requeridas: {faltantes}")


def cargar_e_integrar_datos(
    ruta_train: str | Path,
    ruta_test: str | Path,
    retornar_resumen: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """
    Pestana 2 (Entendimiento e Integracion): Carga ambos CSV, valida esquema,
    concatena en un solo DataFrame y regenera identificadores consecutivos.
    """
    ruta_train_path = Path(ruta_train)
    ruta_test_path = Path(ruta_test)

    if not ruta_train_path.exists():
        raise FileNotFoundError(f"No se encontro el archivo de entrenamiento: {ruta_train_path}")
    if not ruta_test_path.exists():
        raise FileNotFoundError(f"No se encontro el archivo de prueba: {ruta_test_path}")

    df_train = pd.read_csv(ruta_train_path)
    df_test = pd.read_csv(ruta_test_path)

    validar_esquema(df_train)
    validar_esquema(df_test)

    if list(df_train.columns) != list(df_test.columns):
        raise ValueError("Los archivos train y test no tienen el mismo orden o conjunto de columnas.")

    df = pd.concat([df_train, df_test], ignore_index=True)
    ids_duplicados_antes = int(df["id"].duplicated().sum())
    df["id"] = range(1, len(df) + 1)

    if not retornar_resumen:
        return df

    resumen = {
        "filas_train": int(len(df_train)),
        "filas_test": int(len(df_test)),
        "filas_total": int(len(df)),
        "columnas_total": int(df.shape[1]),
        "ids_duplicados_antes": ids_duplicados_antes,
        "ids_duplicados_despues": int(df["id"].duplicated().sum()),
        "nulos_estandar": int(df.isna().sum().sum()),
        "distribucion_clases": df["label"].value_counts().sort_index().to_dict(),
        "distribucion_ataques": df["attack_cat"].astype(str).str.strip().value_counts().to_dict(),
        "preview": df.head(50).copy(),
    }
    return df, resumen


def ejecutar_demo_joins(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Pestana 2 (Integracion): Ejecuta inner, left, right y outer join contra un
    catalogo de severidad sin 'Worms' y con 'ZeroDay' para demostracion en vivo.
    """
    validar_esquema(df, ("id", "attack_cat", "label"))

    df_base = df[["id", "attack_cat", "label"]].copy()
    df_base["attack_cat"] = df_base["attack_cat"].astype(str).str.strip()

    tabla_severidad = pd.DataFrame(
        CATALOGO_SEVERIDAD_DEMO,
        columns=["attack_cat", "nivel_severidad"],
    )

    resultados: dict[str, dict[str, Any]] = {}
    for tipo_join in ("inner", "left", "right", "outer"):
        unido = pd.merge(df_base, tabla_severidad, on="attack_cat", how=tipo_join)
        nulos_sev = int(unido["nivel_severidad"].isna().sum())
        nulos_id = int(unido["id"].isna().sum())

        # Extraer muestra pequena de filas con NaN (si existen) para la tabla en Qt
        sin_coincidencia = unido[unido["nivel_severidad"].isna() | unido["id"].isna()].head(5).copy()

        resultados[tipo_join] = {
            "filas": int(len(unido)),
            "nulos_severidad": nulos_sev,
            "nulos_id": nulos_id,
            "muestra_preview": unido.head(10).copy(),
            "muestra_sin_coincidencia": sin_coincidencia,
        }

    return resultados


def limpiar_datos(
    df: pd.DataFrame,
    retornar_resumen: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """
    Pestana 3 (Limpieza): Reemplaza '-' en service por 'unknown', elimina
    duplicados exactos excluyendo id, estandariza texto y aplica clipping p99.
    """
    validar_esquema(df)
    df = df.copy()
    filas_antes = int(len(df))

    # 1) Valores ocultos ('-') en service
    nulos_ocultos_service = int((df["service"] == "-").sum())
    df["service"] = df["service"].replace("-", np.nan).fillna("unknown")

    # 2) Duplicados reales excluyendo id
    cols_sin_id = [col for col in df.columns if col != "id"]
    df = df.drop_duplicates(subset=cols_sin_id, keep="first").reset_index(drop=True)
    df["id"] = range(1, len(df) + 1)
    filas_despues = int(len(df))
    duplicados_eliminados = filas_antes - filas_despues

    # 3) Estandarizacion de texto
    for col in COLS_CATEGORICAS_OHE:
        df[col] = df[col].astype(str).str.strip().str.lower()
    df["attack_cat"] = df["attack_cat"].astype(str).str.strip()

    # 4) Deteccion IQR y clipping p99 (solo clonamos series necesarias para boxplot)
    datos_boxplot: dict[str, dict[str, np.ndarray]] = {}
    if retornar_resumen:
        for col in COLS_BOXPLOT_DEMO:
            datos_boxplot[col] = {"antes": df[col].to_numpy(dtype=np.float32, copy=True)}

    resumen_outliers: dict[str, dict[str, float | int]] = {}
    for col in COLS_OUTLIERS:
        q1 = float(df[col].quantile(0.25))
        q3 = float(df[col].quantile(0.75))
        iqr = q3 - q1
        lim_inf = q1 - 1.5 * iqr
        lim_sup = q3 + 1.5 * iqr
        cant_iqr = int(((df[col] < lim_inf) | (df[col] > lim_sup)).sum())

        max_antes = float(df[col].max())
        limite_p99 = float(df[col].quantile(0.99))
        df[col] = df[col].clip(upper=limite_p99)
        max_despues = float(df[col].max())

        if retornar_resumen:
            resumen_outliers[col] = {
                "outliers_iqr": cant_iqr,
                "porcentaje_iqr": round((cant_iqr / filas_despues) * 100.0, 2),
                "max_antes": round(max_antes, 2),
                "limite_p99": round(limite_p99, 2),
                "max_despues": round(max_despues, 2),
            }

    if retornar_resumen:
        for col in COLS_BOXPLOT_DEMO:
            datos_boxplot[col]["despues"] = df[col].to_numpy(dtype=np.float32, copy=True)

        resumen = {
            "filas_antes": filas_antes,
            "filas_despues": filas_despues,
            "duplicados_eliminados": duplicados_eliminados,
            "nulos_ocultos_service": nulos_ocultos_service,
            "nulos_restantes": int(df.isna().sum().sum()),
            "resumen_outliers": resumen_outliers,
            "datos_boxplot": datos_boxplot,
        }
        return df, resumen

    return df


def transformar_datos(
    df: pd.DataFrame,
    columna_objetivo: Literal["label", "attack_cat"] = "label",
    retornar_resumen: bool = False,
) -> tuple[pd.DataFrame, pd.Series] | tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """
    Pestana 3 (Transformacion): Optimiza tipos numericos, crea 3 variables
    de trafico de red, agrupa proto fuera del Top 10 y aplica One-Hot Encoding.
    Devuelve X_encoded sin escalar (compatible con Hold-out y Stratified K-Fold).
    """
    validar_esquema(df)
    if columna_objetivo not in df.columns:
        raise ValueError(f"La columna objetivo '{columna_objetivo}' no existe en el DataFrame.")

    df = df.copy()
    mem_antes_mb = float(df.memory_usage(deep=True).sum() / (1024 ** 2))

    # 1) Optimizacion de tipos
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")

    for col in (*COLS_CATEGORICAS_OHE, "attack_cat"):
        df[col] = df[col].astype("category")

    mem_despues_mb = float(df.memory_usage(deep=True).sum() / (1024 ** 2))
    ahorro_pct = ((mem_antes_mb - mem_despues_mb) / mem_antes_mb) * 100.0

    # 2) Feature Engineering
    df["total_pkts"] = (df["spkts"].astype("int32") + df["dpkts"].astype("int32")).astype("int32")
    df["total_bytes"] = (df["sbytes"].astype("float32") + df["dbytes"].astype("float32")).astype("float32")
    df["bytes_ratio"] = (df["sbytes"].astype("float32") / (df["total_bytes"] + 1.0)).astype("float32")

    # 3) Separacion de predictores y objetivo sin target leakage
    cols_excluir = [col for col in ("id", "label", "attack_cat") if col in df.columns]
    X = df.drop(columns=cols_excluir).copy()
    y = df[columna_objetivo].copy()

    # 4) Top 10 protocolos + One-Hot Encoding con uint8 para no inflar RAM
    top_protos = X["proto"].value_counts().head(10).index.tolist()
    X["proto"] = X["proto"].astype(str)
    X["proto"] = X["proto"].where(X["proto"].isin(top_protos), "other")

    columnas_antes_ohe = int(X.shape[1])
    X_encoded = pd.get_dummies(
        X,
        columns=list(COLS_CATEGORICAS_OHE),
        drop_first=True,
        dtype=np.uint8,
    )

    if not retornar_resumen:
        return X_encoded, y

    resumen = {
        "memoria_antes_mb": round(mem_antes_mb, 2),
        "memoria_despues_mb": round(mem_despues_mb, 2),
        "ahorro_memoria_pct": round(ahorro_pct, 1),
        "nuevas_variables": ("total_pkts", "total_bytes", "bytes_ratio"),
        "top_protos": top_protos,
        "columnas_antes_ohe": columnas_antes_ohe,
        "columnas_despues_ohe": int(X_encoded.shape[1]),
    }
    return X_encoded, y, resumen


def dividir_y_escalar(
    X: pd.DataFrame,
    y: pd.Series,
    metodo: Literal["standard", "minmax"] = "standard",
    test_size: float = 0.2,
    random_state: int = 42,
    retornar_resumen: bool = False,
) -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler | MinMaxScaler]
    | tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler | MinMaxScaler, dict[str, Any]]
):
    """
    Pestana 3 (Hold-out y Escalado): Divide de forma estratificada y ajusta el
    escalador unicamente con X_train. Conserva float32 para eficiencia en RAM.
    """
    if not (0.05 <= test_size <= 0.50):
        raise ValueError("test_size debe estar entre 0.05 y 0.50.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if metodo == "minmax":
        scaler: StandardScaler | MinMaxScaler = MinMaxScaler()
    elif metodo == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError("metodo debe ser 'standard' o 'minmax'.")

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train).astype(np.float32),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test).astype(np.float32),
        columns=X_test.columns,
        index=X_test.index,
    )

    if not retornar_resumen:
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler

    resumen = {
        "metodo": metodo,
        "test_size": test_size,
        "shape_train": X_train_scaled.shape,
        "shape_test": X_test_scaled.shape,
        "nulos_train": int(X_train_scaled.isna().sum().sum()),
        "nulos_test": int(X_test_scaled.isna().sum().sum()),
        "proporcion_train": y_train.value_counts(normalize=True).round(4).to_dict(),
        "proporcion_test": y_test.value_counts(normalize=True).round(4).to_dict(),
    }
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, resumen


def ejecutar_pipeline(
    ruta_train: str | Path = "data/UNSW_NB15_training-set.csv",
    ruta_test: str | Path = "data/UNSW_NB15_testing-set.csv",
    columna_objetivo: Literal["label", "attack_cat"] = "label",
    metodo_escalado: Literal["standard", "minmax"] = "standard",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Orquestador completo para pruebas rapidas en consola o scripts externos."""
    df, res_integracion = cargar_e_integrar_datos(ruta_train, ruta_test, retornar_resumen=True)
    df_limpio, res_limpieza = limpiar_datos(df, retornar_resumen=True)
    X_encoded, y, res_transformacion = transformar_datos(
        df_limpio, columna_objetivo=columna_objetivo, retornar_resumen=True
    )
    X_train, X_test, y_train, y_test, scaler, res_escalado = dividir_y_escalar(
        X_encoded,
        y,
        metodo=metodo_escalado,
        test_size=test_size,
        random_state=random_state,
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
        "columnas": list(X_train.columns),
        "resumen": {
            "integracion": res_integracion,
            "limpieza": res_limpieza,
            "transformacion": res_transformacion,
            "escalado": res_escalado,
        },
    }


if __name__ == "__main__":
    # Smoke test del flujo pestana por pestana
    df_int, r_int = cargar_e_integrar_datos(
        "data/UNSW_NB15_training-set.csv",
        "data/UNSW_NB15_testing-set.csv",
        retornar_resumen=True,
    )
    joins = ejecutar_demo_joins(df_int)
    df_clean, r_clean = limpiar_datos(df_int, retornar_resumen=True)
    X_enc, y_ser, r_trans = transformar_datos(df_clean, retornar_resumen=True)
    X_tr, X_te, y_tr, y_te, _, r_esc = dividir_y_escalar(X_enc, y_ser, retornar_resumen=True)

    print("Integracion:", r_int["filas_total"], "filas | Joins LEFT:", joins["left"]["filas"])
    print("Limpieza:   ", r_clean["filas_despues"], "filas | Duplicados:", r_clean["duplicados_eliminados"])
    print("Transform.: ", r_trans["columnas_despues_ohe"], "cols | Ahorro:", r_trans["ahorro_memoria_pct"], "%")
    print("Escalado:   ", r_esc["shape_train"], "| Test:", r_esc["shape_test"])