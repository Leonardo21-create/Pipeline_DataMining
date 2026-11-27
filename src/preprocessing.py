"""
Pipeline de preprocesamiento para el dataset UNSW-NB15.

Este modulo replica exactamente los pasos validados en notebooks/preprocesamiento.ipynb
con funciones simples listas para importar en Scikit-learn y en una interfaz PyQt5.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

COLS_OUTLIERS = ["dur", "sbytes", "dbytes", "rate", "sload", "dload"]


def cargar_e_integrar_datos(ruta_train, ruta_test):
    """
    Fase 1: Carga los CSV de entrenamiento y prueba, los concatena
    y corrige el reinicio de la columna id.
    """
    df_train = pd.read_csv(ruta_train)
    df_test = pd.read_csv(ruta_test)

    df = pd.concat([df_train, df_test], ignore_index=True)
    df["id"] = range(1, len(df) + 1)

    return df


def limpiar_datos(df):
    """
    Fase 2: Maneja valores ocultos ('-' en service), elimina duplicados
    excluyendo id, estandariza texto y aplica tope en percentil 99 a outliers.
    """
    df = df.copy()

    # 1) Valores nulos ocultos en service
    df["service"] = df["service"].replace("-", np.nan)
    df["service"] = df["service"].fillna("unknown")

    # 2) Duplicados reales (excluyendo id)
    cols_sin_id = [col for col in df.columns if col != "id"]
    df = df.drop_duplicates(subset=cols_sin_id, keep="first").reset_index(drop=True)
    df["id"] = range(1, len(df) + 1)

    # 3) Estandarizacion de formato en columnas categoricas
    for col in ["proto", "service", "state"]:
        df[col] = df[col].astype(str).str.strip().str.lower()
    df["attack_cat"] = df["attack_cat"].astype(str).str.strip()

    # 4) Tratamiento de outliers con tope (clipping) en percentil 99
    for col in COLS_OUTLIERS:
        if col in df.columns:
            limite_p99 = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=limite_p99)

    return df


def transformar_datos(df, columna_objetivo="label"):
    """
    Fase 3 (parte 1): Optimiza tipos de datos, crea variables nuevas de trafico
    de red y codifica proto, service y state con One-Hot Encoding.
    Devuelve X_encoded y la serie objetivo y.
    """
    df = df.copy()

    # 1) Optimizacion basica de tipos de datos
    cols_enteras = df.select_dtypes(include=["int64"]).columns
    for col in cols_enteras:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    cols_flotantes = df.select_dtypes(include=["float64"]).columns
    for col in cols_flotantes:
        df[col] = df[col].astype("float32")

    for col in ["proto", "service", "state", "attack_cat"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # 2) Feature Engineering simple de trafico de red
    df["total_pkts"] = (df["spkts"].astype("int32") + df["dpkts"].astype("int32")).astype("int32")
    df["total_bytes"] = (df["sbytes"].astype("float32") + df["dbytes"].astype("float32")).astype("float32")
    df["bytes_ratio"] = (df["sbytes"].astype("float32") / (df["total_bytes"] + 1.0)).astype("float32")

    # 3) Separar variables predictoras (X) y objetivo (y)
    cols_excluir = [col for col in ["id", "label", "attack_cat"] if col in df.columns]
    X = df.drop(columns=cols_excluir).copy()
    y = df[columna_objetivo].copy()

    # 4) Agrupar protocolos poco frecuentes en 'other' y aplicar One-Hot Encoding
    top_protos = X["proto"].value_counts().head(10).index.tolist()
    X["proto"] = X["proto"].astype(str)
    X["proto"] = X["proto"].where(X["proto"].isin(top_protos), "other")

    X_encoded = pd.get_dummies(X, columns=["proto", "service", "state"], drop_first=True, dtype=int)

    return X_encoded, y


def dividir_y_escalar(X, y, metodo="standard", test_size=0.2, random_state=42):
    """
    Fase 3 (parte 2): Divide en train y test de forma estratificada y ajusta
    el escalador SOLO con X_train para evitar data leakage.

    Parametro metodo:
        - "standard": usa StandardScaler (media 0, desviacion 1)
        - "minmax": usa MinMaxScaler (rango 0 a 1)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if metodo == "minmax":
        scaler = MinMaxScaler()
    elif metodo == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError("metodo debe ser 'standard' o 'minmax'")

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def ejecutar_pipeline(
    ruta_train="data/UNSW_NB15_training-set.csv",
    ruta_test="data/UNSW_NB15_testing-set.csv",
    columna_objetivo="label",
    metodo_escalado="standard",
    test_size=0.2,
    random_state=42
):
    """
    Ejecuta el pipeline completo de inicio a fin.
    Pensado para llamarse facilmente desde scripts de Scikit-learn o desde PyQt5.
    """
    df = cargar_e_integrar_datos(ruta_train, ruta_test)
    df = limpiar_datos(df)
    X_encoded, y = transformar_datos(df, columna_objetivo=columna_objetivo)
    X_train, X_test, y_train, y_test, scaler = dividir_y_escalar(
        X_encoded,
        y,
        metodo=metodo_escalado,
        test_size=test_size,
        random_state=random_state
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "columnas": list(X_train.columns)
    }


if __name__ == "__main__":
    resultado = ejecutar_pipeline()
    print("Pipeline ejecutado correctamente.")
    print("X_train:", resultado["X_train"].shape)
    print("X_test: ", resultado["X_test"].shape)
    print("y_train:", resultado["y_train"].shape)
    print("y_test: ", resultado["y_test"].shape)
    print("Nulos en X_train:", resultado["X_train"].isna().sum().sum())
    print("Nulos en X_test: ", resultado["X_test"].isna().sum().sum())
