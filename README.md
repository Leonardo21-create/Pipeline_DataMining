# Pipeline de Preprocesamiento - UNSW-NB15

Pipeline de preprocesamiento de datos para el proyecto de Mineria de Datos enfocado en **deteccion de intrusiones en red** usando el dataset **UNSW-NB15**.

El proyecto prepara los datos paso a paso en un Jupyter Notebook para validacion manual y exporta la logica exacta a un modulo Python limpio listo para integrarse con modelos de **Scikit-learn** y una interfaz grafica en **PyQt5**.

---

## Estructura del proyecto

```text
Proyecto_Final_Pipeline/
|-- data/
|   |-- UNSW_NB15_training-set.csv
|   `-- UNSW_NB15_testing-set.csv
|-- notebooks/
|   `-- preprocesamiento.ipynb   # Desarrollo y validacion paso a paso (con graficas)
|-- src/
|   `-- preprocessing.py         # Modulo reutilizable para Scikit-learn y PyQt5
|-- explicacion.md               # Resumen de conceptos y preguntas clave del pipeline
|-- requirements.txt             # Librerias necesarias del entorno
`-- README.md
```

---

## Instalacion rapida (para companeros de equipo)

Desde la carpeta raiz del proyecto en la terminal (PowerShell en Windows):

```powershell
# 1. Crear entorno virtual
python -m venv .venv

# 2. Instalar librerias
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3. Probar que el pipeline corre sin errores
.\.venv\Scripts\python.exe src\preprocessing.py
```

Salida esperada al ejecutar `src\preprocessing.py`:
- `X_train`: `(130196, 74)` con `0` nulos
- `X_test`: `(32549, 74)` con `0` nulos
- `y_train`: `(130196,)` | `y_test`: `(32549,)`

> **Nota:** Para ejecutar `notebooks/preprocesamiento.ipynb` en VS Code, abre el archivo y selecciona como kernel el interprete de `.venv`.

---

## Que hace el pipeline (resumen por fases)

1. **Fase 1 - Integracion de datos (`cargar_e_integrar_datos`):**
   - Carga `UNSW_NB15_training-set.csv` (175,341 filas) y `UNSW_NB15_testing-set.csv` (82,332 filas).
   - Los concatena en un solo DataFrame (257,673 filas) y regenera una columna `id` unica y consecutiva.
   - En el notebook tambien se incluye una tabla auxiliar de severidad para demostrar `inner`, `left`, `right` y `outer` join con `pd.merge`.

2. **Fase 2 - Limpieza de datos (`limpiar_datos`):**
   - Convierte el valor oculto `'-'` en la columna `service` a `NaN` y lo imputa como `"unknown"` (sin borrar mas del 50% de las filas).
   - Detecta y elimina 94,928 registros duplicados exactos (excluyendo `id`), dejando 162,745 filas unicas.
   - Estandariza texto en columnas categoricas (`proto`, `service`, `state` en minusculas y sin espacios extra).
   - Aplica un tope (*clipping*) en el **percentil 99** a `dur`, `sbytes`, `dbytes`, `rate`, `sload` y `dload` para controlar valores extremos sin eliminar ataques reales.

3. **Fase 3 - Transformacion de datos (`transformar_datos` y `dividir_y_escalar`):**
   - Optimiza tipos numericos (`downcast` de enteros, `float32` y `category`) para ahorrar memoria.
   - Crea 3 nuevas variables de trafico de red:
     - `total_pkts = spkts + dpkts`
     - `total_bytes = sbytes + dbytes`
     - `bytes_ratio = sbytes / (total_bytes + 1)`
   - Separa `X` (excluyendo `id`, `label` y `attack_cat` para evitar *target leakage*) y `y` (`label` por defecto).
   - Agrupa los protocolos poco frecuentes de `proto` fuera del Top 10 bajo `"other"` y aplica **One-Hot Encoding** a `proto`, `service` y `state`.
   - Divide primero en `train` (80%) y `test` (20%) con `stratify=y` y ajusta el escalador (`StandardScaler` o `MinMaxScaler`) **unicamente sobre `X_train`** para evitar *data leakage*.

---

## Como usar `src/preprocessing.py` en Scikit-learn y PyQt5

### Uso rapido (todo el pipeline en una llamada)

```python
from src.preprocessing import ejecutar_pipeline

datos = ejecutar_pipeline(
    ruta_train="data/UNSW_NB15_training-set.csv",
    ruta_test="data/UNSW_NB15_testing-set.csv",
    columna_objetivo="label",     # "label" (binario) o "attack_cat" (multiclase)
    metodo_escalado="standard",   # "standard" (StandardScaler) o "minmax" (MinMaxScaler)
    test_size=0.2,
    random_state=42
)

X_train = datos["X_train"]
X_test = datos["X_test"]
y_train = datos["y_train"]
y_test = datos["y_test"]
scaler = datos["scaler"]
columnas = datos["columnas"]
```

### Uso paso a paso (si necesitas llamar funciones individuales)

```python
from src.preprocessing import (
    cargar_e_integrar_datos,
    limpiar_datos,
    transformar_datos,
    dividir_y_escalar
)

df = cargar_e_integrar_datos(
    "data/UNSW_NB15_training-set.csv",
    "data/UNSW_NB15_testing-set.csv"
)
df = limpiar_datos(df)
X, y = transformar_datos(df, columna_objetivo="label")
X_train, X_test, y_train, y_test, scaler = dividir_y_escalar(X, y, metodo="standard")
```

---

## Documentacion adicional

Para consultar explicaciones detalladas sobre joins, imputacion de `service`, impacto de duplicados, regla IQR, percentil 99 y como compartir el proyecto, revisa [explicacion.md](explicacion.md).
