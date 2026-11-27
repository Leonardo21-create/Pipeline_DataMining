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

---

## Aplicacion completa: Modelado, Visualizacion e Interfaz PyQt5 (CRISP-DM)

Sobre el pipeline base se construyo la aplicacion de escritorio en **PyQt5** siguiendo las 5 fases de **CRISP-DM**, con arquitectura modular y ejecucion en hilos secundarios (`QThread`) para evitar bloqueos de la interfaz.

### Modulos agregados

```text
Proyecto_Final_Pipeline/
|-- main.py                      # Punto de entrada de la aplicacion PyQt5
|-- ui/
|   `-- main_window.ui           # Interfaz grafica en XML (Qt Designer, cargada con uic.loadUi)
`-- src/
    |-- preprocessing.py         # Ampliado con retornar_resumen=True y ejecutar_demo_joins()
    |-- modeling.py              # Regresion Logistica, K-Means, Hold-out, Stratified K-Fold y metricas
    |-- plotting.py              # Graficas puras con matplotlib.figure.Figure + Seaborn (sin pyplot)
    |-- workers.py               # TaskWorker y gestion de hilos QThread con limpieza de memoria
    `-- main_window.py           # Controlador de pestanas, tablas, estado y lienzos embebidos
```

### Requisitos y ejecucion de la interfaz

> **Nota de compatibilidad en Windows:** Usar **Python 3.12** (o 3.11) para el entorno virtual. En versiones mas recientes (Python 3.13+), `PyQt5` puede presentar errores de carga de DLL (`ImportError: DLL load failed while importing QtWidgets`).

```powershell
# Crear entorno con Python 3.12 (si hay varias versiones instaladas)
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Ejecutar la aplicacion grafica
.\.venv\Scripts\python.exe main.py
```

Pruebas independientes de los modulos de backend (*smoke tests*):

```powershell
.\.venv\Scripts\python.exe src\preprocessing.py
.\.venv\Scripts\python.exe src\modeling.py
.\.venv\Scripts\python.exe src\plotting.py
```

### Flujo por pestanas (CRISP-DM)

Las pestanas se desbloquean de forma secuencial para garantizar consistencia entre fases:

1. **1. Entendimiento del Negocio:** Contexto del problema de deteccion de intrusiones, objetivo de negocio, objetivo de mineria de datos y criterios de exito.
2. **2. Datos e Integracion:** Carga de CSV, validacion de esquema, vista previa (`head(50)`), grafica de balance de clases (`Normal` vs. `Ataque`) y demostracion interactiva de `inner`, `left`, `right` y `outer` join sobre `attack_cat`.
3. **3. Preprocesamiento:** Seleccion de escalador (`standard` o `minmax`) y proporcion de prueba (`test_size`), resumen de limpieza y ahorro de memoria, tabla IQR/p99 y boxplots comparativos antes y despues del tope p99 (`dur` y `sbytes`).
4. **4. Modelado:** Seleccion de algoritmo (**Regresion Logistica** o **K-Means**) y tecnica de validacion (**Hold-out** o **Stratified K-Fold**), con ajuste dinamico de hiperparametros (`C`, `max_iter`, `n_clusters`, `K`) y bitacora de ejecucion.
5. **5. Evaluacion:** Historial comparativo de corridas, desglose de las 6 metricas (**Exactitud, Precision, Sensibilidad, Especificidad, F1-Score y AUC-ROC**, mas `Media ± Std` en K-Fold), Matriz de Confusion dinamica y Curva ROC comparativa.

### Decisiones tecnicas clave

- **Cero *data leakage* en Stratified K-Fold:** La validacion cruzada recibe `X_encoded` sin escalar y encapsula `(escalador, modelo)` dentro de un `sklearn.pipeline.Pipeline` independiente por *fold*.
- **Evaluacion limpia de K-Means (`KMeansBinaryClassifier`):** El mapeo de cada cluster a su clase mayoritaria se aprende exclusivamente en `fit(X_train, y_train)`. Al ser no supervisado y no generar probabilidades calibradas, su `AUC-ROC` se reporta explicitamente como `N/A`.
- **Visualizacion segura en PyQt5:** `src/plotting.py` usa exclusivamente `matplotlib.figure.Figure` orientado a objetos (sin `matplotlib.pyplot`) y destruye lienzos previos con `deleteLater()` para evitar fugas de RAM.
- **Eficiencia de memoria:** Las columnas One-Hot usan `uint8` y las matrices escaladas conservan `float32`.

### Resultados de referencia (UNSW-NB15, `StandardScaler`)

| Modelo | Validacion | Exactitud | Precision | Sensibilidad | Especificidad | F1-Score | AUC-ROC | Tiempo |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Regresion Logistica | Hold-out (80/20) | 0.8579 | 0.7991 | 0.9349 | 0.7888 | 0.8617 | 0.9536 | ~1.08 s |
| Regresion Logistica | Stratified K-Fold (k=3) | 0.8580 | 0.7999 | 0.9336 | 0.7901 | 0.8616 | 0.9543 | ~2.72 s |
| K-Means (No Supervisado) | Hold-out (80/20) | 0.5445 | 0.5304 | 0.3283 | 0.7387 | 0.4056 | N/A | ~0.42 s |
| K-Means (No Supervisado) | Stratified K-Fold (k=3) | 0.5399 | 0.5226 | 0.3232 | 0.7347 | 0.3994 | N/A | ~1.14 s |
