# Explicacion y Preguntas Clave del Pipeline

Resumen de todas las dudas resueltas durante la construccion del pipeline de preprocesamiento para el dataset **UNSW-NB15**. Sirve como guia de estudio para el equipo y para preparar la exposicion del proyecto.

---

## 1. Que son los tipos de join: `inner`, `left`, `right` y `outer`?

Son las 4 formas de unir dos tablas con `pd.merge(tabla_izq, tabla_der, on="attack_cat", how=...)` usando una columna en comun (`attack_cat`).

En nuestro ejemplo del notebook:
- **Tabla izquierda (`df_base`):** registros reales del dataset (incluye `Worms`, no tiene `ZeroDay`).
- **Tabla derecha (`tabla_severidad_demo`):** catalogo de prueba (incluye `ZeroDay`, pero omitimos `Worms` a proposito).

| Tipo de join | Que filas conserva | Con `Worms` (solo en izq.) | Con `ZeroDay` (solo en der.) |
| :--- | :--- | :--- | :--- |
| **`inner`** | Solo coincidencias en **ambas** tablas | Se elimina | Se elimina |
| **`left`** | Todas las de la **izquierda** + coincidencias de la derecha | Se conserva (`nivel_severidad = NaN`) | Se elimina |
| **`right`** | Todas las de la **derecha** + coincidencias de la izquierda | Se elimina | Se conserva (`id = NaN`, `label = NaN`) |
| **`outer`** | **Todas** las filas de ambas tablas | Se conserva (`nivel_severidad = NaN`) | Se conserva (`id = NaN`, `label = NaN`) |

En Mineria de Datos, **`left`** es el mas usado porque enriquece la tabla principal sin perder registros originales.

---

## 2. Como sabiamos que habria nulos en `nivel_severidad` y en `id`?

Porque los provocamos **a proposito** en `tabla_severidad_demo` siguiendo la regla de `pd.merge`: cuando una fila no encuentra pareja en la otra tabla, pandas rellena con `NaN` las columnas que venian de la tabla donde falto la coincidencia.

- **Nulos en `nivel_severidad` (`left` y `outer`):** `df_base` tiene filas con `Worms`, pero `tabla_severidad_demo` no lo tiene; al conservar esas filas, la columna de la derecha (`nivel_severidad`) queda en `NaN`.
- **Nulos en `id` y `label` (`right` y `outer`):** `tabla_severidad_demo` tiene `ZeroDay`, que no existe en `df_base`; al conservar esa fila, las columnas de la izquierda (`id` y `label`) quedan en `NaN`.

Si ambas tablas tuvieran exactamente las mismas 10 categorias (como en el Paso 1.6), los 4 joins darian el mismo resultado sin nulos y no se notaria la diferencia.

---

## 3. Explicacion a detalle del Paso 1.5 (`pd.merge` y conteo de nulos)

1. **`df_base = df[["id", "attack_cat", "label"]]`**: Seleccionamos solo 3 columnas para que las salidas en pantalla sean faciles de leer.
2. **`pd.merge(df_base, tabla_severidad_demo, on="attack_cat", how=...)`**: Une `df_base` (izquierda) con `tabla_severidad_demo` (derecha) comparando la columna `attack_cat`.
3. **`len(...)` e `.isna().sum()`**: `len(...)` cuenta las filas resultantes; `.isna()` marca `True` donde hay `NaN` y `.sum()` suma cuantos `True` hubo.
   - `INNER`: 257,499 filas, 0 nulos (se quitan las 174 filas de `Worms` y `ZeroDay`).
   - `LEFT`: 257,673 filas, 174 nulos en `nivel_severidad` (las filas de `Worms`).
   - `RIGHT`: 257,500 filas, 1 nulo en `id` (la fila de `ZeroDay`).
   - `OUTER`: 257,674 filas, 174 nulos en `nivel_severidad` y 1 nulo en `id`.
4. **Filtrado de ejemplos:** `join_left[join_left["nivel_severidad"].isna()].head(3)` muestra ejemplos de `Worms` sin severidad; `join_right[join_right["id"].isna()]` muestra la fila de `ZeroDay` sin `id`.

---

## 4. Por que se imputa `service` con `"unknown"` en el Paso 2.1?

1. **No podemos borrar esas filas (`dropna`):** Hay **141,321** registros con `'-'` de 257,673 (casi el **55%** del dataset). Borrarlos nos haria perder mas de la mitad de los datos.
2. **No conviene rellenar con la moda (como `"dns"`):** Que no haya servicio identificado suele ser informacion real en redes (puertos no estandar, protocolos de bajo nivel o escaneos/ataques). Inventar `"dns"` sesgaria al modelo.
3. **Por que `"unknown"` funciona bien:** Conserva todas las filas, elimina `NaN` para Scikit-learn y mantiene la senal de "servicio no identificado" como una categoria limpia y explicita.

---

## 5. Como afecta al dataset haber encontrado casi un 40% de duplicados (Paso 2.2)?

Se encontraron **94,928** duplicados exactos excluyendo `id` (~36.8% del total). Eliminarlos afecta positivamente en cuatro aspectos:

1. **Evita fuga de datos (*data leakage*) entre train y test:** Evita que una misma fila repetida caiga tanto en entrenamiento como en prueba, lo que inflaria falsamente las metricas del modelo.
2. **Reduce sobreajuste (*overfitting*) y sesgo:** Impide que el modelo memorice flujos repetidos cientos de veces en lugar de aprender patrones generales.
3. **Cambia la proporcion de clases:** Muchos duplicados pertenecen a ataques repetitivos (como `Generic` o `DoS`). Por eso al dividir en train/test usamos `stratify=y` para preservar las proporciones reales.
4. **Mejor rendimiento:** Pasar de 257,673 a 162,745 filas unicas acelera el entrenamiento y reduce memoria sin perder patrones distintos.

---

## 6. Como funciona la regla IQR (Paso 2.4) y que concluimos al ver tantos outliers?

### Regla IQR (Rango Intercuartilico)
- **`Q1` (percentil 25):** deja debajo el 25% mas bajo de los datos.
- **`Q3` (percentil 75):** deja debajo el 75% de los datos.
- **`IQR = Q3 - Q1`:** rango donde vive el 50% central de los datos (la caja del boxplot).
- **Fronteras:** limite inferior `Q1 - 1.5 * IQR` y limite superior `Q3 + 1.5 * IQR`. Todo valor fuera de esas fronteras se cuenta como *outlier*.

### Conclusion
- El trafico de red es muy **asimetrico (sesgado a la derecha)**: la mayoria de conexiones son cortas y pequenas, mientras unas cuantas mueven enormes cantidades de datos.
- Muchos valores extremos son **ataques reales** (por ejemplo, DoS o exfiltracion), no errores de captura.
- Por eso **no se deben borrar** esas filas; en su lugar aplicamos un tope suave en el percentil 99.

---

## 7. Como funciona el tope en el percentil 99 (`clip`) y por que en las graficas se siguen viendo tantos puntos?

### Como funciona `clip(upper=limite_p99)`
- El **percentil 99** es el valor por debajo del cual esta el 99% de los datos; solo el 1% mas extremo lo supera.
- Si un valor es menor o igual al percentil 99, **no cambia**.
- Si un valor es mayor al percentil 99, **se recorta a ese valor tope** sin borrar la fila.

### Por que siguen apareciendo muchos puntos en el boxplot
1. El `clip` solo recorta el **1% superior**, mientras que la regla del boxplot (`Q3 + 1.5 * IQR`) sigue marcando entre el 10% y 20% de los datos en distribuciones muy sesgadas.
2. Con **162,745 filas**, incluso un 10% son mas de 16,000 puntos dibujados en linea horizontal.
3. El efecto real se nota en la **escala del eje X** entre "Antes" y "Despues": al eliminar el 1% gigantesco, la escala se vuelve mucho mas equilibrada para entrenar modelos.

---

## 8. Que es `src/preprocessing.py` y en que se diferencia del notebook?

- **`notebooks/preprocesamiento.ipynb`:** Es el cuaderno de laboratorio para explorar paso a paso, ver graficas y documentar la rubrica.
- **`src/preprocessing.py`:** Es el modulo Python reutilizable con la misma logica exacta en 5 funciones simples (sin graficas). Existe porque un script de Scikit-learn o una interfaz en **PyQt5** no puede importar directamente un `.ipynb`, pero si puede importar `from src.preprocessing import ejecutar_pipeline`.

---

## 9. Cual es la manera mas efectiva de compartir toda la carpeta del proyecto?

- **Regla de oro:** **Nunca compartir `.venv/`** (ni `__pycache__/`), porque pesa cientos de MB y contiene rutas fijas de una sola PC.
- **Que si compartir:** `data/`, `notebooks/`, `src/`, `requirements.txt`, `README.md` y `explicacion.md`.
- **Opciones recomendadas:**
  1. **GitHub** (mejor si el equipo sigue editando codigo en paralelo; agregar `.venv/` y `__pycache__/` a `.gitignore`).
  2. **Archivo `.zip`** (mas rapido para entregar los archivos de una sola vez; los CSV comprimidos bajan a ~10-15 MB).
- Quien reciba el proyecto solo debe crear su propio `.venv` e instalar con `pip install -r requirements.txt`.
