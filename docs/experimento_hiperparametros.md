# Experimento de búsqueda de hiperparámetros

Script: `experiments/hyperparam_search.py`. Pasa a código la búsqueda de hiperparámetros del Random Forest
del notebook 03 (sección M5) y deja el resultado en MLflow: un experimento con un run por intento y el mejor
modelo registrado con el alias `champion`.

Sigue el esquema del experimento de la cátedra (`notebook_example/experiment_mlflow.ipynb` de la rama
`example_implementation`): datos desde MinIO con `awswrangler`, Optuna con un run padre y un run anidado por
intento, modelo guardado con `cloudpickle` y firma, y un registro en dos pasos (`_dev` y `_prod`).

La configuración por defecto es 30 intentos, 300 árboles, todo el train y 5 folds agrupados por paciente. Es la de la
notebook `notebooks/TP_FINAL_OAA/TP_FINAL_OAA.ipynb`, que parte de la del TP de Aprendizaje de Máquina (60 intentos y 500
árboles) con menos intentos y árboles para acortar el tiempo. Para acortarlo todavía más, la cantidad de intentos y la
fracción del train que se usa se pueden cambiar desde la línea de comandos (`--intentos` y `--submuestreo`); los árboles y
los folds quedan fijos.

## Qué problema resuelve

El servicio sirve un solo modelo, el Random Forest. Sus hiperparámetros no se fijan a mano: se buscan con
validación cruzada y se deja registro de cada intento, para que cualquiera pueda ver qué se probó, qué
resultado dio y por qué el modelo registrado es ese.

## Flujo

```
s3://data/processed/train.parquet
                        │
                        ▼
        5 folds agrupados por paciente (semilla 42)
        preprocesador ajustado solo con el train de cada fold
                        │
                        ▼
     Optuna (TPE, semilla 42) propone una combinación ◄──┐
                        │                                │
                        ▼                                │
        Random Forest de 300 árboles en los 5 folds      │
        → ROC-AUC medio                                  │
        run anidado "Trial: N" en MLflow ────────────────┘
                        │  (30 intentos)
                        ▼
              mejor combinación
                        │
        reentrenamiento con todo el train
                        │
   ┌────────────────────┼─────────────────────┐
   ▼                    ▼                     ▼
métricas fuera     curva de calibración   métricas en el test (solo se informan)
de fold (umbral)   (artefacto PNG)
                        │
                        ▼
     Model Registry: <nombre>_dev  →  <nombre>_prod con alias `champion`
```

## Cómo correrlo

Con el entorno de `requirements-experimentos.txt` (versiones de la cátedra, Python 3.13) y el stack levantado
(`docker compose --profile all up`), con las variables del `.env` exportadas en la terminal:

```bash
export AWS_ACCESS_KEY_ID=minio
export AWS_SECRET_ACCESS_KEY=minio123
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000   # desde tu máquina; dentro de Docker es http://s3:9000

python experiments/hyperparam_search.py --train s3://data/processed/train.parquet
```

Sin Docker se puede probar con archivos locales y una base SQLite:

```bash
python experiments/hyperparam_search.py --tracking-uri sqlite:///mlflow_local.db
```

| Opción | Por defecto | Para qué sirve |
|---|---|---|
| `--train` | train de `configs/base.yaml` | Parquet de train, ruta local o `s3://`. |
| `--test` | `test.parquet` junto al train | Parquet de test. |
| `--tracking-uri` | `MLFLOW_TRACKING_URI` o `http://localhost:5001` | Servidor de MLflow. |
| `--intentos` | 30 | Cantidad de evaluaciones de la búsqueda con Optuna. |
| `--submuestreo` | 1 | Fracción de los pacientes del train que se usa, entre 0 y 1. Con 1 se usa todo el train. |
| `--experimento` | `reingreso_diabetes` | Nombre del experimento. |
| `--nombre-modelo` | `reingreso_diabetes_rf` | Prefijo en el registro: se crean `<nombre>_dev` y `<nombre>_prod`. |

**Corrida corta.** Para probar el flujo completo sin esperar los 4 minutos, o en una máquina con pocos núcleos:

```bash
python experiments/hyperparam_search.py --train s3://data/processed/train.parquet --intentos 10 --submuestreo 0.5
```

El submuestreo elige pacientes al azar (con la semilla del proyecto) y se queda con todos sus encuentros, para que los folds
sigan agrupados por paciente. Se aplica a la búsqueda y al reentrenamiento final: el modelo que queda como `champion`
se entrenó con menos datos. Por eso sus métricas no son comparables con las de la notebook, y el run padre guarda
`n_intentos` y `submuestreo` para que se vea con qué se corrió. El test se mide siempre completo. Con los valores por
defecto no cambia nada. Una corrida de 3 intentos con el 30 % de los pacientes (21.488 filas) tardó 18 segundos contra una
base SQLite local.

Las credenciales y el servidor de MinIO se toman del entorno (las mismas variables que usa el stack), nunca de
argumentos ni del código. Si falta `AWS_ENDPOINT_URL_S3`, se copia de `MLFLOW_S3_ENDPOINT_URL`.

## Qué se busca

Mismo espacio que el notebook, para poder comparar resultados:

| Hiperparámetro | Rango |
|---|---|
| `max_features` | 0,1 a 0,8 |
| `min_samples_leaf` | 1 a 300 (escala logarítmica) |
| `max_depth` | 4 a 40 |

`n_estimators` no se busca: se fija en 300, como en la notebook. La métrica es el ROC-AUC medio de los folds, con folds
agrupados por paciente para que un mismo paciente no quede en entrenamiento y en validación.

## Qué se obtiene con los datos del TP

Con los 30 intentos y los 300 árboles de ahora, la búsqueda llega a un ROC-AUC medio de 0,6806 en la validación cruzada y de
0,6774 en el test, con `max_features` 0,395, `min_samples_leaf` 38 y `max_depth` 14. Es prácticamente lo mismo que el ganador de
la carrera de la notebook (0,6799 y 0,6766, con `max_features` 0,3, `min_samples_leaf` 30 y `max_depth` 16, que son los de
`configs/model.yaml`): es la meseta que describe el TP, donde combinaciones distintas dan el mismo puntaje. Con 60 intentos y
500 árboles el TP original llegaba a `max_features` 0,4150, `min_samples_leaf` 14 y `max_depth` 10, con 0,6811 y 0,6773.

**Tarda unos 4 minutos** con 14 núcleos (248 segundos medidos con 30 intentos y 300 árboles, con el stack de la cátedra
completo levantado): son unos 8 segundos por intento, porque cada uno entrena 5 bosques de 300 árboles. Con 60 intentos y 500
árboles tardaba 22 minutos (22 segundos por intento), el mismo costo que en el TP original, donde el Random Forest tardaba de
11 a 14 minutos por cada mecanismo de búsqueda. En una máquina con menos núcleos va a tardar más.

## Qué deja en MLflow

- **Run padre** `best_hyperparam_<fecha>`: parámetros de la corrida, etiquetas (`project`, `optimizer_engine`,
  `model_family`, `feature_set_version`), el mejor ROC-AUC (`best_train_auc`), recall y precisión al umbral
  0,14 fuera de fold, las métricas en el test (`test_roc_auc`, `test_f1_umbral`, …) y el artefacto
  `calibracion_oof.png`.
- **Un run anidado por intento** (`Trial: 0`, `Trial: 1`, …): los tres hiperparámetros, `auc_media`,
  `auc_desvio` y `segundos`.
- **Modelo `<nombre>_dev`**: un Pipeline con el preprocesador y el Random Forest, reentrenado con todo el train.
- **Modelo `<nombre>_prod`**: recibe una versión con los hiperparámetros y las métricas como tags (incluido el
  `umbral`) y el alias `champion`. Es el que va a servir la API.

El test se usa solo para informar: la elección de hiperparámetros termina antes de tocarlo.

Para cargarlo:

```python
import mlflow
modelo = mlflow.sklearn.load_model("models:/reingreso_diabetes_rf_prod@champion")
probabilidad = modelo.predict_proba(X)[:, 1]
```

## Cosas que hay que saber

- **Entrada del modelo:** las 18 variables más `patient_nbr` como última columna. `patient_nbr` aparece en dos roles. Como **identificador** sirve para agrupar (el split, los folds de la validación
  cruzada y los folds internos del target encoding): ahí la columna cruda se pasa como última entrada del modelo, y al predecir
  se puede completar con un valor de relleno. Como **base de una variable derivada**, `patient_nbr_ordinal` (la cantidad de
  encuentros *previos* del paciente en el registro, de 0 a «5 o mas») es una de las 18 variables y **sí es una entrada real**.
- **Formato de guardado:** el modelo se guarda con `cloudpickle`, porque el preprocesador usa clases propias
  (`Winsorizador`, `QuitarColumna`, `CodificadorObjetivoAgrupado`). Para cargarlo hace falta tener el paquete del
  proyecto instalado en quien lo carga (la API) y confiar en el servidor de MLflow.
- **Umbral:** la búsqueda optimiza el ROC-AUC, que no depende del umbral. El 0,14 viene de `configs/model.yaml`
  y se usa para las métricas de recall y precisión y como tag del modelo `_prod`.
- **Versiones:** las de la cátedra: `mlflow==3.16.1`, `awswrangler==3.17.1`, `optuna==5.0.0`,
  `scikit-learn==1.9.1`, `pyarrow<26`.
- **Este script y `compare_models.py`:** los dos pueden dejar un `champion`. La elección del modelo con la regla del
  TP la hace `compare_models.py` (ver `docs/comparacion_modelos.md`); este script busca los hiperparámetros del
  Random Forest.

## Estado

- Probado con SQLite local y archivos locales: corrida completa con 60 intentos (22 min 12 s), con sus 60 runs anidados, registro del
  champion, carga del modelo y predicción en el test.
- Probado contra el stack de la cátedra en Docker (MLflow en el puerto 5001, MinIO y PostgreSQL), con los datos leídos desde
  `s3://data/processed/`: corrida completa con 60 intentos (23 min 7 s), 60 runs anidados, gráfico de calibración y modelos
  guardados en MinIO, y `champion` cargado desde el registro. Con 60 intentos los hiperparámetros ganadores son los del TP.
- Probado con la configuración actual (30 intentos y 300 árboles) contra el stack de la cátedra en Docker, con los datos leídos
  desde `s3://data/processed/`: 4 min 8 s, 30 runs anidados y `champion` registrado, con ROC-AUC de 0,6806 en validación
  cruzada y de 0,6774 en el test.
- Opciones `--intentos` y `--submuestreo`: probadas con una base SQLite local, con valores inválidos (se rechazan con un mensaje),
  con 3 intentos y el 30 % de los pacientes, y con la corrida completa por defecto, que da el mismo resultado que antes.
- No probado en la Mac del docente.
