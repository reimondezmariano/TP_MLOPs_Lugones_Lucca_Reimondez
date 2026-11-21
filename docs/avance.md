# Avance para la entrega parcial

Entrega parcial del 22/11/2026. Este documento resume lo hecho hasta ahora, lo que falta y cómo seguimos hasta la entrega final del 13/12/2026.

## Resumen

Elegimos el nivel en contenedores. Para la parcial nos concentramos en el pipeline de entrenamiento, que según lo visto en la clase 4 es más o menos la mitad del proyecto: infraestructura, ETL en Airflow y experimento de hiperparámetros en MLflow. Las tres partes están terminadas y mergeadas en `main`, y corren juntas sobre el stack del grupo: el DAG de ETL deja los datos en MinIO, la búsqueda de hiperparámetros los toma de ahí y registra el modelo `champion` en MLflow.

| Componente | Responsable | Estado |
|---|---|---|
| Infraestructura y configuración de Airflow | Mariano | Terminado |
| README (primera versión) | Mariano | Terminado |
| DAG de ETL `etl_diabetes` y paquete `readmission` | Florencia | Terminado |
| Búsqueda de hiperparámetros en MLflow y modelo `champion` | Carlos | Terminado |
| Regla basal en el paquete | Florencia | Terminado (adelantado de la final) |
| API REST | Mariano | Para la final |
| Reentrenamiento champion/challenger | Carlos | Para la final |
| Predicción por lotes con Metaflow y Valkey | Carlos | Para la final, con la infraestructura ya preparada |
| Frontend en Streamlit | Florencia | Para la final |

## Lo hecho

### Mariano: infraestructura, configuración y documentación

- Repositorio del grupo creado a partir de [amq2-service-ml](https://github.com/facundolucianna/amq2-service-ml), conservando la licencia y citando el origen.
- `docker-compose.yaml` sin credenciales escritas: todas se leen del `.env`, y si falta alguna el compose corta con un mensaje que indica cuál. Los DAGs de ejemplo de Airflow quedan desactivados.
- Variables y conexiones de Airflow en `airflow/secrets/` (`variables.yaml` y `connections.yaml`), leídas con `LocalFilesystemBackend`. Los DAGs no tienen rutas, parámetros ni credenciales escritos a mano.
- Versiones de scikit-learn (1.9.1) y MLflow (3.16.1) fijadas en las imágenes de Airflow y FastAPI, iguales a las de `requirements-experimentos.txt`, para que el modelo se cargue con las mismas versiones con que se guardó.
- DAG `check_virtualenv` para verificar la instalación: lee las variables, crea entornos virtuales en el worker de Celery y escribe y lee un Parquet en MinIO pasando solo la ruta por XCom.
- Valkey para la predicción por lotes: en lugar de sumar un contenedor, se reutiliza el que ya usa Celery, con una base aparte (Celery usa la 0 y las predicciones la 1). Se publica su puerto y se crea el bucket `batch` en MinIO.
- Primera versión del README: problema, diagrama de bloques, guía de instalación y configuración.
- Revisión del pull request de la búsqueda de hiperparámetros.

Problemas encontrados y cómo se resolvieron:

- Las tareas con `@task.virtualenv` fallaban al crear el entorno, porque `uv` intenta usar una carpeta de caché que el usuario de Airflow no puede escribir dentro de la imagen. Se resolvió definiendo `UV_CACHE_DIR` en el compose.
- En Linux y WSL, si `AIRFLOW_UID` no coincide con el usuario local, Airflow deja las carpetas de `airflow/` a nombre de otro usuario y después no se pueden editar. El README indica poner el valor de `id -u` en el `.env`.
- El `.env` del repositorio está marcado como *skip-worktree*, así que git no registra los cambios locales. Sirve para que cada uno ponga su `AIRFLOW_UID` sin subirlo, pero cuando hay que cambiar el `.env` del repo primero hay que desmarcarlo.

### Florencia: datos y variables

- DAG `etl_diabetes` en Airflow: descarga el dataset de UCI, lo limpia con las reglas de la notebook y separa train y test agrupando por paciente (71.583 y 30.180 encuentros), todo guardado en MinIO.
- Al final del DAG deja la descripción de los datos en `data.json` y registra el linaje del dataset en MLflow.
- Script en `tests/` que verifica que train y test coinciden con la descripción de `data.json`.
- Paquete `readmission` con la configuración y el preprocesador de las 18 variables, que usan la búsqueda de hiperparámetros y el reentrenamiento.
- Regla basal de internaciones previas en el paquete, adelantada de la final: Carlos la probó con la comparación de modelos y da lo mismo que la notebook (0,606 en validación cruzada y 0,603 en test).

### Carlos: modelo y ciclo de vida

#### Qué está hecho

| Entregable | Qué es | Estado |
|---|---|---|
| Notebook final `TP_FINAL_OAA.ipynb` | El TP de Aprendizaje de Máquina convertido en una sola notebook, con los resultados de la carrera de hiperparámetros guardados | En `notebooks/TP_FINAL_OAA/`, mergeado |
| `experiments/hyperparam_search.py` | Búsqueda de hiperparámetros del Random Forest con Optuna y MLflow; deja registrado el modelo `champion` | Mergeado |
| `configs/model.yaml` | Hiperparámetros del Random Forest y umbral de decisión | Mergeado |
| `docs/experimento_hiperparametros.md` | Guía del experimento, con diagrama de bloques y cómo correrlo | Mergeado |

#### El modelo de partida

El modelo es el del TP de Aprendizaje de Máquina: predice, con la información disponible al alta, si un paciente diabético reingresa al hospital dentro de los 30 días (dataset *Diabetes 130-US Hospitals*, 101.763 encuentros). Se mantienen la regla basal de una variable, la regresión logística y el Random Forest. Se sacaron XGBoost y CatBoost para que el modelo sea liviano y el entrenamiento corto: la versión sin esos dos algoritmos pasó de 25 min 52 s a 8 min 23 s en la misma máquina, con los mismos resultados en el test.

- **Split:** `GroupShuffleSplit` agrupado por `patient_nbr` (30 % de test, semilla 42), después de quitar los 3 encuentros con género «Unknown/Invalid»: 71.583 encuentros de train y 30.180 de test. Un mismo paciente nunca queda en los dos conjuntos. El DAG de ETL reproduce exactamente estas cifras.
- **Variables:** las 18 variables finales del TP. `patient_nbr` aparece en dos roles: como identificador sirve para agrupar (el split, los folds de la validación cruzada y los folds internos del target encoding), y como base de `patient_nbr_ordinal`, la cantidad de encuentros previos del paciente, que es una de las 18 variables y sí es una entrada real.
- **Notebook:** corre de arriba abajo. La carrera de hiperparámetros (grilla, búsqueda aleatoria y Optuna, con 30 evaluaciones por mecanismo y un Random Forest de 300 árboles) tarda unos 40 minutos; por eso se guardan sus resultados en `artefactos_ml/` y la notebook los reutiliza si los encuentra: con ellos corre en unos 4 minutos. Si alguien cambia la cantidad de evaluaciones o de árboles, hay que borrar esos tres archivos `.pkl`.

#### Búsqueda de hiperparámetros

- Optuna (TPE, semilla 42) con el espacio de búsqueda del TP, 30 intentos, Random Forest de 300 árboles y validación cruzada de 5 folds agrupados por paciente. El TP de Aprendizaje de Máquina usó 60 intentos y 500 árboles; se redujeron para acortar el tiempo. Con 300 árboles el ROC-AUC baja hasta 0,0003, mucho menos que el desvío entre folds (entre 0,004 y 0,006), y la búsqueda tarda unos 4 minutos en lugar de unos 22.
- **Configurable:** `--intentos` (por defecto 30) y `--submuestreo` (fracción de los pacientes del train, por defecto 1) permiten una corrida corta en una máquina con pocos núcleos. Con los valores por defecto el resultado es el de siempre. Con submuestreo, el modelo registrado se entrena con menos datos y sus métricas no son comparables con las de la notebook.
- Un run padre y un run anidado por intento en MLflow (`Trial: N`), con sus parámetros, ROC-AUC medio, desvío y tiempo.
- Con la mejor combinación reentrena con todo el train, mide en el test (solo informa) y registra el modelo como `reingreso_diabetes_rf_dev`. Lo promueve a `reingreso_diabetes_rf_prod` con el alias `champion`, el umbral como etiqueta de la versión y la curva de calibración como artefacto.
- Lee los datos desde MinIO (`s3://data/processed/`) o desde archivos locales.
- **Resultado:** ROC-AUC de 0,6806 en validación cruzada y de 0,6774 en el test, prácticamente igual al del ganador de la carrera de la notebook (0,6799 y 0,6766). Los hiperparámetros que encuentra (`max_features` 0,395, `min_samples_leaf` 38, `max_depth` 14) no coinciden con los de la notebook (0,3, 30 y 16): el óptimo es una meseta, donde combinaciones distintas dan el mismo puntaje. Los valores de la notebook quedan en `configs/model.yaml`.

#### Cómo se verificó

- Con el stack del grupo, el DAG `etl_diabetes` y el paquete `readmission`: el DAG terminó en verde y dejó en `s3://data/processed/` los Parquet con las cifras de la notebook; la búsqueda corrió completa en unos 4 minutos y dejó `reingreso_diabetes_rf_prod` con el alias `champion`, con los mismos números que la notebook.
- La búsqueda convive con el run de linaje del ETL en el mismo experimento `reingreso_diabetes`.
- Las opciones `--intentos` y `--submuestreo` se probaron con valores inválidos (se rechazan con un mensaje), con una corrida corta (3 intentos y el 30 % de los pacientes: 18 segundos) y con la corrida completa.
- También se probó con SQLite local y archivos locales, con los mismos resultados.

Para reproducirlo, con el entorno de `requirements-experimentos.txt` y el stack levantado:

```bash
export AWS_ACCESS_KEY_ID=minio AWS_SECRET_ACCESS_KEY=minio123 MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export MLFLOW_TRACKING_URI=http://localhost:5001

python experiments/hyperparam_search.py --train s3://data/processed/train.parquet                                  # unos 4 minutos
python experiments/hyperparam_search.py --train s3://data/processed/train.parquet --intentos 10 --submuestreo 0.5   # corrida corta
```

Detalles en [docs/experimento_hiperparametros.md](experimento_hiperparametros.md).

## Decisiones tomadas

- **Modelo liviano.** Se conservan solo la regla basal, la regresión logística y el Random Forest, este último con 300 árboles. La regresión logística entra como challenger en el reentrenamiento, para que se comparen dos modelos distintos y no el mismo reentrenado.
- **Preprocesador con clases propias.** El preprocesador del paquete `readmission` reproduce el de la notebook, con clases propias (`Winsorizador`, `QuitarColumna` y `CodificadorObjetivoAgrupado`, el target encoding fuera de fold agrupado por paciente). Por eso el modelo se guarda con `cloudpickle`, como en el ejemplo de la cátedra, y cargarlo exige tener instalado el paquete `readmission`, con la misma versión de Python y de las librerías con que se guardó.
- **Mismas columnas en entrenamiento y servicio.** El ETL guarda `data.json` con columnas, tipos, categorías válidas y orden, y la API arma la entrada a partir de ese archivo.
- **Umbral por versión del modelo.** Cada versión registrada en MLflow guarda su umbral como etiqueta (`umbral`): 0,14 para el Random Forest y 0,13 para la regresión logística. La etiqueta guarda el valor como texto decimal y puede figurar como `0.13999999999999999`. No hay un umbral fijo en la configuración que pueda quedar desactualizado tras un reentrenamiento.
- **Predicción por lotes con Metaflow** en lugar de un DAG de Airflow, siguiendo lo que pondera la cátedra.

## Cómo la API consume el modelo

Esto define la parte de la API para la final:

- Carga `models:/reingreso_diabetes_rf_prod@champion` desde MLflow.
- Lee el umbral de la etiqueta `umbral` de esa versión y lo convierte con `float()`.
- Recibe las 18 variables, validadas según `data.json`. El modelo espera además `patient_nbr` como última columna: solo sirve para agrupar durante el entrenamiento, al predecir no cambia el resultado, y la API la completa con un valor de relleno (0) sin pedírsela al usuario.
- La imagen de FastAPI tiene que incluir el paquete `readmission` y usar la misma versión de Python con que se entrenó el modelo. Hoy los scripts usan Python 3.13 y la imagen 3.12, así que hay que alinearlas.

## Lo que falta y cómo seguimos

| Semana | Qué hacemos |
|---|---|
| 23/11 al 29/11 | API después de la clase 6: carga del champion, validación de las 18 variables con Pydantic, `/docs` descriptivo y la imagen con el paquete `readmission`. Predicción por lotes con Metaflow sobre el MinIO y el Valkey del grupo |
| 30/11 al 6/12 | Script y DAG `retrain_model` champion/challenger, comparación de modelos (`compare_models.py`), frontend en Streamlit, tests de la API, aserciones de validación dentro del DAG `etl_diabetes` y raw con fecha de la corrida para no pisar las anteriores |
| 7/12 al 13/12 | Sin desarrollo nuevo: secciones del README sobre los datos y el modelo, prueba en un clon limpio siguiendo el README, ajustes y consulta del 9/12 |

## Sistemas en los que se probó

- Ubuntu en WSL2 (Windows), Docker Desktop.
- macOS, Docker Desktop: el ETL, el paquete `readmission` y la infraestructura.
- Python 3.13 con las versiones de la cátedra (MLflow 3.16.1, scikit-learn 1.9.1, Optuna 5.0.0, awswrangler 3.17.1) para los scripts de `experiments/`.
