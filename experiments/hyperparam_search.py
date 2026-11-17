"""Búsqueda de hiperparámetros del Random Forest con MLflow (notebook 03, sección M5, pasada a script).

Sigue el esquema del experimento de la cátedra (rama example_implementation): datos desde MinIO con
awswrangler, Optuna con un run padre y un run anidado por intento, modelo registrado en MLflow con
cloudpickle y firma, y un modelo `_prod` cuyo alias `champion` es el que sirve la API.

Qué hace:
    1. Lee el train ya procesado: de MinIO (s3://data/processed/train.parquet, lo deja el DAG de ETL) o de un
       archivo local.
    2. Arma 5 folds agrupados por paciente (StratifiedGroupKFold, semilla 42) y ajusta el preprocesador
       solo con el train de cada fold, como el notebook.
    3. Prueba combinaciones con Optuna (TPE). Cada intento es un run anidado en MLflow con sus parámetros,
       el ROC-AUC medio de los folds y el desvío.
    4. Con la mejor combinación reentrena sobre todo el train y registra el modelo como `<nombre>_dev`.
    5. Mide el modelo en el test (solo se informa; no participa en la elección) y lo promueve a `<nombre>_prod`
       con el alias `champion`, el umbral de decisión como tag y la curva de calibración como artefacto.

La configuración no se cambia desde la línea de comandos: espacio de búsqueda de M5 de la notebook, 300 árboles, todo el
train y 5 folds agrupados por paciente, como en el TP. Las evaluaciones con Optuna (TPE, semilla 42) son 30 y no las 60 del
TP, para acortar el tiempo (D10): el ROC-AUC queda igual, pero los hiperparámetros ganadores pueden diferir.

Variables de entorno (las mismas que usa el stack de la cátedra; van en el .env, nunca en el código):
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY   credenciales de MinIO
    MLFLOW_S3_ENDPOINT_URL                     por ejemplo http://localhost:9000 desde tu máquina
    AWS_ENDPOINT_URL_S3                        si falta, se copia de MLFLOW_S3_ENDPOINT_URL
    MLFLOW_TRACKING_URI                        si falta y no se pasa --tracking-uri, http://localhost:5001

Ejemplos:
    python experiments/hyperparam_search.py --train s3://data/processed/train.parquet

    # sin Docker: base SQLite local y archivos locales
    python experiments/hyperparam_search.py --tracking-uri sqlite:///mlflow_local.db
"""
import argparse
import datetime
import logging
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana: tiene que correr dentro de un contenedor
import awswrangler as wr
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

# El paquete del repo del grupo es `readmission`; `readmision` es el nombre de la versión previa del paquete.
try:
    from readmission import config  # noqa: E402
    from readmission.preprocess import columnas_entrada, construir_preprocesador  # noqa: E402
except ImportError:
    from readmision import config  # noqa: E402
    from readmision.preprocess import columnas_entrada, construir_preprocesador  # noqa: E402

log = logging.getLogger("hyperparam_search")

# Servidor de MLflow del stack de la cátedra, visto desde la máquina del usuario.
TRACKING_POR_DEFECTO = "http://localhost:5001"

# Espacio de búsqueda del notebook 03 (M5) para Random Forest.
# n_estimators no se busca: es una perilla de costo (--n-arboles), igual que en la notebook (300 fijo).
# Configuración del TP (notebook 03, M5): no se cambia desde la línea de comandos.
N_INTENTOS = 30  # evaluaciones de la búsqueda (el TP usó 60; ver D10)
N_FOLDS = 5      # folds de la validación cruzada

ESPACIO = {
    "max_features": ("float", 0.1, 0.8),
    "min_samples_leaf": ("logint", 1, 300),
    "max_depth": ("int", 4, 40),
}


def sugerir(trial: optuna.Trial) -> dict:
    """Pide a Optuna una combinación del espacio de búsqueda.

    Args:
        trial: intento en curso del estudio de Optuna.

    Returns:
        Diccionario con `max_features`, `min_samples_leaf` y `max_depth`.
    """
    p = {}
    for nombre, (tipo, bajo, alto) in ESPACIO.items():
        if tipo == "float":
            p[nombre] = trial.suggest_float(nombre, bajo, alto)
        elif tipo == "int":
            p[nombre] = trial.suggest_int(nombre, bajo, alto)
        else:
            p[nombre] = trial.suggest_int(nombre, bajo, alto, log=True)
    return p


def leer_parquet(ruta: str) -> pd.DataFrame:
    """Lee un Parquet desde un archivo local o desde S3 (MinIO) con awswrangler.

    Las credenciales no se pasan por argumento: awswrangler las toma de AWS_ACCESS_KEY_ID y
    AWS_SECRET_ACCESS_KEY, para que no queden escritas en el código ni en el historial de comandos. El servidor
    se toma de AWS_ENDPOINT_URL_S3; si falta se copia de MLFLOW_S3_ENDPOINT_URL, que es la que define el stack.

    Args:
        ruta: ruta local o dirección `s3://bucket/clave.parquet`.

    Returns:
        La tabla leída.
    """
    if not ruta.startswith("s3://"):
        return pd.read_parquet(ruta)
    if "MLFLOW_S3_ENDPOINT_URL" in os.environ:
        os.environ.setdefault("AWS_ENDPOINT_URL_S3", os.environ["MLFLOW_S3_ENDPOINT_URL"])
    return wr.s3.read_parquet(ruta)


def obtener_o_crear_experimento(nombre: str) -> str:
    """Devuelve el id del experimento de MLflow, creándolo si no existe.

    Args:
        nombre: nombre del experimento.

    Returns:
        Id del experimento.
    """
    if experimento := mlflow.get_experiment_by_name(nombre):
        return experimento.experiment_id
    return mlflow.create_experiment(nombre)


def callback_campeon(estudio: optuna.Study, intento: optuna.trial.FrozenTrial) -> None:
    """Avisa por log cada vez que un intento mejora al mejor valor visto hasta el momento.

    Args:
        estudio: estudio de Optuna en curso.
        intento: intento que acaba de terminar.
    """
    anterior = estudio.user_attrs.get("mejor_anterior")
    if estudio.best_value != anterior:
        estudio.set_user_attr("mejor_anterior", estudio.best_value)
        if anterior:
            mejora = abs(anterior - estudio.best_value) / estudio.best_value * 100
            log.info("intento %d: nuevo mejor ROC-AUC %.4f (mejora de %.4f %%)", intento.number + 1,
                     intento.value, mejora)


def preparar_folds(train: pd.DataFrame, objetivo: str, feat: dict, semilla: int, n_folds: int) -> list:
    """Devuelve, por fold, las matrices ya preprocesadas. El preprocesador se ajusta solo con el train del fold.

    Se calcula una vez y se reutiliza en todos los intentos: el preprocesamiento no depende de los
    hiperparámetros del modelo, y repetirlo en cada intento sería el grueso del tiempo.

    Args:
        train: tabla de train con las columnas de entrada y el objetivo.
        objetivo: nombre de la columna objetivo (reingreso a 30 días).
        feat: contenido de configs/features.yaml.
        semilla: semilla de los folds y del preprocesador.
        n_folds: cantidad de folds.

    Returns:
        Lista con un diccionario por fold: `Ztr`, `Zva`, `ytr`, `yva` e `idx_va` (posiciones de validación).
    """
    X, y = train[columnas_entrada(feat)], train[objetivo].to_numpy()
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=semilla)
    folds = []
    for k, (idx_tr, idx_va) in enumerate(cv.split(X, y, groups=train["patient_nbr"])):
        prep = construir_preprocesador(feat, semilla)
        Ztr = prep.fit_transform(X.iloc[idx_tr], y[idx_tr])
        Zva = prep.transform(X.iloc[idx_va])
        folds.append({"Ztr": Ztr, "Zva": Zva, "ytr": y[idx_tr], "yva": y[idx_va], "idx_va": idx_va})
        log.info("fold %d/%d listo (%d train, %d validación)", k + 1, n_folds, len(idx_tr), len(idx_va))
    return folds


def evaluar(params: dict, folds: list, n_arboles: int, semilla: int) -> tuple[np.ndarray, np.ndarray]:
    """Evalúa una combinación de hiperparámetros con validación cruzada.

    Args:
        params: `max_features`, `min_samples_leaf` y `max_depth`.
        folds: salida de `preparar_folds`.
        n_arboles: cantidad de árboles del Random Forest.
        semilla: semilla del modelo.

    Returns:
        ROC-AUC de cada fold y probabilidades fuera de fold, en el orden de las filas del train.
    """
    n = sum(len(f["yva"]) for f in folds)
    oof, aucs = np.zeros(n), []
    for f in folds:
        modelo = RandomForestClassifier(n_estimators=n_arboles, n_jobs=-1, random_state=semilla, **params)
        modelo.fit(f["Ztr"], f["ytr"])
        pv = modelo.predict_proba(f["Zva"])[:, 1]
        oof[f["idx_va"]] = pv
        aucs.append(roc_auc_score(f["yva"], pv))
    return np.array(aucs), oof


def curva_calibracion(y: np.ndarray, probas: np.ndarray, ruta: Path) -> None:
    """Guarda la curva de calibración fuera de fold: probabilidad predicha vs. frecuencia observada.

    Args:
        y: reingreso real (0 o 1) de cada fila del train.
        probas: probabilidad fuera de fold de cada fila.
        ruta: archivo PNG de salida.
    """
    freq, media = calibration_curve(y, probas, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    tope = max(media.max(), freq.max())
    ax.plot([0, tope], [0, tope], "--", color="gray", label="Calibración perfecta")
    ax.plot(media, freq, "o-", label="Random Forest (fuera de fold)")
    ax.set_xlabel("Probabilidad predicha")
    ax.set_ylabel("Frecuencia observada de reingreso")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def promover_a_prod(cliente: MlflowClient, nombre_prod: str, modelo_uri: str, run_id: str, tags: dict) -> str:
    """Registra una versión en el modelo `_prod` y le pone el alias `champion`, como hace la cátedra.

    Args:
        cliente: cliente de MLflow.
        nombre_prod: nombre del modelo productivo en el registro; se crea si no existe.
        modelo_uri: dirección `models:/m-...` del modelo ya registrado como `_dev`.
        run_id: run que produjo el modelo.
        tags: etiquetas de la versión (hiperparámetros y métricas), como texto.

    Returns:
        Número de la versión creada.
    """
    try:
        cliente.create_registered_model(nombre_prod, description="Predice reingreso hospitalario a 30 días")
    except MlflowException:
        pass  # ya existía: solo se agrega una versión nueva
    version = cliente.create_model_version(name=nombre_prod, source=modelo_uri, run_id=run_id, tags=tags)
    cliente.set_registered_model_alias(nombre_prod, "champion", version.version)
    return version.version


def main(argv=None) -> None:
    """Busca los hiperparámetros del Random Forest con Optuna, registra cada intento y deja el mejor modelo como champion.

    Args:
        argv: argumentos de la línea de comandos; si es None se toman de `sys.argv`.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default=None,
                    help="Parquet de train, ruta local o s3://...; por defecto el de configs/base.yaml")
    ap.add_argument("--test", default=None,
                    help="Parquet de test; por defecto el mismo directorio que --train con test.parquet")
    ap.add_argument("--tracking-uri", default=None,
                    help=f"servidor de MLflow (o variable MLFLOW_TRACKING_URI; si no, {TRACKING_POR_DEFECTO})")
    ap.add_argument("--experimento", default="reingreso_diabetes")
    ap.add_argument("--nombre-modelo", default="reingreso_diabetes_rf",
                    help="prefijo del nombre en el Model Registry; se crean <nombre>_dev y <nombre>_prod")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    warnings.filterwarnings("ignore")
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    base, feat, modelo_cfg = config.cargar("base"), config.cargar("features"), config.cargar("model")
    semilla, objetivo = base["seed"], base["objetivo"]["columna"]
    umbral = float(modelo_cfg["umbrales"]["Random Forest"])
    n_arboles = modelo_cfg["modelos"]["Random Forest"]["n_estimators"]  # 300, como en la notebook

    ruta_train = args.train or str(config.ruta(base, "datos_procesados") / "train.parquet")
    ruta_test = args.test or ruta_train.replace("train.parquet", "test.parquet")
    train = leer_parquet(ruta_train)
    log.info("train: %d filas, %.2f %% reingresos", len(train), 100 * train[objetivo].mean())

    uri = args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or TRACKING_POR_DEFECTO
    mlflow.set_tracking_uri(uri)
    id_experimento = obtener_o_crear_experimento(args.experimento)
    log.info("MLflow: %s | experimento %s (id %s)", uri, args.experimento, id_experimento)

    folds = preparar_folds(train, objetivo, feat, semilla, N_FOLDS)
    X, y = train[columnas_entrada(feat)], train[objetivo].to_numpy()
    mejor = {"auc": -1.0}

    estudio = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=semilla))

    nombre_padre = "best_hyperparam_" + datetime.datetime.today().strftime("%Y/%m/%d-%H:%M:%S")
    with mlflow.start_run(experiment_id=id_experimento, run_name=nombre_padre) as padre:
        mlflow.log_params({"n_intentos": N_INTENTOS, "n_arboles": n_arboles,
                           "n_folds": N_FOLDS, "semilla": semilla, "filas_train": len(train), "umbral": umbral})
        mlflow.set_tags({"project": "Readmision Diabetes", "optimizer_engine": "optuna",
                         "model_family": "sklearn", "feature_set_version": 1})

        def objetivo_optuna(trial: optuna.Trial) -> float:
            """Evalúa con validación cruzada la combinación que propone Optuna y la registra como run anidado.

            Args:
                trial: intento en curso del estudio de Optuna.

            Returns:
                ROC-AUC medio de los folds, que Optuna intenta maximizar.
            """
            params = sugerir(trial)
            with mlflow.start_run(experiment_id=id_experimento, run_name=f"Trial: {trial.number}", nested=True):
                t0 = time.perf_counter()
                aucs, oof = evaluar(params, folds, n_arboles, semilla)
                mlflow.log_params(params)
                mlflow.log_metrics({"auc_media": aucs.mean(), "auc_desvio": aucs.std(ddof=1),
                                    "segundos": time.perf_counter() - t0})
            log.info("intento %d/%d | AUC %.4f ± %.4f | %s", trial.number + 1, N_INTENTOS, aucs.mean(),
                     aucs.std(ddof=1), params)
            if aucs.mean() > mejor["auc"]:
                mejor.update(auc=float(aucs.mean()), desvio=float(aucs.std(ddof=1)), oof=oof)
            return float(aucs.mean())

        estudio.optimize(objetivo_optuna, n_trials=N_INTENTOS, callbacks=[callback_campeon])

        # Mejor combinación: métricas fuera de fold, reentrenamiento con todo el train y registro.
        p = estudio.best_params
        mlflow.log_params({f"mejor_{k}": v for k, v in p.items()})
        mlflow.log_metrics({"best_train_auc": estudio.best_value, "mejor_auc_desvio": mejor["desvio"],
                            "oof_recall_umbral": recall_score(y, mejor["oof"] >= umbral),
                            "oof_precision_umbral": precision_score(y, mejor["oof"] >= umbral, zero_division=0)})
        with tempfile.TemporaryDirectory() as tmp:
            png = Path(tmp) / "calibracion_oof.png"
            curva_calibracion(y, mejor["oof"], png)
            mlflow.log_artifact(str(png))

        prep = construir_preprocesador(feat, semilla)
        Z = prep.fit_transform(X, y)
        rf = RandomForestClassifier(n_estimators=n_arboles, n_jobs=-1, random_state=semilla, **p).fit(Z, y)
        # El Pipeline junta preprocesador y modelo: quien sirva el modelo recibe las 18 variables (más
        # `patient_nbr`, que el preprocesador pide al final solo para agrupar) y no tiene que repetir pasos.
        modelo = Pipeline([("prep", prep), ("rf", rf)])

        # Test: solo se informa, la elección de hiperparámetros ya terminó (nada de filtración de datos).
        metricas_test = {}
        try:
            test = leer_parquet(ruta_test)
            pt = modelo.predict_proba(test[columnas_entrada(feat)])[:, 1]
            yt = test[objetivo].to_numpy()
            metricas_test = {"test_roc_auc": roc_auc_score(yt, pt), "test_f1_umbral": f1_score(yt, pt >= umbral),
                             "test_recall_umbral": recall_score(yt, pt >= umbral),
                             "test_precision_umbral": precision_score(yt, pt >= umbral, zero_division=0)}
            mlflow.log_metrics(metricas_test)
        except Exception as error:  # el test es informativo: si no está, el registro del modelo sigue
            log.warning("no se pudo evaluar en el test (%s): %s", ruta_test, error)

        # cloudpickle porque el preprocesador usa clases propias (Winsorizador, etc.) que skops no admite.
        # Cargar el modelo exige tener el paquete del proyecto instalado y confiar en el servidor de MLflow.
        muestra = X.head(100)
        info = mlflow.sklearn.log_model(name="model", sk_model=modelo, signature=infer_signature(muestra, modelo.predict(muestra)),
                                        serialization_format="cloudpickle",
                                        registered_model_name=f"{args.nombre_modelo}_dev")

        tags = {k: str(v) for k, v in {**modelo[-1].get_params(), "model": "RandomForestClassifier",
                                       "umbral": umbral, "best_train_auc": round(estudio.best_value, 4),
                                       **{k: round(v, 4) for k, v in metricas_test.items()}}.items()}
        version = promover_a_prod(MlflowClient(), f"{args.nombre_modelo}_prod", info.model_uri, info.run_id, tags)
        log.info("modelo %s_prod v%s con alias champion | AUC fuera de fold %.4f | run %s", args.nombre_modelo,
                 version, estudio.best_value, padre.info.run_id)


if __name__ == "__main__":
    main()
