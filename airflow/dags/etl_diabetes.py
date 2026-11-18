"""DAG de ETL del dataset Diabetes 130-US Hospitals (1999-2008).

Pipeline de reingreso hospitalario temprano (menos de 30 dias) en pacientes
diabeticos. Descarga los datos originales desde UCI, los limpia y arma las
variables nuevas siguiendo la Parte 1 de notebooks/TP_FINAL_OAA/TP_FINAL_OAA.ipynb.
"""

import datetime

from airflow.decorators import dag, task

markdown_text = """
### ETL del dataset de reingreso hospitalario en pacientes diabeticos

1. Descarga el dataset [Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
   de UCI y lo guarda sin cambios en `s3://data/raw/diabetic_data.csv`.
2. Limpia los datos y arma las variables nuevas con las reglas que no dependen del split,
   y deja el resultado en `s3://data/interim/diabetic_limpio.parquet`.
3. Separa train y test agrupando por paciente y los guarda en `s3://data/processed`.
4. Guarda la descripcion de los datos en `s3://data/data_info/data.json` y registra
   el linaje del dataset en MLflow.
"""

default_args = {
    "owner": "Florencia Lucca",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
}


@dag(
    dag_id="etl_diabetes",
    description="ETL del dataset de diabetes: datos originales, limpieza y split en train y test.",
    doc_md=markdown_text,
    tags=["ETL", "Diabetes"],
    default_args=default_args,
    # Corrida manual: el ETL se dispara cuando hace falta regenerar los datos.
    schedule=None,
    # Fecha en el pasado para que Airflow habilite la corrida desde el primer momento.
    start_date=datetime.datetime(2026, 11, 1),
    catchup=False,
    dagrun_timeout=datetime.timedelta(minutes=30),
)
def etl_diabetes():

    @task.virtualenv(
        task_id="obtener_datos_originales",
        requirements=["ucimlrepo==0.0.3", "pandas", "boto3"],
        system_site_packages=True,
    )
    def obtener_datos_originales() -> str:
        """Descarga el dataset desde UCI y lo guarda sin cambios en S3.

        Returns:
            Ruta en S3 del CSV original, que recibe la tarea siguiente por XCom.
        """
        import io

        import boto3
        from ucimlrepo import fetch_ucirepo

        bucket = "data"
        key = "raw/diabetic_data.csv"

        # data.original conserva encounter_id y patient_nbr. patient_nbr hace
        # falta despues para que un mismo paciente no quede en train y en test.
        datos = fetch_ucirepo(id=296).data.original
        print(f"Dataset descargado: {datos.shape[0]} filas x {datos.shape[1]} columnas")

        # Se serializa en memoria y no en el disco del worker: la tarea
        # siguiente puede correr en otro worker y no veria ese archivo.
        buffer = io.StringIO()
        datos.to_csv(buffer, index=False)
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())

        ruta = f"s3://{bucket}/{key}"
        print(f"Datos originales guardados en {ruta}")
        return ruta

    @task.virtualenv(
        task_id="limpiar_y_crear_variables",
        requirements=["pandas", "pyarrow<26", "boto3"],
        system_site_packages=True,
    )
    def limpiar_y_crear_variables(ruta_raw: str) -> str:
        """Limpia el dataset y arma las variables nuevas (Parte 1 de la notebook).

        Solo aplica reglas fijas, que no dependen de la particion: la moda de
        race y el recorte en el percentil 99 se calculan con train en la tarea
        del split.

        Args:
            ruta_raw: ruta en S3 del CSV original.

        Returns:
            Ruta en S3 del Parquet con los datos limpios.
        """
        import io

        import boto3
        import numpy as np
        import pandas as pd

        s3 = boto3.client("s3")
        bucket, key_raw = ruta_raw.replace("s3://", "").split("/", 1)
        objeto = s3.get_object(Bucket=bucket, Key=key_raw)
        datos = pd.read_csv(io.BytesIO(objeto["Body"].read()), low_memory=False)

        # 1.2 Recodificacion de tipos. El "?" ya llega como nulo desde UCI,
        # pero se reemplaza igual por si el origen vuelve a ser el CSV.
        datos = datos.replace(r"^\s*\?\s*$", np.nan, regex=True)
        cols_texto = datos.select_dtypes(include=["object", "string"]).columns.tolist()
        for c in cols_texto:
            datos[c] = datos[c].str.strip()

        cols_enteras = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
                        "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]
        for c in cols_enteras:
            datos[c] = datos[c].astype(int)

        mapa_admission_type = {
            1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Newborn",
            5: "Not Available", 6: "NULL", 7: "Trauma Center", 8: "Not Mapped",
        }
        mapa_discharge_disposition = {
            1: "Discharged to home", 2: "Discharged/transferred to another short term hospital",
            3: "Discharged/transferred to SNF", 4: "Discharged/transferred to ICF",
            5: "Discharged/transferred to another type of inpatient care institution",
            6: "Discharged/transferred to home with home health service", 7: "Left AMA",
            8: "Discharged/transferred to home under care of Home IV provider",
            9: "Admitted as an inpatient to this hospital",
            10: "Neonate discharged to another hospital for neonatal aftercare",
            11: "Expired", 12: "Still patient or expected to return for outpatient services",
            13: "Hospice / home", 14: "Hospice / medical facility",
            15: "Discharged/transferred within this institution to Medicare approved swing bed",
            16: "Discharged/transferred/referred another institution for outpatient services",
            17: "Discharged/transferred/referred to this institution for outpatient services",
            18: "NULL", 19: "Expired at home. Medicaid only, hospice.",
            20: "Expired in a medical facility. Medicaid only, hospice.",
            21: "Expired, place unknown. Medicaid only, hospice.",
            22: "Discharged/transferred to another rehab fac including rehab units of a hospital",
            23: "Discharged/transferred to a long term care hospital",
            24: "Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare",
            25: "Not Mapped", 26: "Unknown/Invalid",
            27: "Discharged/transferred to a federal health care facility",
            28: "Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital",
            29: "Discharged/transferred to a Critical Access Hospital (CAH)",
        }
        mapa_admission_source = {
            1: "Physician Referral", 2: "Clinic Referral", 3: "HMO Referral",
            4: "Transfer from a hospital", 5: "Transfer from a Skilled Nursing Facility (SNF)",
            6: "Transfer from another health care facility", 7: "Emergency Room",
            8: "Court/Law Enforcement", 9: "Not Available",
            10: "Transfer from a critical access hospital", 11: "Normal Delivery",
            12: "Premature Delivery", 13: "Sick Baby", 14: "Extramural Birth",
            15: "Not Available", 17: "NULL", 18: "Transfer From Another Home Health Agency",
            19: "Readmission to Same Home Health Agency", 20: "Not Mapped",
            21: "Unknown/Invalid",
            22: "Transfer from hospital inpt/same facility resulting in a separate claim",
            23: "Born inside this hospital", 24: "Born outside this hospital",
            25: "Transfer from Ambulatory Surgery Center", 26: "Transfer from Hospice",
        }
        datos["admission_type_id"] = datos["admission_type_id"].astype(int).map(mapa_admission_type)
        datos["discharge_disposition_id"] = datos["discharge_disposition_id"].astype(int).map(mapa_discharge_disposition)
        datos["admission_source_id"] = datos["admission_source_id"].astype(int).map(mapa_admission_source)

        def agrupar_diag(codigo):
            codigo = str(codigo)
            if codigo == "nan":
                return np.nan
            if codigo.startswith("V"):
                return "Factores de salud"
            if codigo.startswith("E"):
                return "Causas externas"
            try:
                c = float(codigo)
            except ValueError:
                return "Otro"
            if 390 <= c <= 459: return "Circulatorio"
            if 460 <= c <= 519: return "Respiratorio"
            if 520 <= c <= 579: return "Digestivo"
            if 250 <= c < 251: return "Diabetes"
            if 800 <= c <= 999: return "Lesiones/cirugia traumatologica"
            if 580 <= c <= 629: return "Genitourinario"
            if 710 <= c <= 739: return "Musculoesqueletico"
            if 140 <= c <= 239: return "Neoplasias"
            return "Otro"

        for c in ["diag_1", "diag_2", "diag_3"]:
            datos[c] = datos[c].apply(agrupar_diag)

        # 1.6 Las 3 filas con gender invalido se descartan antes del split:
        # sin este paso el split no reproduce el de la notebook.
        datos = datos[datos["gender"] != "Unknown/Invalid"].copy()

        # 2 Target binario: reingreso en menos de 30 dias.
        datos["readmitted_temprano"] = (datos["readmitted"] == "<30").astype(int)

        # 4.1 Faltantes con reglas fijas.
        datos = datos.drop(columns=["weight"])
        for c in ["max_glu_serum", "A1Cresult"]:
            datos[c] = datos[c].fillna("No medido")  # el faltante indica que no se pidio el analisis
        for c in ["medical_specialty", "payer_code"]:
            datos[c] = datos[c].fillna("Faltante")
        for c in ["diag_1", "diag_2", "diag_3"]:
            datos[c] = datos[c].fillna("Sin_diagnostico")

        # 5 Feature engineering.
        datos = datos.sort_values(["patient_nbr", "encounter_id"]).reset_index(drop=True)
        encuentros_previos = datos.groupby("patient_nbr").cumcount()
        datos["patient_nbr_ordinal"] = encuentros_previos.apply(lambda x: str(x) if x <= 4 else "5 o mas")

        mapa_familias = {
            "metformin": ["Biguanidas"], "repaglinide": ["Meglitinidas"], "nateglinide": ["Meglitinidas"],
            "chlorpropamide": ["Sulfonilureas"], "glimepiride": ["Sulfonilureas"], "acetohexamide": ["Sulfonilureas"],
            "glipizide": ["Sulfonilureas"], "glyburide": ["Sulfonilureas"], "tolbutamide": ["Sulfonilureas"],
            "pioglitazone": ["Glitazonas"], "rosiglitazone": ["Glitazonas"], "acarbose": ["Alfa_glucosidasa"],
            "miglitol": ["Alfa_glucosidasa"], "troglitazone": ["Glitazonas"], "tolazamide": ["Sulfonilureas"],
            "examide": [], "citoglipton": [], "insulin": ["Insulina"],
            "glyburide-metformin": ["Sulfonilureas", "Biguanidas"], "glipizide-metformin": ["Sulfonilureas", "Biguanidas"],
            "glimepiride-pioglitazone": ["Sulfonilureas", "Glitazonas"], "metformin-rosiglitazone": ["Biguanidas", "Glitazonas"],
            "metformin-pioglitazone": ["Biguanidas", "Glitazonas"],
        }
        familias = ["Biguanidas", "Sulfonilureas", "Meglitinidas", "Glitazonas", "Alfa_glucosidasa", "Insulina"]
        # Si una familia tiene varias drogas, queda el cambio mas relevante.
        prioridad = {"Up": 3, "Down": 2, "Steady": 1, "No": 0}
        for familia in familias:
            cols_familia = [droga for droga, fams in mapa_familias.items() if familia in fams]
            datos[f"familia_{familia.lower()}"] = datos[cols_familia].apply(
                lambda fila: max(fila, key=lambda v: prioridad.get(v, 0)), axis=1
            )

        def discretizar_conteo(x):
            return str(x) if x <= 2 else "3 o mas"

        datos["number_emergency_disc"] = datos["number_emergency"].apply(discretizar_conteo)
        datos["number_inpatient_disc"] = datos["number_inpatient"].apply(discretizar_conteo)

        print(f"Datos limpios: {datos.shape[0]} filas x {datos.shape[1]} columnas")
        print(f"Tasa de reingreso temprano: {datos['readmitted_temprano'].mean() * 100:.2f} %")
        print(f"Nulos restantes fuera de race: {int(datos.drop(columns=['race']).isnull().sum().sum())}")

        key_limpio = "interim/diabetic_limpio.parquet"
        buffer = io.BytesIO()
        datos.to_parquet(buffer, index=False)
        s3.put_object(Bucket=bucket, Key=key_limpio, Body=buffer.getvalue())

        ruta = f"s3://{bucket}/{key_limpio}"
        print(f"Datos limpios guardados en {ruta}")
        return ruta

    @task(task_id="leer_parametros")
    def leer_parametros() -> dict:
        """Lee de las variables de Airflow los parametros del split.

        Returns:
            Diccionario con test_size, la semilla, la ruta del data.json y los datos de MLflow.
        """
        from airflow.sdk import Variable

        # Los valores por defecto son los de la notebook, por si variables.yaml no los define.
        return {
            "test_size": float(Variable.get("test_size", default="0.30")),
            "semilla": int(Variable.get("random_seed", default="42")),
            "data_info_path": Variable.get("data_info_path", default="s3://data/data_info/data.json"),
            "mlflow_tracking_uri": Variable.get("mlflow_tracking_uri", default="http://mlflow:5000"),
            "mlflow_experiment_name": Variable.get("mlflow_experiment_name", default="reingreso_diabetes"),
        }

    @task.virtualenv(
        task_id="separar_train_test",
        requirements=["pandas", "pyarrow<26", "boto3", "scikit-learn==1.9.1"],
        system_site_packages=True,
    )
    def separar_train_test(ruta_limpio: str, parametros: dict) -> dict:
        """Separa train y test por paciente y guarda los datos procesados (seccion 2.2).

        Los valores se guardan sin recortar: el recorte en el percentil 99 lo hace
        el preprocesador del modelo, ajustado solo con train.

        Args:
            ruta_limpio: ruta en S3 del Parquet con los datos limpios.
            parametros: test_size y semilla leidos de las variables de Airflow.

        Returns:
            Rutas en S3 de train y test.
        """
        import io

        import boto3
        import pandas as pd
        from sklearn.model_selection import GroupShuffleSplit

        s3 = boto3.client("s3")
        bucket, key_limpio = ruta_limpio.replace("s3://", "").split("/", 1)
        objeto = s3.get_object(Bucket=bucket, Key=key_limpio)
        datos = pd.read_parquet(io.BytesIO(objeto["Body"].read()))

        # Agrupado por paciente: un mismo paciente nunca queda en train y en test.
        gss = GroupShuffleSplit(n_splits=1, test_size=parametros["test_size"],
                                random_state=parametros["semilla"])
        idx_train, idx_test = next(gss.split(datos, groups=datos["patient_nbr"]))
        train = datos.iloc[idx_train].copy()
        test = datos.iloc[idx_test].copy()

        pacientes_en_ambos = set(train["patient_nbr"]) & set(test["patient_nbr"])
        assert len(pacientes_en_ambos) == 0, "Hay pacientes en train y en test"

        # Las 18 variables de la seccion 6.3 y patient_nbr como ultima columna de entrada:
        # el modelo la usa solo para agrupar por paciente en el target encoding.
        variables = [
            "number_inpatient_disc", "patient_nbr_ordinal", "discharge_disposition_id",
            "number_diagnoses", "number_emergency_disc", "num_lab_procedures",
            "time_in_hospital", "medical_specialty", "num_medications", "familia_insulina",
            "age", "diag_1", "payer_code", "diag_3", "diag_2", "admission_source_id",
            "familia_biguanidas", "A1Cresult",
        ]
        columnas = variables + ["patient_nbr", "readmitted_temprano"]
        train = train[columnas].reset_index(drop=True)
        test = test[columnas].reset_index(drop=True)

        print(f"Train: {len(train)} filas ({len(train) / len(datos) * 100:.1f} %), "
              f"{train['patient_nbr'].nunique()} pacientes")
        print(f"Test: {len(test)} filas ({len(test) / len(datos) * 100:.1f} %)")
        print(f"Tasa de reingreso temprano: train {train['readmitted_temprano'].mean() * 100:.2f} %, "
              f"test {test['readmitted_temprano'].mean() * 100:.2f} %")

        rutas = {}
        for nombre, tabla in [("train", train), ("test", test)]:
            key = f"processed/{nombre}.parquet"
            buffer = io.BytesIO()
            tabla.to_parquet(buffer, index=False)
            s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
            rutas[nombre] = f"s3://{bucket}/{key}"
            print(f"{nombre} guardado en {rutas[nombre]}")

        return rutas

    @task.virtualenv(
        task_id="registrar_info_y_linaje",
        requirements=["pandas", "pyarrow<26", "boto3", "mlflow==3.16.1"],
        system_site_packages=True,
    )
    def registrar_info_y_linaje(rutas: dict, parametros: dict) -> str:
        """Guarda la descripcion de los datos procesados y registra el linaje en MLflow.

        El data.json describe las columnas de entrada del modelo: tipos y categorias
        vistas en train. Lo usan el reentrenamiento y la API para validar lo que reciben.

        Args:
            rutas: rutas en S3 de train y test.
            parametros: parametros leidos de las variables de Airflow.

        Returns:
            Ruta en S3 del data.json.
        """
        import datetime
        import io
        import json

        import boto3
        import mlflow
        import pandas as pd

        s3 = boto3.client("s3")
        tablas = {}
        for nombre, ruta in rutas.items():
            bucket, key = ruta.replace("s3://", "").split("/", 1)
            objeto = s3.get_object(Bucket=bucket, Key=key)
            tablas[nombre] = pd.read_parquet(io.BytesIO(objeto["Body"].read()))
        train, test = tablas["train"], tablas["test"]

        objetivo = "readmitted_temprano"
        id_grupo = "patient_nbr"
        entradas = train.drop(columns=[objetivo])
        numericas = [c for c in entradas.columns
                     if c != id_grupo and pd.api.types.is_numeric_dtype(entradas[c])]
        categoricas = [c for c in entradas.columns
                       if c != id_grupo and c not in numericas]

        fecha = datetime.datetime.now().strftime("%Y/%m/%d-%H:%M:%S")
        info = {
            "columnas": entradas.columns.to_list(),
            "columna_objetivo": objetivo,
            "id_grupo": id_grupo,
            "columnas_numericas": numericas,
            "columnas_categoricas": categoricas,
            "tipos": {c: str(t) for c, t in entradas.dtypes.items()},
            "categorias": {c: sorted(entradas[c].astype(str).unique().tolist())
                           for c in categoricas},
            "filas_train": len(train),
            "filas_test": len(test),
            "tasa_reingreso_train": round(float(train[objetivo].mean()), 4),
            "tasa_reingreso_test": round(float(test[objetivo].mean()), 4),
            "test_size": parametros["test_size"],
            "semilla": parametros["semilla"],
            "fecha": fecha,
        }

        bucket_info, key_info = parametros["data_info_path"].replace("s3://", "").split("/", 1)
        s3.put_object(Bucket=bucket_info, Key=key_info,
                      Body=json.dumps(info, indent=2, ensure_ascii=False).encode("utf-8"))
        print(f"data.json guardado en {parametros['data_info_path']}")

        # Linaje: deja registrado en MLflow con que datos se va a entrenar.
        mlflow.set_tracking_uri(parametros["mlflow_tracking_uri"])
        experimento = mlflow.set_experiment(parametros["mlflow_experiment_name"])
        fuente = "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008"
        with mlflow.start_run(run_name=f"ETL_run_{fecha}",
                              experiment_id=experimento.experiment_id,
                              tags={"experiment": "etl", "dataset": "Diabetes 130-US Hospitals"}):
            for nombre, tabla in [("train", train), ("test", test)]:
                dataset = mlflow.data.from_pandas(tabla, source=fuente, targets=objetivo,
                                                  name=f"diabetes_{nombre}")
                mlflow.log_input(dataset, context=nombre)
            mlflow.log_params({
                "test_size": parametros["test_size"],
                "semilla": parametros["semilla"],
                "filas_train": len(train),
                "filas_test": len(test),
                "pacientes_train": int(train[id_grupo].nunique()),
            })
            mlflow.log_dict(info, "data_info/data.json")

        return parametros["data_info_path"]

    parametros = leer_parametros()
    ruta_raw = obtener_datos_originales()
    ruta_limpio = limpiar_y_crear_variables(ruta_raw)
    rutas = separar_train_test(ruta_limpio, parametros)
    registrar_info_y_linaje(rutas, parametros)


etl_diabetes()