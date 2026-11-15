"""DAG de ETL del dataset Diabetes 130-US Hospitals (1999-2008).

Primer tramo del pipeline de reingreso hospitalario temprano (menos de 30 dias)
en pacientes diabeticos. Por ahora descarga los datos originales desde UCI y los
deja sin modificar en el bucket de datos de MinIO.
"""

import datetime

from airflow.decorators import dag, task

markdown_text = """
### ETL del dataset de reingreso hospitalario en pacientes diabeticos

Descarga el dataset [Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
del repositorio de UCI y lo guarda sin cambios en `s3://data/raw/diabetic_data.csv`.
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

    obtener_datos_originales()


etl_diabetes()