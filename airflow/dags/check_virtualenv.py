"""DAG de verificación del entorno de ejecución de Airflow.

Comprueba, antes de escribir los DAGs reales, que:

1. Las variables de airflow/secrets/variables.yaml se leen bien.
2. Las tareas @task.virtualenv crean su entorno e instalan dependencias
   en el worker de Celery.
3. Dentro del entorno virtual llegan las credenciales y el endpoint de MinIO.
4. Se puede escribir y leer Parquet en MinIO pasando solo la ruta por XCom.

Se corre a mano desde la interfaz de Airflow. Si las tres tareas terminan en
verde, la infraestructura está lista para el DAG de ETL.
"""

from datetime import datetime

from airflow.sdk import Variable, dag, task

# Carpeta donde el worker guarda los entornos ya creados. Sin esto, cada
# corrida reinstala todas las dependencias y la tarea tarda varios minutos.
VENV_CACHE = "/tmp/venv_cache"

REQUIREMENTS = ["pandas", "pyarrow", "boto3"]


@dag(
    dag_id="check_virtualenv",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["infraestructura", "verificacion"],
)
def check_virtualenv():
    """Verifica variables, entornos virtuales y acceso a MinIO."""

    @task
    def check_variables() -> dict:
        """Lee las variables que van a usar los DAGs del proyecto.

        Returns:
            dict: Bucket de datos y semilla, para pasarlos a las tareas
            siguientes.

        Raises:
            KeyError: Si falta alguna variable en variables.yaml.
        """
        required = [
            "data_bucket",
            "raw_data_path",
            "processed_data_path",
            "data_info_path",
            "target_column",
            "random_seed",
            "test_size",
            "mlflow_tracking_uri",
        ]
        values = {name: Variable.get(name) for name in required}
        for name, value in values.items():
            print(f"{name} = {value}")
        return {"bucket": values["data_bucket"], "seed": values["random_seed"]}

    @task.virtualenv(
        requirements=REQUIREMENTS,
        system_site_packages=False,
        venv_cache_path=VENV_CACHE,
    )
    def write_parquet(config: dict) -> str:
        """Escribe un Parquet chico en MinIO desde un entorno virtual.

        Los imports van dentro de la función porque se ejecuta en otro
        proceso, con el intérprete del entorno virtual.

        Args:
            config: Bucket y semilla devueltos por check_variables.

        Returns:
            str: Ruta s3:// del archivo escrito.
        """
        import io
        import os

        import boto3
        import numpy as np
        import pandas as pd

        # boto3 toma credenciales y endpoint del entorno. Si faltan acá,
        # el operador no está heredando las variables del worker.
        for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_ENDPOINT_URL_S3"):
            if not os.environ.get(name):
                raise RuntimeError(f"{name} no llegó al entorno virtual")

        rng = np.random.default_rng(int(config["seed"]))
        df = pd.DataFrame({"x": rng.normal(size=100), "y": rng.integers(0, 2, size=100)})

        # Se serializa en memoria para no dejar archivos en el disco del worker.
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)

        key = "tmp/check_virtualenv/sample.parquet"
        boto3.client("s3").put_object(Bucket=config["bucket"], Key=key, Body=buffer.getvalue())
        path = f"s3://{config['bucket']}/{key}"
        print(f"Escrito {path} con {len(df)} filas")
        return path

    @task.virtualenv(
        requirements=REQUIREMENTS,
        system_site_packages=False,
        venv_cache_path=VENV_CACHE,
    )
    def read_parquet(path: str) -> None:
        """Lee el Parquet que dejó la tarea anterior y lo borra.

        Args:
            path: Ruta s3:// recibida por XCom.

        Raises:
            ValueError: Si el archivo no tiene la forma esperada.
        """
        import io

        import boto3
        import pandas as pd

        bucket, key = path.removeprefix("s3://").split("/", 1)
        s3 = boto3.client("s3")
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        df = pd.read_parquet(io.BytesIO(body))

        if df.shape != (100, 2):
            raise ValueError(f"Forma inesperada: {df.shape}")
        print(f"Leído {path}: {df.shape}")

        s3.delete_object(Bucket=bucket, Key=key)

    read_parquet(write_parquet(check_variables()))


check_virtualenv()
