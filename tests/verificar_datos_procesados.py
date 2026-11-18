"""Verifica que los datos procesados del ETL coincidan con su descripcion en data.json.

Lee de MinIO train, test y data.json, y controla que coincidan columnas, tipos,
categorias y cantidad de filas, y que ningun paciente quede en train y en test.

Uso, con los contenedores levantados y desde la raiz del repo:
    python tests/verificar_datos_procesados.py
"""

import io
import json
import os

import boto3
import pandas as pd

ENDPOINT = os.environ.get("AWS_ENDPOINT_URL_S3", "http://localhost:9000")
BUCKET = "data"


def normalizar_tipo(tipo: str) -> str:
    return "texto" if tipo in ("object", "str", "string") else tipo


def leer(s3, key: str) -> bytes:
    return s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()


def main():
    s3 = boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "minio"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "minio123"),
    )
    info = json.loads(leer(s3, "data_info/data.json"))
    train = pd.read_parquet(io.BytesIO(leer(s3, "processed/train.parquet")))
    test = pd.read_parquet(io.BytesIO(leer(s3, "processed/test.parquet")))

    objetivo = info["columna_objetivo"]
    for nombre, tabla in [("train", train), ("test", test)]:
        entradas = tabla.drop(columns=[objetivo])
        assert entradas.columns.to_list() == info["columnas"], f"{nombre}: las columnas no coinciden"
        # Las columnas de texto se comparan como "texto": segun la version de pandas,
        # el mismo tipo se llama "object" o "str".
        tipos = {c: normalizar_tipo(str(t)) for c, t in entradas.dtypes.items()}
        esperados = {c: normalizar_tipo(t) for c, t in info["tipos"].items()}
        assert tipos == esperados, f"{nombre}: los tipos no coinciden: {tipos} contra {esperados}"
        assert tabla[objetivo].isin([0, 1]).all(), f"{nombre}: el objetivo no es 0/1"
        for col in info["columnas_categoricas"]:
            nuevas = set(entradas[col].astype(str)) - set(info["categorias"][col])
            # En test puede aparecer alguna categoria que no esta en train: se avisa sin cortar,
            # porque el preprocesador la resuelve (target encoding con la media global).
            if nuevas and nombre == "train":
                raise AssertionError(f"train: {col} tiene categorias fuera de data.json: {nuevas}")
            if nuevas:
                print(f"Aviso: en test, {col} tiene categorias que no estan en train: {sorted(nuevas)}")
        print(f"{nombre}: columnas, tipos y categorias coinciden con data.json")

    assert len(train) == info["filas_train"], "train: la cantidad de filas no coincide"
    assert len(test) == info["filas_test"], "test: la cantidad de filas no coincide"
    id_grupo = info["id_grupo"]
    assert not set(train[id_grupo]) & set(test[id_grupo]), "Hay pacientes en train y en test"

    print(f"Todo OK: train {len(train)} filas, test {len(test)} filas, sin pacientes compartidos.")


if __name__ == "__main__":
    main()
