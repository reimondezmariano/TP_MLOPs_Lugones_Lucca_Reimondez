"""Lectura de la configuracion del proyecto desde la carpeta configs/."""

from pathlib import Path

import yaml

# Este archivo esta en src/readmission/, asi que la raiz del repo esta dos niveles arriba.
RAIZ = Path(__file__).resolve().parents[2]
CARPETA_CONFIGS = RAIZ / "configs"


def cargar(nombre: str) -> dict:
    """Lee un archivo de configuracion de configs/.

    Args:
        nombre: nombre del archivo sin la extension, por ejemplo "base" o "features".

    Returns:
        Contenido del archivo como diccionario.
    """
    ruta_archivo = CARPETA_CONFIGS / f"{nombre}.yaml"
    with open(ruta_archivo, encoding="utf-8") as archivo:
        return yaml.safe_load(archivo)


def ruta(base: dict, clave: str) -> Path:
    """Devuelve una ruta de base.yaml como ruta absoluta dentro del repo.

    Args:
        base: contenido de configs/base.yaml.
        clave: nombre de la ruta dentro de la seccion rutas, por ejemplo "datos_procesados".

    Returns:
        Ruta absoluta.
    """
    return RAIZ / base["rutas"][clave]