"""Preprocesamiento de las 18 variables de entrada del modelo (seccion 7 de la notebook).

Arma el mismo preprocesador que la notebook, a partir de configs/features.yaml:
recorte en el percentil 99 de dos numericas, ordinal encoding, target encoding
out-of-fold con suavizado agrupado por paciente, one-hot encoding y escalado.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


class Winsorizador(BaseEstimator, TransformerMixin):
    """Recorta en el percentil q, aprendido en el fit, las columnas indicadas por posicion.

    Args:
        posiciones: posiciones de las columnas a recortar.
        q: percentil del recorte, entre 0 y 1.
    """

    def __init__(self, posiciones=(), q=0.99):
        self.posiciones = posiciones
        self.q = q

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.techos_ = {j: np.quantile(X[:, j], self.q) for j in self.posiciones}
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)
        for j, techo in self.techos_.items():
            X[:, j] = np.minimum(X[:, j], techo)
        return X


class QuitarColumna(BaseEstimator, TransformerMixin):
    """Elimina una columna por posicion: la categoria de referencia del one-hot.

    Args:
        indice: posicion de la columna a eliminar.
    """

    def __init__(self, indice=0):
        self.indice = indice

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        return np.delete(np.asarray(X, dtype=float), self.indice, axis=1)


def _medias_suavizadas(y_fit, x_fit, media_global, m):
    """Media del target por categoria, suavizada hacia la media global con peso m."""
    agg = y_fit.groupby(x_fit.astype(object), observed=True).agg(["mean", "count"])
    return (agg["mean"] * agg["count"] + media_global * m) / (agg["count"] + m)


class CodificadorObjetivoAgrupado(BaseEstimator, TransformerMixin):
    """Target encoding out-of-fold con suavizado, con folds agrupados por paciente.

    La ultima columna recibida es el identificador del paciente: solo sirve para
    agrupar los folds internos y no se devuelve.

    Args:
        suavizado: peso de la media global en el suavizado.
        n_folds: cantidad de folds internos.
        semilla: semilla de los folds.
    """

    def __init__(self, suavizado=20, n_folds=5, semilla=42):
        self.suavizado = suavizado
        self.n_folds = n_folds
        self.semilla = semilla

    def fit(self, X, y):
        cats, y = X.iloc[:, :-1].reset_index(drop=True), pd.Series(np.asarray(y))
        self.media_global_ = float(y.mean())
        self.medias_ = {c: _medias_suavizadas(y, cats[c], self.media_global_, self.suavizado)
                        for c in cats.columns}
        return self

    def fit_transform(self, X, y):
        # Las filas de entrenamiento se codifican fuera de fold, para que una fila
        # no use su propio target.
        cats, y = X.iloc[:, :-1].reset_index(drop=True), pd.Series(np.asarray(y))
        grupos = X.iloc[:, -1].to_numpy()
        self.fit(X, y)
        out = np.full(cats.shape, np.nan)
        folds = StratifiedGroupKFold(n_splits=self.n_folds, shuffle=True, random_state=self.semilla)
        for idx_fit, idx_apl in folds.split(cats, y, groups=grupos):
            for j, c in enumerate(cats.columns):
                medias = _medias_suavizadas(y.iloc[idx_fit], cats[c].iloc[idx_fit],
                                            self.media_global_, self.suavizado)
                out[idx_apl, j] = (cats[c].astype(object).iloc[idx_apl]
                                   .map(medias).astype(float).to_numpy())
        return np.where(np.isnan(out), self.media_global_, out)

    def transform(self, X):
        cats = X.iloc[:, :-1].reset_index(drop=True)
        out = np.column_stack([cats[c].astype(object).map(self.medias_[c]).astype(float).to_numpy()
                               for c in cats.columns])
        return np.where(np.isnan(out), self.media_global_, out)


def columnas_entrada(feat: dict) -> list:
    """Columnas que recibe el modelo, en orden: las 18 variables y el identificador al final.

    Args:
        feat: contenido de configs/features.yaml.

    Returns:
        Lista de nombres de columnas.
    """
    variables = (list(feat["numericas"]) + list(feat["ordinales"])
                 + list(feat["target_encoding"]["columnas"]) + list(feat["one_hot"]["niveles"]))
    return variables + [feat["id_grupo"]]


def construir_preprocesador(feat: dict, semilla: int) -> Pipeline:
    """Arma el preprocesador de la seccion 7 de la notebook.

    Las categorias de cada variable son fijas (vienen de features.yaml), asi que
    no dependen de los datos de cada fold.

    Args:
        feat: contenido de configs/features.yaml.
        semilla: semilla de los folds internos del target encoding.

    Returns:
        Pipeline sin ajustar: ColumnTransformer y escalado.
    """
    numericas = list(feat["numericas"])
    ordinales = list(feat["ordinales"])
    te = feat["target_encoding"]
    niveles_ohe = feat["one_hot"]["niveles"]
    vars_ohe = list(niveles_ohe)

    posiciones_winsor = [numericas.index(c) for c in feat["winsorizar"]["columnas"]]

    # Posicion de la categoria de referencia dentro de todas las columnas del one-hot,
    # que siguen el orden de vars_ohe.
    ref = feat["one_hot"]["referencia"]
    k = vars_ohe.index(ref["variable"])
    indice_referencia = (sum(len(niveles_ohe[v]) for v in vars_ohe[:k])
                         + list(niveles_ohe[ref["variable"]]).index(ref["categoria"]))

    columnas = ColumnTransformer([
        ("num", Winsorizador(posiciones=posiciones_winsor, q=feat["winsorizar"]["percentil"]), numericas),
        ("ord", OrdinalEncoder(categories=[list(feat["ordinales"][c]) for c in ordinales]), ordinales),
        ("te", CodificadorObjetivoAgrupado(suavizado=te["suavizado"], n_folds=te["n_folds"], semilla=semilla),
         list(te["columnas"]) + [feat["id_grupo"]]),
        ("ohe", Pipeline([
            ("oh", OneHotEncoder(categories=[list(niveles_ohe[v]) for v in vars_ohe], sparse_output=False)),
            ("ref", QuitarColumna(indice=indice_referencia)),
        ]), vars_ohe),
    ], remainder="drop")

    return Pipeline([("pre", columnas), ("esc", StandardScaler())])