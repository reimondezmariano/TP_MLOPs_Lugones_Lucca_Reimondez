"""Regla basal de una sola variable (seccion 8.1 de la notebook).

Si el paciente tuvo al menos k internaciones en el anio previo, se predice que
reingresa. El corte k es el primer nivel cuya tasa de reingreso en train supera
la tasa global de train. No es un modelo: sirve de piso para comparar.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class ReglaBasal(BaseEstimator, ClassifierMixin):
    """Regla de decision sobre una variable ordinal, con la interfaz de scikit-learn.

    Args:
        variable: columna que usa la regla.
        niveles: niveles de la variable, de menor a mayor riesgo.
    """

    def __init__(self, variable="number_inpatient_disc", niveles=("0", "1", "2", "3 o mas")):
        self.variable = variable
        self.niveles = niveles

    def _codigos(self, X):
        codigo = {n: i for i, n in enumerate(self.niveles)}
        nivel = pd.Series(np.asarray(X[self.variable]).astype(object)).astype(str).map(codigo)
        if nivel.isna().any():
            raros = sorted(pd.Series(np.asarray(X[self.variable]))[nivel.isna()].astype(str).unique())
            raise ValueError(f"{self.variable} tiene niveles fuera de {list(self.niveles)}: {raros}")
        return nivel.to_numpy(dtype=int)

    def fit(self, X, y):
        nivel = self._codigos(X)
        y = np.asarray(y)
        self.classes_ = np.array([0, 1])
        self.tasa_global_ = float(y.mean())
        self.tasas_ = (pd.Series(y).groupby(nivel).mean()
                       .reindex(range(len(self.niveles))).to_numpy())
        de_riesgo = [i for i, t in enumerate(self.tasas_) if t > self.tasa_global_]
        # Sin ningun nivel por encima de la tasa global, la regla predice siempre 0.
        self.k_ = min(de_riesgo) if de_riesgo else len(self.niveles)
        return self

    def puntaje(self, X):
        """Nivel de cada paciente (0 a n-1): mas internaciones previas, mas riesgo."""
        return self._codigos(X)

    def predict(self, X):
        return (self._codigos(X) >= self.k_).astype(int)

    def predict_proba(self, X):
        # El nivel llevado a [0, 1]: conserva el orden, asi que ROC-AUC y AUC-PR
        # dan lo mismo que con el nivel.
        p = self._codigos(X) / (len(self.niveles) - 1)
        return np.column_stack([1 - p, p])
