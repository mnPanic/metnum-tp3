import metnum
import pandas as pd
import numpy as np

import sklearn.metrics

from typing import List
from typing import Dict
from typing import Callable

Matrix = np.array
Vector = np.array

# Utils
def segment_by_col(df: pd.DataFrame, col: str) -> List[pd.DataFrame]:
    values = df[col].dropna().unique()
    return [ df[ df[col]==value ].copy() for value in values ]

# Metricas
def rmse(y_true, y_pred):
    return sklearn.metrics.mean_squared_error(y_true, y_pred, squared=False)

def rmsle(y_true, y_pred):
    return np.sqrt(sklearn.metrics.mean_squared_log_error(y_true, y_pred))

# Funciones default
def _default_add_features(df: pd.DataFrame):
    """Agrega features modifcando el dataframe por referencia"""
    pass

def _default_segment(df: pd.DataFrame) -> List[pd.DataFrame]:
    """Devuelve una lista de segmentos del DataFrame"""

    # devuelve un unico segmento con el dataframe entero,
    # equivalente a no segmentar nada.
    return [df]

def _default_get_A(df: pd.DataFrame) -> Matrix:
    """
    Convierte el dataframe a la matriz de variables independientes
    de CML (i.e, A de Ax = b). Es la matriz que usualmente va a
    contener en sus columnas la aplicacion de las familias de
    funciones sobre los datos.

        A = f1(x_1, ..., w_1),  f2(...), ..., fn(...)
            f1(x_2, ..., w_2),  f2(...), ..., fn(...)
            .
            .
            .
            f1(x_n, ..., w_n),  f2(...), ..., fn(...)

    """

    # si o si se tiene que implementar
    raise NotImplementedError


class RegressionWrapper():
    """
    Un RegressionWrapper se encarga de representar el uso que le vamos
    a dar al LinearRegressor, previamente agregando features (feature
    engineering), segmentando el dataset y aplicando la regresion
    sobre cada segmento, para luego mergearlo y retornarlo mateniendo
    el orden original.

    La idea es que este sea una clase base, y que podamos hacer mas
    regresores simplemente subclasseando este, e implementando
    `_segment`, `_create_features`, `_get_dependent` y 
    `_get_independent`.
    """

    FuncGetA = Callable[[pd.DataFrame], Matrix]
    FuncSegment = Callable[[pd.DataFrame], List[pd.DataFrame]]
    FuncAddFeatures = Callable[[pd.DataFrame], None]

    def __init__(
            self,

            # columna a explicar, con ella se construye el vector b
            explain: str, # ex. precios

            func_get_A: FuncGetA = _default_get_A, 
            func_segment: FuncSegment = _default_segment,
            func_add_features: FuncAddFeatures = _default_add_features,

            # nombre de la columna en la que van las predicciones
            predict_col: str = "prediction",
        ):

        self._predict_col = predict_col
        self._explain_col = explain

        self._get_A = func_get_A
        self._segment = func_segment
        self._add_features = func_add_features

        # lista de clasificadores, hay uno por segmento
        self._clfs: List[metnum.LinearRegression] = []

    def fit(self, df: pd.DataFrame):
        self._add_features(df)
        segments = self._segment(df)

        for segment in segments:
            clf = metnum.LinearRegression()
            A = self._get_A(segment)
            b = self._get_b(segment)
            
            clf.fit(A, b)

            self._clfs.append(clf)

    def predict(self, df_orig: pd.DataFrame) -> pd.DataFrame:
        """
        Le agrega al dataframe una columna "prediction" que contiene 
        las predicciones dado lo entrenado.
        """
        df = df_orig.copy()

        self._add_features(df)
        segments = self._segment(df)

        # Segmento el dataframe y hago el predict por cada uno
        for i, segment in enumerate(segments):
            A = self._get_A(segment)
            segment[self._predict_col] = self._clfs[i].predict(A)

        # Concateno los segmentos manteniendo el orden original
        return pd.concat(segments).sort_index()

    def _get_b(self, df: pd.DataFrame) -> Vector:
        """
        Obtiene del dataframe el vector de variables dependientes.
        (i.e b de Ax = b)

            b = y_1     # feature dependiente 1
                y_2     # feature dependiente 2
                .
                .
                .
                y_n     # feature dependiente n

        """
        return df[self._explain_col].values.reshape(-1, 1)

    def get_x(self, i: int = None) -> List[List[float]]:
        if i is None:
            return [self._clfs[k].get_x() for k in range(len(self._clfs))]
        return [self._clfs[i].get_x()]
    
    def score(self, df: pd.DataFrame, kind: str) -> (float, pd.DataFrame):
        """Shorthand para calcular un solo score"""
        score, _ =  self.scores(df, [kind])
        return score

    def scores(
            self,
            df: pd.DataFrame,
            kinds: List[str]=None,
        ) -> (Dict[str, float], pd.DataFrame):
        """
        Scores es un wrapper de predict que calcula los scores.

        Ejemplo de uso

            >>> clf = RegressionWrapper(...)
            >>> clf.fit(df)

            >>> clf.scores(df, ["r2", "rmse"])
            {"r2": 0.5, "rmse": 0.34}

        """

        SCORES = {
            # kind : scoring fn
            # https://scikit-learn.org/stable/modules/classes.html#regression-metrics
            "r2":    sklearn.metrics.r2_score,
            "rmse":  rmse,
            "rmsle": rmsle,
        }

        pred = self.predict(df)
        y_pred = pred[self._predict_col].values

        # Filtramos predicciones negativas.
        negativos = np.argwhere(y_pred < 0)
        y_pred = np.delete(y_pred, negativos)
        y_true = np.delete(pred[self._explain_col].values, negativos)

        if kinds is None:
            kinds = SCORES.keys()

        scores = dict.fromkeys(kinds)
        for kind in kinds:
            scores[kind] = SCORES[kind](y_true, y_pred)
        
        return scores

class ProyectionRegression(RegressionWrapper):
    """
    ProyectionRegression implementa un regresor que toma como familia
    de funciones las proyecciones de las features. Esto resulta en
    una matriz A que tiene la siguiente pinta,

        A = x_1     ...     w_1
            x_2     ...     w_2
            .
            .
            .
            x_n     ...     w_n
    
    Donde x, ..., w son las features elegidas.
    No realiza ninguna segmentacion.
    """

    def __init__(
            self, 
            features: List[str],
            **kwargs,
        ):
        """
        Construye un ProyectionRegression con las features (variables
        independientes) que explican a la independiente.

        Los kwargs adicionales son pasados al constructor de 
        RegressionWrapper.

        Ejemplo de uso
        
            clf = ProyectionRegression(
                features=['metroscubiertos'], 
                explain='precio',
            )

            clf.fit(df)
            result = clf.predict(df)
            # ahora result tiene una columna 'prediction'
            # con las predicciones realizadas

        """

        def get_A(df: pd.DataFrame) -> Matrix:
            return np.stack(
                [df[feature].values for feature in features], 
                axis=-1,    # para que los stackee como columnas
            )

        super().__init__(
            func_get_A=get_A,
            **kwargs
        )


class PolynomialRegressor(RegressionWrapper):
    """
    Regresor polinomial del grado especificado, para multiples features.
    Usa la familia de funciones

        { F_ij(X) = (x_i)**(j-1) :  1 <= i <= n
                                    ^ 1 <= j <= d } 
    
    Donde d es el grado del polinomio, y n el número de features.
    Sean X_1, ... , X_m, esto resulta en una matriz A de la pinta

            F_11(X_1) F_12(X_1) ... F_1d(X_1) F_21(X_1) F_21(X_1) ... F_nd(X_1)
            F_11(X_2) F_12(X_2) ... F_1d(X_2) F_21(X_2) F_21(X_2) ... F_nd(X_2)
            .                   ...                               ...
        A=  .                   ...                               ...
            .                   ...                               ...
            F_11(X_m) F_12(X_m) ... F_1d(X_m) F_21(X_m) F_21(X_m) ... F_nd(X_m)

        A \in R^{m x (n * d)}
    """

    def __init__(
            self, 
            features: List[str],
            degree: int,
            **kwargs,
        ):

        def get_A(df: pd.DataFrame) -> Matrix:
            cols = []
            for f in features:
                cols.extend(
                    [df[f].values**n for n in range(0, degree+1)]
                )
            return np.stack(cols, axis=-1)

        super().__init__(
            func_get_A=get_A,
            **kwargs
        )
