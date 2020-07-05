import metnum

import pandas as pd
import numpy as np

from typing import List
from typing import Callable

Matrix = np.array
Vector = np.array

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

def _default_get_b(df: pd.DataFrame) -> Vector:
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
    FuncGetB = Callable[[pd.DataFrame], Vector]
    FuncSegment = Callable[[pd.DataFrame], List[pd.DataFrame]]
    FuncAddFeatures = Callable[[pd.DataFrame], None]

    def __init__(
            self,

            func_get_A: FuncGetA = _default_get_A, 
            func_get_b: FuncGetB = _default_get_b,
            func_segment: FuncSegment = _default_segment,
            func_add_features: FuncAddFeatures = _default_add_features,

            # nombre de la columna en la que van las predicciones
            predict_col: str = "prediction",
        ):

        self._predict_col = predict_col

        self._get_A = func_get_A
        self._get_b = func_get_b
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

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Le agrega al dataframe una columna "prediction" que contiene 
        las predicciones dado lo entrenado.
        """

        self._add_features(df)
        segments = self._segment(df)

        # Segmento el dataframe y hago el predict por cada uno
        for i, segment in enumerate(segments):
            A = self._get_A(segment)
            segment[self._predict_col] = self._clfs[i].predict(A)

        # Concateno los segmentos manteniendo el orden original
        return pd.concat(segments)

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
            explain: str,
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

        def get_b(df: pd.DataFrame) -> Vector:
            return df[explain].values.reshape(-1, 1)

        def get_A(df: pd.DataFrame) -> Matrix:
            return np.stack(
                [df[feature].values for feature in features], 
                axis=-1,    # para que los stackee como columnas
            )

        super().__init__(
            func_get_b=get_b,
            func_get_A=get_A,
            **kwargs
        )


class PolynomialRegressor(RegressionWrapper):
    """
    Regresor polinomial del grado especificado, de una sola feature.
    """

    def __init__(
            self, 
            feature: str,
            explain: str,
            degree: int,
            **kwargs,
        ):

        def get_b(df: pd.DataFrame) -> Vector:
            return df[explain].values.reshape(-1, 1)

        def get_A(df: pd.DataFrame) -> Matrix:
            return np.stack(
                [df[feature].values**n for n in range(0, degree+1)], 
                axis=-1,    # para que los stackee como columnas
            )

        super().__init__(
            func_get_b=get_b,
            func_get_A=get_A,
            **kwargs
        )

    def get_x(self, i: int):
        return self._clfs[i].get_x()
