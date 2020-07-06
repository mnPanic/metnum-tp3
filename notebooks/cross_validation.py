import wrappers
import statistics

import numpy as np
import pandas as pd

from typing import Dict
from typing import Callable

ScoringFunc = Callable[[np.ndarray, np.ndarray], float]

def cross_validate(
        clf: wrappers.RegressionWrapper,
        df: pd.DataFrame,
        scoring: str,
        K: int,
        debug=False, 
        **kwargs
    ) -> float:
    """"
    Hace K fold del classifier `clf`, llamando a la funcion de scoring
    para calcular los scores de cada fold.
    """

    scores = []
    
    set_size = int(len(df.index)/K)
    
    for i in range(K):
        # Particionar
        l_bound = set_size * i
        r_bound = set_size * (i+1)
        
        df_train = pd.concat([df[:l_bound], df[r_bound:]])
        df_val = df[l_bound:r_bound]

        # Entrenar
        clf.fit(df_train)

        # Scoring
        scores.append(clf.score(df_val, scoring))
    
    if debug: print("scores:", scores)

    return statistics.mean(scores)
