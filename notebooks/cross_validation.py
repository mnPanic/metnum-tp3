import wrappers
import statistics

import numpy as np
import pandas as pd

from typing import Dict
from typing import List

def cross_validate(
        clf: wrappers.RegressionWrapper,
        df: pd.DataFrame,
        scorings: List[str],
        K: int,
        debug=False, 
        **kwargs
    ) -> (Dict[str, float], pd.DataFrame):
    """
    Hace K fold del classifier `clf`, para cada score.
    Devuelve un dict scoring:mean_score y ademas el dataframe con todos los
    resultados.
    """

    scores = {k: [] for k in scorings}
    dfs = []

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
        res, df_pred = clf.scores(df_val, scorings)
        dfs.append(df_pred)
        for scoring, value in res.items():
            scores[scoring].append(value)
    
    if debug: print("scores:", scores)

    avgs = dict.fromkeys(scorings, 0.0)
    for k in avgs.keys():
        avgs[k] = statistics.mean(scores[k])

    return avgs, pd.concat(dfs)
