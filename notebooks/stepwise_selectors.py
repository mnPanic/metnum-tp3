import metnum
import pandas as pd
import operator
import wrappers

from typing import List


def forward_stepwise_selection(
        df : pd.DataFrame,
        fs: List[str],
        e: str) -> List[str] :

    best_features = []
    while (len(best_features) < len(fs)) :
        remaining_features = list(set(fs) - set(best_features))
        scores = {}

        for f in remaining_features :
            tmp = best_features
            tmp.append(f)
            clf = wrappers.ProyectionRegression(
                features=tmp,
                explain=e
            )
            clf.fit(df)
            clf.predict(df)
        
            scores[f] = clf.score(df, "r2")
        best_features.append(max(scores.items(), key=operator.itemgetter(1))[0])

    predictor_score = {}
    pred_features =[]
    for f in best_features:
        pred_features.append(f)
        clf = wrappers.ProyectionRegression(
            features = pred_features,
            explain = e,
        )
        clf.fit(df)
        clf.predict(df)
        # Cambiar por k fold,
        # El mejor va a ser f 
        predictor_score[f] = clf.score(df, "r2")
    f = max(predictor_score.items(), key=operator.itemgetter(1))[0]
    res = []
    for k in best_features:
        if k != f:
            res.append(k)
        break
    res.append(f)
    return res