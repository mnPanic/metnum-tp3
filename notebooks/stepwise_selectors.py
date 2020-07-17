import metnum
import pandas as pd
import operator
import wrappers
import cross_validation as cv

from typing import Callable
from typing import List
from typing import Dict

FuncBuilder = Callable[[List[str], str, str], wrappers.RegressionWrapper]

def func_builder_projection_sbc(
        fs: List[str],
        e: str,
        seg_col: str,
    ) -> wrappers.RegressionWrapper:
    return wrappers.ProjectionRegression(
        features = fs,
        explain = e,
        func_segment= lambda df: wrappers.segment_by_col(df, seg_col),
    )

def fb_poly_linear_seg_by_col(
        fs: List[str],
        e: str,
        seg_col: str,
    ) -> wrappers.RegressionWrapper:
    return wrappers.PolynomialRegressor(
        features = fs,
        degree = 1,
        explain = e,
        func_segment= lambda df: wrappers.segment_by_col(df, seg_col),
    )

def func_builder_polynomial_d3_sbc(
        fs: List[str],
        e: str,
        seg_col: str,
    ) -> wrappers.RegressionWrapper:
    return wrappers.PolynomialRegressor(
        features = fs,
        degree = 3,
        explain = e,
        func_segment= lambda df: wrappers.segment_by_col(df, seg_col),
    )

def func_builder_polynomial_d7_sbc(
        fs: List[str],
        e: str,
        seg_col: str,
    ) -> wrappers.RegressionWrapper:
    return wrappers.PolynomialRegressor(
        features = fs,
        degree = 7,
        explain = e,
        func_segment= lambda df: wrappers.segment_by_col(df, seg_col),
    )

def forward_stepwise_selection(
        func_builder: FuncBuilder,
        data: pd.DataFrame,
        fs: List[str],
        e: str,
        seg_col: str,
        K: int,
    ) -> (List[str], Dict[int, float]):

    best_fs = [[]]
    while len(fs) > 0:
        best_score = 0
        best_scoring_feature = ""
        first_check = True
        
        for f in fs:
            tmp_fs = best_fs[-1][:]
            tmp_fs.append(f)
            clf = func_builder(tmp_fs, e, seg_col)
            clf.fit(data)

            feature_score, _ = clf.score(data, "r2")

            if first_check or feature_score > best_score:
                best_score = feature_score
                best_scoring_feature = f
                first_check = False

        best_fs.append(best_fs[-1] + [best_scoring_feature])
        fs.remove(best_scoring_feature)
    
    scores_by_step = {}
    res_index = 0
    best_score = 0
    first_check = True
    for i in range(len(best_fs)):
        clf = func_builder(best_fs[i], e, seg_col)

        r2_scores, _ = cv.cross_validate(clf, data, ["r2"], K)
        tmp_r2_score = r2_scores["r2"]

        scores_by_step[i] = tmp_r2_score

        if first_check or tmp_r2_score > best_score:
            best_score = tmp_r2_score
            res_index = i
            first_check = False
        
    return best_fs[res_index], scores_by_step

# def forward_stepwise_selection(
#         df : pd.DataFrame,
#         fs: List[str],
#         e: str,
#         K: int) -> List[str] :

#     best_features = []
#     while (len(best_features) < len(fs)) :
#         remaining_features = list(set(fs) - set(best_features))
#         scores = {}

#         for f in remaining_features :
#             tmp = best_features
#             tmp.append(f) # {x_1, x_2, a}
#             clf = wrappers.ProyectionRegression(
#                 features=tmp,
#                 explain=e
#             )
#             clf.fit(df)
#             clf.predict(df)
        
#             scores[f] = clf.score(df, "r2")
#         best_features.append(max(scores.items(), key=operator.itemgetter(1))[0])

#     predictor_score = {}
#     pred_features = []
#     for f in best_features:
#         pred_features.append(f)
#         clf = wrappers.ProyectionRegression(
#             features = pred_features,
#             explain = e,
#         )
#         predictor_score[f] = cv.cross_validate(clf, df, "r2", K)

#     f = max(predictor_score.items(), key=operator.itemgetter(1))[0]
#     res = []
#     for k in best_features:
#         if k != f:
#             res.append(k)
#         break
#     res.append(f)
#     return res