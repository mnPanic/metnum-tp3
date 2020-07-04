import metnum

from typing import List
from dataclasses import dataclass

@dataclass
class Ciudades():
    clfs: List[metnum.LinearRegressor]

    def features(self):
        pass

    def segmentar(self):
        pass

    def fit(self):
        # feature engineering
        # segmenta
        # for each segmento
        #   fit
        pass

    def predict(self):
        # feature(): feature engineering
        # segmenta
        # for each segmento
        #   fit
        # merge
        pass
