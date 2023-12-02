"""
File: RandomForest.py
Author: Koi Stephanos
Date: 2023-12-02
Description: 
    This file contains the Random Forest regression model, extending the SklearnPredictor class.
    It is designed to predict NBA shooting percentages using extended statistical data.

Additional Notes:
    - Dependencies: scikit-learn for Random Forest implementation.

Modifications:
    - None

Copyright:
    © 2023 Koi Stephanos. All rights reserved.
    Unauthorized copying of this file, via any medium, is strictly prohibited.
    Proprietary and confidential.
"""

from sklearn.ensemble import RandomForestRegressor
from .SklearnPredictor import SklearnPredictor

class RandomForest(SklearnPredictor):
    def __init__(self, n_estimators=100, criterion="friedman_mse"):
        super().__init__(name="RandomForest")
        self.model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion)
