"""
File: LinearRegressionModel.py
Author: Koi Stephanos
Date: 2023-12-02
Description: 
    This file implements a Linear Regression model for predicting NBA shooting percentages.
    It extends the SklearnPredictor class and includes methods for model training and evaluation.

Additional Notes:
    - Dependencies: scikit-learn for Linear Regression implementation.

Modifications:
    - None

Copyright:
    © 2023 Koi Stephanos. All rights reserved.
    Unauthorized copying of this file, via any medium, is strictly prohibited.
    Proprietary and confidential.
"""

from sklearn.linear_model import LinearRegression as LinearRegressionModel
from .SklearnPredictor import SklearnPredictor

class LinearRegression(SklearnPredictor):
    def __init__(self):
        super().__init__(name="LinearRegression")
        self.model = LinearRegressionModel()