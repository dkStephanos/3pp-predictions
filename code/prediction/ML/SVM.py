"""
File: SVM.py
Author: Koi Stephanos
Date: 2023-12-02
Description: 
    This file includes the implementation of the Support Vector Machine (SVM) model for regression.
    It is part of a project to utilize machine learning for analyzing NBA shooting statistics.

Additional Notes:
    - Dependencies: scikit-learn for the SVM implementation.

Modifications:
    - None

Copyright:
    © 2023 Koi Stephanos. All rights reserved.
    Unauthorized copying of this file, via any medium, is strictly prohibited.
    Proprietary and confidential.
"""

from sklearn import svm
from .SklearnPredictor import SklearnPredictor

class SVM(SklearnPredictor):
    def __init__(self, kernel="poly", degree=3, gamma='scale', C=1.0, epsilon=.1):
        super().__init__(name="SVM")
        self.model = svm.SVR(kernel=kernel, degree=degree, gamma=gamma, C=C, epsilon=epsilon)
