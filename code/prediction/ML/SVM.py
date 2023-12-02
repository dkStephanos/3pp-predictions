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
    def __init__(self, C, kernel="poly"):
        super().__init__(name="SVM")
        self.clf = svm.SVR(C=C, kernel=kernel)
