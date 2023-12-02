"""
File: DecisionTree.py
Author: Koi Stephanos
Date: 2023-12-02
Description: 
    This file contains the Decision Tree regression model, extending the SklearnPredictor class.
    It is designed to predict NBA shooting percentages using extended statistical data.

Additional Notes:
    - Dependencies: scikit-learn for Decision Tree implementation.

Modifications:
    - None

Copyright:
    © 2023 Koi Stephanos. All rights reserved.
    Unauthorized copying of this file, via any medium, is strictly prohibited.
    Proprietary and confidential.
"""

from sklearn.tree import DecisionTreeClassifier
from .SklearnPredictor import SklearnPredictor

class DecisionTree(SklearnPredictor):
    def __init__(self, criterion):
        super().__init__(name="DecisionTree")
        self.model = DecisionTreeClassifier(criterion=criterion)