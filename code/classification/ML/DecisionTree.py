from sklearn.tree import DecisionTreeClassifier
from .SklearnPredictor import SklearnPredictor

class DecisionTree(SklearnPredictor):
    def __init__(self,criterion):
        super().__init__(name="DecisionTree")
        self.model = DecisionTreeClassifier(criterion=criterion)