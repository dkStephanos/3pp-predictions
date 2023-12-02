from sklearn import svm
from .SklearnPredictor import SklearnPredictor

class SVM(SklearnPredictor):
    def __init__(self, C, kernel="poly"):
        super().__init__(name="SVM")
        self.clf = svm.SVR(C=C, kernel=kernel)
