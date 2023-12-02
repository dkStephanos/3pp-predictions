from sklearn.linear_model import LinearRegression
from .SklearnPredictor import SklearnPredictor

class LinearRegression(SklearnPredictor):
    def __init__(self):
        super().__init__(name="LinearRegression")
        self.clf = LinearRegression()