import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, learning_curve, validation_curve

class SklearnPredictor:
    def __init__(self, name):
        self.name = name
        self.min_max_scaler = preprocessing.MinMaxScaler()

    def get_model(self):
        return self.clf

    def set_data(self, df, target_col):
        self.X = df.drop(columns=[target_col])
        self.y = df[target_col]

    def get_data(self):
        return [self.X, self.y]

    def fit_and_predict(self, X_train, X_test, y_train):      
        X_train = self.min_max_scaler.fit_transform(X_train)
        X_test = self.min_max_scaler.transform(X_test)
        
        self.clf.fit(X_train, y_train)
        self.predictions = self.clf.predict(X_test)

    def split_test_data(self, test_size, is_fixed=False):
        if is_fixed:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, shuffle=True, random_state=42, test_size=test_size)
        else:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, shuffle=True, test_size=test_size)

        return X_train, X_test, y_train, y_test

    def get_avg_metrics_for_n_iterations(self, n_iterations, test_size, is_fixed=False):
        mse = 0
        r2 = 0

        for i in range(n_iterations):
            X_train, X_test, y_train, y_test = self.split_test_data(test_size, is_fixed)
            self.fit_and_predict(X_train, X_test, y_train)
            mse += mean_squared_error(y_test, self.predictions)
            r2 += r2_score(y_test, self.predictions)

        mse_avg = mse / n_iterations
        r2_avg = r2 / n_iterations

        return [round(mse_avg, 3), round(r2_avg, 3)]

    def get_learning_curve(self):
        train_sizes = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]

        train_sizes, train_scores, validation_scores = learning_curve(
            self.clf, self.X, self.y, train_sizes=train_sizes, cv=5, scoring='neg_mean_squared_error', shuffle=True)

        train_scores_mean = -train_scores.mean(axis=1)
        validation_scores_mean = -validation_scores.mean(axis=1)

        plt.style.use('seaborn')
        plt.plot(train_sizes, train_scores_mean, label='Training error')
        plt.plot(train_sizes, validation_scores_mean, label='Validation error')
        plt.ylabel('MSE', fontsize=14)
        plt.xlabel('Training set size', fontsize=14)
        plt.title(f'Learning curves for {self.name} model', fontsize=18, y=1.03)
        plt.legend()
        plt.show()

    # Optional: Include if you need model-specific parameter tuning
    def get_validation_curve(self, param_name, param_range):
        train_scores, test_scores = validation_curve(
            self.clf, self.X, self.y, param_name=param_name, param_range=param_range, scoring='neg_mean_squared_error', n_jobs=-1)

        train_scores_mean = -np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = -np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.title(f"Validation Curve with {self.name}")
        plt.xlabel(param_name)
        plt.ylabel("MSE")
        plt.plot(param_range, train_scores_mean, label="Training score", color="darkorange")
        plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, color="darkorange", alpha=0.2)
        plt.plot(param_range, test_scores_mean, label="Cross-validation score", color="navy")
        plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, color="navy", alpha=0.2)
        plt.legend(loc="best")
        plt.show()
