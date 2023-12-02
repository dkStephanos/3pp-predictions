"""
File: NeuralNetwork.py
Author: Koi Stephanos
Date: 2023-12-02
Description: 
    Contains the implementation of a neural network model using scikit-learn's MLPRegressor.
    This file is part of an analytical toolkit for predicting basketball players' performance metrics.

Additional Notes:
    - Utilizes MLPRegressor for regression tasks.
    - Includes methods for preprocessing data, training the model, and evaluating performance.

Modifications:
    - None

Copyright:
    © 2023 Koi Stephanos. All rights reserved.
    Unauthorized copying of this file, via any medium, is strictly prohibited.
    Proprietary and confidential.
"""

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class NeuralNetwork:
    def __init__(self):
        pass
    
    def get_avg_metrics_for_n_iterations(self, data_frame, alpha_to_use, hidden_layer_size, activation_to_use, solver_to_use, test_size, target_col, n_iterations, is_fixed=False, print_results=True):
        """
        Runs the regression model n times and averages the performance metrics.

        Args:
            data_frame (pd.DataFrame): The DataFrame containing the dataset.
            alpha_to_use (float): The alpha parameter for regularization.
            hidden_layer_size (tuple): The size of the hidden layers.
            activation_to_use (str): The activation function to use.
            solver_to_use (str): The solver for optimization.
            test_size (float): The proportion of the dataset to include in the test split.
            target_col (str): The name of the target column.
            n_iterations (int): The number of iterations to run.
            is_fixed (bool): Whether to use a fixed seed for reproducibility.

        Returns:
            dict: A dictionary containing the averaged MSE and R2 scores.
        """
        mse_scores = []
        r2_scores = []

        for _ in range(n_iterations):
            results = self.regress(data_frame, alpha_to_use, hidden_layer_size, activation_to_use, solver_to_use, test_size, target_col, is_fixed=is_fixed)
            mse_scores.append(results[0])
            r2_scores.append(results[1])

        self.avg_mse = sum(mse_scores) / len(mse_scores)
        self.avg_r2 = sum(r2_scores) / len(r2_scores)

        if print_results:
            self.print_stats(alpha_to_use, solver_to_use, hidden_layer_size, activation_to_use, test_size, is_fixed)
    
    def regress(self, data_frame, alpha_to_use, hidden_layer_size, activation_to_use, solver_to_use, test_size, target_col, is_fixed=False):
        self.y_actual = data_frame[target_col]
        X_train, X_test, y_train, y_test = self.split_test_data(data_frame, test_size, target_col, is_fixed)      
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        mlp = MLPRegressor(activation=activation_to_use, hidden_layer_sizes=hidden_layer_size, solver=solver_to_use, alpha=alpha_to_use, learning_rate_init=.03, early_stopping=True, max_iter=1000)
        mlp_model = mlp.fit(X_train, y_train)                                                          
        self.mlp_predictions = mlp_model.predict(X_test)                                                    
        mse = metrics.mean_squared_error(y_test, self.mlp_predictions)
        r2 = metrics.r2_score(y_test, self.mlp_predictions)
        
        # After fitting the model, store the loss curve
        if hasattr(mlp_model, 'loss_curve_'):
            self.loss_curve = mlp_model.loss_curve_

        return [mse, r2]

    def split_test_data(self, data_frame, test_size, target_col, is_fixed=False):
        X = data_frame.drop(columns=[target_col])
        Y = data_frame[target_col]
        if is_fixed:    
            X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle=True, random_state=42, test_size=test_size)
        else:           
            X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle=True, test_size=test_size)

        return X_train, X_test, y_train, y_test

    def print_stats(self, alpha, solver_to_use, hidden_layer_size, activation, test_size, is_fixed):
        print(f"""The results for NN with settings:
        ------------------------------------------------
        Solver: {solver_to_use}
        Activation: {activation}
        Alpha: {str(alpha)}
        Hidden Layer Dimensions: {str(hidden_layer_size)}
        Fixed seed: {str(is_fixed)}
        Test Set Percentage: {str(test_size)}
        ------------------------------------------------
        
        MSE: {self.avg_mse}
        R2 Score: {self.avg_r2}
        """)

    def plot_loss_curve(self):
        """
        Plots the loss curve of the neural network model.
        """
        if hasattr(self, 'loss_curve'):
            plt.figure(figsize=(8, 6))
            plt.plot(self.loss_curve, label='Loss Curve')
            plt.title('Loss Curve of Neural Network')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print("No loss curve data available. Train the model first.")
            
    def plot_actual_vs_predicted(self):
        """
        Plots actual vs. predicted values to evaluate the performance of the neural network model.
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(self.y_actual, self.mlp_predictions, alpha=0.5)
        plt.title('Actual vs. Predicted Values')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.plot([self.y_actual.min(), self.y_actual.max()], [self.y_actual.min(), self.y_actual.max()], 'k--') # Ideal line
        plt.grid(True)
        plt.show()
