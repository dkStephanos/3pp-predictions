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
    
    def regress(self, dataFrame, alphaToUse, hiddenLayerSize, activationToUse, solverToUse, test_size, target_col, is_fixed=False, print_results=True):
        self.y_actual = dataFrame[target_col]
        X_train, X_test, y_train, y_test = self.split_test_data(dataFrame, test_size, target_col, is_fixed)      
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        mlp = MLPRegressor(activation=activationToUse, hidden_layer_sizes=hiddenLayerSize, solver=solverToUse, alpha=alphaToUse, learning_rate_init=.03, early_stopping=True, max_iter=1000)
        mlp_model = mlp.fit(X_train, y_train)                                                          
        self.mlp_predictions = mlp_model.predict(X_test)                                                    
        mse = metrics.mean_squared_error(y_test, self.mlp_predictions)
        r2 = metrics.r2_score(y_test, self.mlp_predictions)
        
        # After fitting the model, store the loss curve
        self.loss_curve = mlp_model.loss_curve_

        if(print_results):
            self.print_stats(alphaToUse, solverToUse, hiddenLayerSize, activationToUse, test_size, is_fixed, y_test, self.mlp_predictions)

        return [mse, r2]

    def split_test_data(self, dataFrame, test_size, target_col, is_fixed=False):
        X = dataFrame.drop(columns=[target_col])
        Y = dataFrame[target_col]
        if is_fixed:    
            X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle=True, random_state=42, test_size=test_size)
        else:           
            X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle=True, test_size=test_size)

        return X_train, X_test, y_train, y_test

    def print_stats(self, Alpha, solverToUse, hiddenLayerSize, activation, test_size, is_fixed, y_test, mlp_predictions):
        print(f"""The results for NN with settings:
                Solver: {solverToUse}
                Activation: {activation}
                Alpha: {str(Alpha)}
                Hidden Layer Dimensions: {str(hiddenLayerSize)}
                Fixed seed: {str(is_fixed)}
                Test Set Percentage: {str(test_size)}
                
                MSE: {metrics.mean_squared_error(y_test, mlp_predictions)}
                R2 Score: {metrics.r2_score(y_test, mlp_predictions)}
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

        Args:
            y_actual (array-like): The actual target values.
            y_predicted (array-like): The predicted target values by the model.
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(self.y_actual, self.mlp_predictions, alpha=0.5)
        plt.title('Actual vs. Predicted Values')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.plot([self.y_actual.min(), self.y_actual.max()], [self.y_actual.min(), self.y_actual.max()], 'k--') # Ideal line
        plt.grid(True)
        plt.show()