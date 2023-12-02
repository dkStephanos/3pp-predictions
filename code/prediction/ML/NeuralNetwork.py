from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class NeuralNetwork:
    @staticmethod
    def regress(dataFrame, alphaToUse, hiddenLayerSize, activationToUse, solverToUse, test_size, target_col, isFixed=False, printResults=True):
        X_train, X_test, y_train, y_test = NeuralNetwork.splitTestData(dataFrame, test_size, target_col, isFixed)      
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        mlp = MLPRegressor(activation=activationToUse, hidden_layer_sizes=hiddenLayerSize, solver=solverToUse, alpha=alphaToUse, learning_rate_init=.03, early_stopping=True, max_iter=1000)
        mlp_model = mlp.fit(X_train, y_train)                                                          
        mlp_predictions = mlp_model.predict(X_test)                                                    
        mse = metrics.mean_squared_error(y_test, mlp_predictions)
        r2 = metrics.r2_score(y_test, mlp_predictions)

        if(printResults):
            NeuralNetwork.printStats(alphaToUse, solverToUse, hiddenLayerSize, activationToUse, test_size, isFixed, y_test, mlp_predictions)

        return [mse, r2]

    @staticmethod
    def splitTestData(dataFrame, test_size, target_col, isFixed=False):
        X = dataFrame.drop(columns=[target_col])
        Y = dataFrame[target_col]
        if isFixed:    
            X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle=True, random_state=42, test_size=test_size)
        else:           
            X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle=True, test_size=test_size)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def printStats(Alpha, solverToUse, hiddenLayerSize, activation, test_size, isFixed, y_test, mlp_predictions):
        print("\t The results for NN with settings: ")
        print("\t Solver: " + solverToUse)
        print("\t Activation: " + activation)
        print("\t Alpha: " + str(Alpha))
        print("\t Hidden Layer Dimensions: " + str(hiddenLayerSize))
        print("\t Fixed seed: " + str(isFixed))
        print("\t Test Set Percentage: " + str(test_size))
        print("\n\t are as follows: ")

        print("\n    MSE: ", metrics.mean_squared_error(y_test, mlp_predictions))
        print("\n    R2 Score: ", metrics.r2_score(y_test, mlp_predictions))
