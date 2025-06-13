import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def getTrainSetAndTestSet(DataPath):
    """
    Processes the dataset, splitting it into training and testing sets.
    AT, V, AP, and RH are used as sample features, and PE as the sample output label.
    The dataset is split with a 3:1 ratio for training and testing.
    """
    data = pd.read_csv(DataPath)
    X = data[['AT', 'V', 'AP', 'RH']]  # AT, V, AP and RH are used as sample features. 
    y = data[['PE']]  # PE is used as sample output. 
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1) # Randomly split training and test sets, by default 25% of the dataset is used as the test set. 

    # Print the dimensions of the training and test sets
    print("Dimensions of X_train:", X_train.shape)
    print("Dimensions of X_test:", X_test.shape)
    print("Dimensions of y_train:", y_train.shape)
    print("Dimensions of y_test:", y_test.shape)

    return X_train, X_test, y_train, y_test

def TrainLinearRegression(X_train, y_train):
    """
    Establishes and trains a LinearRegression model.
    """
    linreg = LinearRegression() # Untrained machine learning model. 
    linreg.fit(X_train, y_train) # Train the model with input data x_train and output data y_train. 

    # Output the intercept and coefficients of the linear regression. 
    print("\nLinear Regression Intercept:", linreg.intercept_)
    print("Linear Regression Coefficients:", linreg.coef_)

    return linreg

def EvaluationModel(linreg, X_test, y_test):
    """
    Evaluates the model's performance by calculating Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
    between the true values and predicted values of the test set.
    """
    y_pred = linreg.predict(X_test)

    # Output Mean Squared Error (MSE)
    mse = metrics.mean_squared_error(y_test, y_pred)
    print("\nMean Squared Error (MSE):", mse)

    # Output Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    print("Root Mean Squared Error (RMSE):", rmse)

    return y_pred

def Visualization(y_test, y_pred):
    """
    Utilizes the matplotlib library to plot the relationship between predicted and actual values.
    The closer the predicted and actual values are to the black dashed line, the smaller the error.
    """
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    # The 'k--' means the line is black and dashed. 'lw' specifies line width. 
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=5)
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    plt.show()

if __name__ == "__main__":
    # Ensure you have 'Folds5x2_pp.csv' in the same directory or provide the full path.
    data_path = 'Folds5x2_pp.csv'

    # 1. Data preprocessing 
    X_train, X_test, y_train, y_test = getTrainSetAndTestSet(data_path)

    # 2. Train Linear Regression model 
    linreg_model = TrainLinearRegression(X_train, y_train)

    # 3. Use test set to evaluate model performance 
    y_pred = EvaluationModel(linreg_model, X_test, y_test)

    # 4. Visualization 
    Visualization(y_test, y_pred)