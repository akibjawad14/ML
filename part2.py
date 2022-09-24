from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import sys
import seaborn as sns
# import driver
# import part2
# import part1
# from sklearn import linear_model
# from sklearn.metrics import mean_squared_error, r2_score
# import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
import requests
import io
from sklearn.preprocessing import MinMaxScaler


def library_reg(X_train, y_train, X_test, y_test):
    mm = MinMaxScaler()

    # X = df_transformed[["age", "distance_MRT", "num_stores"]]
    # Y = df_transformed["price"]

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    print("The respective attribute weights are: ", lr.coef_)
    print("The intercept is: ", round(lr.intercept_, 8))

    y_train_predict = lr.predict(X_train)
    train_rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
    train_r2 = r2_score(y_train, y_train_predict)
    print("Training R2: ", round(train_r2, 5))
    print('Training RMSE: ', round(train_rmse, 5))

    y_test_predict = lr.predict(X_test)
    test_rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
    test_r2 = r2_score(y_test, y_test_predict)
    print('\n\nP2 -- Testing')
    print("Test R2: ", round(test_r2, 5))
    print("Test RMSE: ", round(test_rmse, 5), '\n')
