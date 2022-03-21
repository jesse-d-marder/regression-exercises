import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression 

def plot_residuals(y, yhat):
    """ Plots residuals vs y """
    residuals = yhat - y
    
    plt.scatter(x=y, y = residuals)
    plt.axhline(0)
    plt.xlabel('x')
    plt.ylabel('residual (yhat - y)')
    
def regression_erros