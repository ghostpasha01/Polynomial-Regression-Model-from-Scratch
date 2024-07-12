from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer,  ColumnTransformer
from sklearn.metrics import *
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler, PolynomialFeatures
import numpy as np
from ML_Pipeline.kuma_utils.preprocessing.imputer import LGBMImputer

class Regression:
    #Polynomial regression
    def polynomial_regression(self,x_train,x_test,y_train,y_test,degree):
        num_cols = x_train.select_dtypes(exclude=['object']).columns.tolist()
        # cat_cols = x_train.select_dtypes(include=['object']).columns.tolist()
        ct=ColumnTransformer([
            ('scale',MinMaxScaler(),num_cols)
        ])
        #building pipeline
        pipeline=Pipeline([
            ('ct',ct),
            ('poly',PolynomialFeatures(degree=degree)),
            ('model',LinearRegression())
        ])

        #fitting on train data
        pipeline.fit(x_train,y_train)

        #prediction on test data
        y_pred=pipeline.predict(x_test)
        return y_pred

    #Function for metrics of model
    def metrics(self,y_test,y_pred):
        print(f'R2 score:{r2_score(y_test,y_pred):.4f}')
        print(f'MSE:{mean_squared_error(y_test,y_pred):.4f}')
        print(f'RMSE:{np.sqrt(mean_squared_error(y_test,y_pred)):.4f}')
        print(f'MAE:{mean_absolute_error(y_test,y_pred):.4f}')
