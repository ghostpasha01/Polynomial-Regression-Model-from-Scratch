import pandas as pd
from ML_Pipeline.Preprocessing import Preprocessing
from ML_Pipeline.Model import Regression
from ML_Pipeline.kuma_utils.preprocessing.imputer import LGBMImputer
import projectpro
projectpro.checkpoint('78ae27')
import warnings
warnings.filterwarnings("ignore")

#reading dataset
data=pd.read_csv('Input/NBA_Dataset_csv.csv')

#removing outliers
target_col='Points_Scored' #target column name
df=Preprocessing(data).remove_outlier(target_col)

#train test split of data
target_col='Points_Scored' #target column name
X_train,X_test,y_train,y_test=Preprocessing(df).split_data(target_col)

#onehot encoding of train and test data
X_train,X_test=Preprocessing(df).onehot_encode(X_train,X_test)

#lgbm imputer for missing value imputation
X_train,X_test=Preprocessing(df).lgbm_imputer(X_train,X_test)

#Polynomial regression model building
degree_of_poly=1 #degree of polynomial
y_pred=Regression().polynomial_regression(X_train,X_test,y_train,y_test,degree_of_poly)
projectpro.checkpoint('78ae27')
Regression().metrics(y_test,y_pred)
