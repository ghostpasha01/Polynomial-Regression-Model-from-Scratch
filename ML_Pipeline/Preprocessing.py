from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from category_encoders import OneHotEncoder
from ML_Pipeline.kuma_utils.preprocessing.imputer import LGBMImputer
import pandas as pd


class Preprocessing:
    def __init__(self,data):
        self.data=data

    #function to remove outliers
    def remove_outlier(self,target):
        cols=self.data.drop(target,axis=1).select_dtypes(exclude=['object']).columns.tolist()
        Q1 = self.data[cols].quantile(0.25)
        Q3 = self.data[cols].quantile(0.75)
        IQR = Q3 - Q1
        self.data =self.data[~((self.data[cols] < (Q1 - 1.5 * IQR)) | (self.data[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
        print('Outliers removed..')
        return self.data

    #function for one hot encoding
    def onehot_encode(self,x_train,x_test):
        cat_cols=x_train.select_dtypes(include=['object']).columns.tolist()
        one=OneHotEncoder(cols=cat_cols,return_df=True,use_cat_names=True)
        x_train=one.fit_transform(x_train)
        x_test=one.transform(x_test)
        print('Onehot encoding done..')
        return x_train,x_test

    #function for lgbm imputation
    def lgbm_imputer(self,x_train,x_test):
        imputer=LGBMImputer(n_iter=15,verbose=True)
        x_train=imputer.fit_transform(x_train)
        x_test=imputer.transform(x_test)
        x_train=pd.DataFrame(x_train,columns=x_train.columns)
        x_test=pd.DataFrame(x_test,columns=x_test.columns)
        print('Missing values imputed..')
        return x_train,x_test

    # splitting data.
    def split_data(self, target_col):
        X=self.data.drop(target_col,axis=1)
        Y=self.data[target_col]
        # split a dataset into train and test sets
        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        return X_train, X_test,y_train,y_test


