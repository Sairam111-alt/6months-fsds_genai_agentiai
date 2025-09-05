import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"F:\Data Science\25th Aug\Practical\emp_sal.csv")

x = dataset.iloc[:, 1:2].values  
y = dataset.iloc[:, 2].values


#SVM Model
from sklearn.svm import SVR
svr_regressor=SVR(kernel='poly', degree=5,gamma='scale',C=10.0)
svr_regressor.fit(x,y)

svr_model_pred= svr_regressor.predict([[6.5]])
print(svr_model_pred)

# Knn Model
from sklearn.neighbors import KNeighborsRegressor
knn_reg_model=KNeighborsRegressor(n_neighbors=3, weights='distance')
knn_reg_model.fit(x,y)

knn_reg_pred=knn_reg_model.predict([[6.5]])
print(knn_reg_pred)