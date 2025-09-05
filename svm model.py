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