from pandas import DataFrame
import pandas as pd

import matplotlib.pyplot as plt
from sklearn import linear_model

data=pd.read_csv('house.csv')

column=['Id','MSSubClass','LotFrontage','LotArea','price'] 

df = DataFrame(data,columns=column).fillna(value=0)

X = df[['MSSubClass','LotFrontage','LotArea']] 

Y = df['price']

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

newMSSubClass=20
newLotFrontage=65
newLotArea=10000

print ('Predicted House Price: \n', regr.predict([[newMSSubClass,newLotFrontage,newLotArea]]))