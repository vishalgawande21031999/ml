from pandas import DataFrame
import matplotlib.pyplot as plt

import pandas as pd
from sklearn import linear_model

linear={'x':[1,2,3,4,5,6],'y':[1,2,3,4,5,6]}

df=DataFrame(linear,columns=['x','y'])
print(df)
plt.scatter(df['x'], df['y'], color='red')
plt.xlabel('x', fontsize=14)

plt.ylabel('y', fontsize=14)
plt.show()
x=df[['x']]
y=df[['y']]
regr = linear_model.LinearRegression()
regr.fit(x, y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

New_x=8
print("new value is:",regr.predict([[x]]))
