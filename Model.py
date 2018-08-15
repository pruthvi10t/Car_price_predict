import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')

y_data = df['price']
x_data = df.drop('price',axis=1)

lr = LinearRegression()


def distributionPlot(redfunction,bluefunction,redname,bluename,Title):
    width = 12
    height = 10
    plt.figure(figsize=(width,height))

    axis_1 = sns.distplot(redfunction,hist=False,color="r",label=redname)
    axis_2 = sns.distplot(bluefunction,hist=False,color="b",label=bluename,ax=axis_1)
    
    plt.title(Title)
    plt.xlabel('Price in Dollars ')
    plt.ylabel('Proportion in Cars')

    plt.show()
    plt.close()

x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.8,random_state=1)

print("Number of Test Samples : ",x_test.shape[0])
print("Number of Training Samples : ",x_train.shape[0])

lr.fit(x_train[['horsepower','curb-weight','engine-size','highway-mpg']],y_train)
y_pred = lr.predict(x_test[['horsepower','curb-weight','engine-size','highway-mpg']])

y_test = lr.predict(x_test[['horsepower','curb-weight','engine-size','highway-mpg']])

Title = 'Distributed Plot of Predicted Value using Training Data vs Training Data Distribution'
distributionPlot(y_train,y_pred,"Precited Values","Actual Values",Title)

Title = 'Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution'
distributionPlot(y_train,y_test,"Actual Values","Predicted Values",Title)
