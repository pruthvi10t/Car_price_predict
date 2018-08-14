import numpy as np
import pandas as pd
import requests
from io import StringIO
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('ibm.csv')

missing_data = df.isnull()

for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")

#Replacing the missing values with their respective column mean

average_1 = df["normalized-losses"].astype("float").mean(axis=0)
df["normalized-losses"].replace(np.nan,average_1,inplace=True)

average_2 = df['bore'].astype('float').mean(axis=0)
df['bore'].replace(np.nan,average_2,inplace= True)

average_3 = df['stroke'].astype('float').mean(axis=0)
df['stroke'].replace(np.nan,average_3,inplace=True)

average_4=df['horsepower'].astype('float').mean(axis=0)
df['horsepower'].replace(np.nan, average_4,inplace=True)

average_5=df['peak-rpm'].astype('float').mean(axis=0)
df['peak-rpm'].replace(np.nan, average_4,inplace=True)


df['num-of-doors'].value_counts().idxmax()

#replacing the missing values with fours doors as its the most frequent 
df["num-of-doors"].replace(np.nan,"four",inplace=True)

#droping rows of price with missing columns

df.dropna(subset=["price"],axis=0,inplace=True)
df.reset_index(drop=True,inplace=True)

#Analysing engine size as a predictor

plot1 = sns.regplot(x="engine-size",y="price",data=df).set_title("Engine Vs Price")
plt.ylim(0,)
plt.show(plot1)

print (df[["engine-size","price"]].corr())

#Analysing Highway-mpg as a predictor

plot2 = sns.regplot(x="highway-mpg",y="price",data=df).set_title("Mpg Vs Price")
plt.show(plot2)

print (df[["highway-mpg","price"]].corr())

#Analysing Peak-rpm as a predictor

plot3 = sns.regplot(x="peak-rpm",y="price",data=df).set_title("Rpm Vs Price")
plt.show(plot3)

print (df[["peak-rpm","price"]].corr())

plot4 = sns.boxplot(x="body-style", y="price", data=df)
plt.show(plot4)

#Analysing Other features using boxplot

sns.boxplot(x="body-style", y="price", data=df)
sns.boxplot(x="engine-location", y="price", data=df)
sns.boxplot(x="drive-wheels", y="price", data=df)

