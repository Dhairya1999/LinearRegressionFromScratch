import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import OrdinalEncoder


data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00640/Occupancy_Estimation.csv')
print(data.isna().sum())
print(data.duplicated().sum())
print(data.select_dtypes(include=['object']).columns.tolist())

fig = plt.figure(figsize = (12,10))
plt.title('Correlations')
cor = data.corr()
sns.heatmap(cor, annot=True, cmap = plt.cm.CMRmap_r)
plt.show()

cor_feat = set()
for i in range(len(cor.columns)):
    for j in range(i):
        if abs(cor.iloc[i, j]) > 0.8:
            colname = cor.columns[i]
            cor_feat.add(colname)

print(cor_feat)

print(data)
del data['Date']
del data['Time']
del data['S3_Light']
del data['S4_Temp']
print(data)


sc = StandardScaler()
cols = data.columns
data = sc.fit_transform(data)
data = pd.DataFrame(data, columns= cols)
print(data)


X = data.iloc[:, :-1]
y = data.iloc[:,-1]
print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 1000)

print(X_train)
print(y_train)