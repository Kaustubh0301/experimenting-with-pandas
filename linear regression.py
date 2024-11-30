import pandas as pd 
from sklearn.metrics import accuracy_score
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn import preprocessing , svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import random as rn 
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv(r'E:\CSE1021 - Introduction to Problem Solving and Programming\Heart_Disease_Prediction.csv')
# Print dataset columns
print(data.columns)

# Convert 'Heart Disease' column to binary
data['Heart Disease'] = data['Heart Disease'].map({'Presence': 1, 'Absence': 0})

# Define features and binarize target variable
X= data[['Age', 'Sex', 'BP', 'Cholesterol', 'Max HR', 'ST depression']]
Y = data['Heart Disease']

print(X)
print(Y)
sns.scatterplot(data=data)
plt.plot(X,Y)
plt.show()
rn.seed(1)
X_train , X_test , Y_train , Y_test = train_test_split(X,Y, test_size=.30) 

model = LinearRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
Y_pred_rounded = np.round(Y_pred)

accuracy = accuracy_score(Y_test, Y_pred_rounded)
print(f"Accuracy Score: {accuracy * 100:.2f}%")

