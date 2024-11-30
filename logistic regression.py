import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import random as rn

# Load dataset
data = pd.read_csv(r'E:\CSE1021 - Introduction to Problem Solving and Programming\Heart_Disease_Prediction.csv')

# Convert 'Heart Disease' column to binary
data['Heart Disease'] = data['Heart Disease'].map({'Presence': 1, 'Absence': 0})

# Print dataset columns
print(data.columns)

# Define features (independent variables) and target (dependent variable)
X = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Max HR', 'ST depression']]
Y = data['Heart Disease']

# Split the dataset into training and testing sets
rn.seed(1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=1)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Make predictions
Y_pred = model.predict(X_test)

# Print accuracy score
print(f"Accuracy: {accuracy_score(Y_test, Y_pred)}")


