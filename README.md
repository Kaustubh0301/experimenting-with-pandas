Report: 
Heart Disease Prediction Analysis


Objective:
The purpose of this analysis is to predict heart disease presence (Presence or Absence) using various machine learning models, including Linear Regression, Logistic Regression, and K-Nearest Neighbors (KNN). The data is sourced from the provided dataset and contains clinical parameters relevant to heart disease prediction.


Dataset Summary:
The dataset comprises 270 records with 14 attributes, including:
•	Age, Sex, BP (Blood Pressure), Cholesterol, Max HR (Maximum Heart Rate), ST depression, etc.
•	Heart Disease is the target variable, which has been converted into binary: 1 for Presence and 0 for Absence.


Features Used:
For modeling, the following features were selected:
•	Age
•	Sex
•	BP
•	Cholesterol
•	Max HR
•	ST depression


The target variable was:
•	Heart Disease (Binary: 1 for Presence, 0 for Absence)

Models Used

Linear Regression
A regression algorithm to predict a continuous output, later rounded for binary classification.
Evaluation metric: Accuracy.

Logistic Regression
A classification algorithm directly suited for binary outputs.
Evaluation metric: Accuracy.

K-Nearest Neighbors (KNN)
A non-parametric classification algorithm based on proximity.
Evaluation metric: Accuracy.

Results:

Linear Regression
•	Training: The model was trained using the provided clinical features.
•	Testing: Predictions were made, and the results were rounded to classify into 0 or 1.
•	Accuracy:
The model achieved an accuracy of 78.02%.

Logistic Regression
•	Training: The logistic regression model was fit on the training set.
•	Testing: Predictions were made for the test set.
•	Accuracy:
The model achieved an accuracy of 83.95%.

K-Nearest Neighbors (KNN)
•	Training: KNN was trained with 5 neighbors (default k=5).
•	Testing: The test set was classified based on proximity.
•	Accuracy:
The model achieved an accuracy of 82.72%.

Observations:

•	Logistic Regression remains the best-performing model for this dataset, with the highest accuracy (83.95%).

•	KNN had a lower accuracy (69.13%) compared to Logistic Regression and Linear Regression. This may be due to:
o	Suboptimal choice of features.
o	The model's sensitivity to the dataset distribution.

•	Linear Regression, though not a classification algorithm, showed decent performance (78.02%) when adapted for binary classification.







Visualizations:

•	A scatter plot of the dataset provided an initial understanding of the relationships between features.
•	Performance metrics (accuracy scores) highlight the comparative effectiveness of the models.
 

Conclusion:

Based on the results, Logistic Regression is the most effective model for predicting heart disease using this dataset. Linear Regression also showed strong performance and could be considered depending on specific use cases.






