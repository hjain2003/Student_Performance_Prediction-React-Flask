import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the dataset
dataset = pd.read_csv('Student_Performance.csv')

# Replacing N/A numerical column input features with mean
numerical_cols = ['hours_studied', 'previous_score', 'sleep_hours', 'sample_papers_solved']
for col in numerical_cols:
    dataset[col].fillna(dataset[col].mean(), inplace=True)

# Removing categorical column input features
categorical_columns = ['extracurricular_activities']
dataset = dataset.dropna(subset=categorical_columns)

# Removing duplicate rows
dataset = dataset[~dataset.duplicated()]

# Encoding binary categories
dataset['extracurricular_activities'] = dataset['extracurricular_activities'].map({'Yes': 1, 'No': 0})

# Extracting input and output features
X = dataset.drop(['performance'], axis=1)
y = dataset['performance']

# Splitting training and testing data (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler to a file
joblib.dump(scaler, 'scaler.pkl')

# Training the model
linear_regressor = LinearRegression()
linear_regressor.fit(X_train_scaled, y_train)

# Load model into a .pkl file
joblib.dump(linear_regressor, 'linear_regression_model.pkl')

# Making predictions for evaluation
y_pred_linear = linear_regressor.predict(X_test_scaled)

# Evaluation parameters
linear_mse = mean_squared_error(y_test, y_pred_linear)
linear_r2 = r2_score(y_test, y_pred_linear)

# Printing results
print(f'\nLinear Regression Mean Squared Error: {linear_mse}')
print(f'Linear Regression R2 score: {linear_r2}')
