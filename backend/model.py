import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import joblib

dataset = pd.read_csv('SalaryData.csv')

numerical_cols = ['Age', 'Years_of_Experience']
for col in numerical_cols:
    dataset[col].fillna(dataset[col].mean(), inplace=True)

categorical_columns = ['Gender', 'Education_Level', 'Job_Title']
dataset = dataset.dropna(subset=categorical_columns)

dataset = dataset[~dataset.duplicated()]

# Mapping of string values to encoded values
gender_mapping = {'Male': 0, 'Female': 1}
education_level_mapping = {"Bachelor's": 1, "Master's": 2, "PhD": 3}

# Extract unique job titles and create a mapping
unique_job_titles = dataset['Job_Title'].unique()
job_title_mapping = {job_title: [0] * len(unique_job_titles) for job_title in unique_job_titles}
for job_title, encoded_value in job_title_mapping.items():
    encoded_value[unique_job_titles.tolist().index(job_title)] = 1

dataset['Gender'] = dataset['Gender'].map(gender_mapping)
dataset['Education_Level'] = dataset['Education_Level'].map(education_level_mapping)
dataset = pd.get_dummies(dataset, columns=['Job_Title'], prefix='', prefix_sep='')

X = dataset.drop(['Salary'], axis=1)
y = dataset['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

decision_tree = DecisionTreeRegressor(random_state=42)
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
lasso_model = Lasso(alpha=1.0)
ridge_model = Ridge(alpha=1.0)

lasso_model.fit(X_train_scaled, y_train)
decision_tree.fit(X_train_scaled, y_train)
random_forest.fit(X_train_scaled, y_train)
ridge_model.fit(X_train_scaled, y_train)

y_pred_lasso = lasso_model.predict(X_test_scaled)
y_pred_dt = decision_tree.predict(X_test_scaled)
y_pred_rf = random_forest.predict(X_test_scaled)
y_pred_ridge = ridge_model.predict(X_test_scaled)

mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)


print("Lasso Regression:")
print("Mean Squared Error: {:.2f}".format(mse_lasso))
print("R-squared (R2) Score: {:.2f}".format(r2_lasso))

print("\nDecision Tree:")
print("Mean Squared Error: {:.2f}".format(mse_dt))
print("R-squared (R2) Score: {:.2f}".format(r2_dt))

print("\nRandom Forest:")
print("Mean Squared Error: {:.2f}".format(mse_rf))
print("R-squared (R2) Score: {:.2f}".format(r2_rf))

print("\nRidge Regression:")
print("Mean Squared Error: {:.2f}".format(mse_ridge))
print("R-squared (R2) Score: {:.2f}".format(r2_ridge))

joblib.dump(lasso_model, 'hybrid_predictions.pkl')



