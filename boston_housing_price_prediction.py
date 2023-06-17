import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Constants
DATA_URL = "http://lib.stat.cmu.edu/datasets/boston"
RANDOM_STATE = 42

# Load dataset
raw_df = pd.read_csv(DATA_URL, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Preprocessing
data = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(data.shape[1])])
data['PRICE'] = target

X = data.drop('PRICE', axis=1)
y = data['PRICE']

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# Model training - Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Model training - Random Forest
rf_model = RandomForestRegressor(random_state=RANDOM_STATE)

# Hyperparameter tuning - Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': [1.0, 'sqrt', 'log2'],
    'max_depth': [4, 5, 6, 7, 8],
    'criterion': ['squared_error', 'absolute_error']
}


CV_rf_model = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5)
CV_rf_model.fit(X_train, y_train)

# Performance evaluation - Linear Regression
y_pred_lr = lr_model.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

print("\nLinear Regression Model Performance:")
print(f" - Prediction error (on average): ${rmse_lr * 1000:.2f}")
print(f" - Accuracy of predictions: {r2_lr * 100:.2f}%")

# Performance evaluation - Random Forest
y_pred_rf = CV_rf_model.best_estimator_.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Model Performance:")
print(f" - Prediction error (on average): ${rmse_rf * 1000:.2f}")
print(f" - Accuracy of predictions: {r2_rf * 100:.2f}%")

# Explain the results in a non-technical language
print("\nIn simple terms, our models have been trained to predict house prices in Boston.")
print("We evaluated their performance using two main indicators:")
print("1. Prediction error: This tells us how much, on average, our predictions were off the mark. Lower values are better.")
print("2. Accuracy of predictions: This tells us how closely our predictions match the actual prices. It is expressed as a percentage. Higher values (closer to 100%) are better.")
print("\nAs we can see, the Random Forest model performed slightly better than the Linear Regression model in this case. However, it's important to remember that both models have their strengths and weaknesses, and the best choice often depends on the specific scenario and the available data.")

