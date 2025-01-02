from xgboost import XGBRegressor
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from xgboost import plot_importance
import matplotlib.pyplot as plt


data = pd.read_csv("initial_contractYrs_data.csv")
data['contract_value'] = data["contract_value"] / (1e6)
target = data["contract_value"]
data.drop("contract_player_name", axis=1, inplace=True)
data.drop("contract_value", axis=1, inplace=True)
negative_vals = ["total_passing_sacks", "total_passing_sacks_yards_lost", "total_passing_interceptions"]
for val in negative_vals:
    data[val] = -1 * data[val]
X = data
data = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)

'''
test_params = {
    'n_estimators': [100, 200, 250, 275, 1000],
    'max_depth': [1, 3, 5, 10],
    'learning_rate' : [0, 0.01, 0.05, 0.1, 0.15, 0.2],
    'subsample' : [0.5, 0.8, 1.0]
}

gridSearch = GridSearchCV(
    estimator=XGBRegressor(random_state=42),
    param_grid=test_params,
    scoring = 'r2',
    cv=3,
    verbose=1
)

gridSearch.fit(X_train, y_train)
best_params = gridSearch.best_params_
model = XGBRegressor(**best_params, random_state = 42)
print(f"Best Parameters: {best_params}")
'''

model = XGBRegressor(learning_rate=0.01, max_depth=3, n_estimators=275,subsample=0.5)
rfe = RFE(estimator=model, n_features_to_select=4)
rfe.fit(X_train, y_train)
selected_features = X_train.columns[rfe.support_]
print(f"Selected features: {selected_features}")
X_train = X_train[selected_features]
X_test = X_test[selected_features]
model.fit(X_train, y_train)
res = model.predict(X_test)

mse = mean_squared_error(y_test, res)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, res)
r2 = r2_score(y_test, res)

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R2: {r2}")

plot_importance(model, importance_type = 'weight')
plt.show()