import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

input_data_path = '/Users/dakshsharma/Desktop/ML/cs4641-group-48/initial_contract_yrs_data.csv'
data = pd.read_csv(input_data_path)

features = [
    'total_passing_attempts',
    'total_passing_completions',
    'total_passing_yards',
    'average_passing_rating',
    'total_passing_touchdowns',
    'total_passing_interceptions',
    'total_passing_sacks',
    'total_passing_sacks_yards_lost',
    'total_rushing_attempts',
    'total_rushing_yards',
    'total_rushing_touchdowns',
    'total_games_played'
]
target = 'contract_value'

data['total_passing_yards'] *= 50
data['average_passing_rating'] *= 0
data['total_passing_touchdowns'] *= 1000
data['total_rushing_touchdowns'] *= 200
data['total_rushing_yards'] *= 500
data['total_passing_sacks'] *= 0
data['total_games_played'] *= 0
data['total_rushing_attempts'] *= 0

data_features = data[features]
target_feature = data[target]

data_train, data_test, target_train, target_test = train_test_split(
    data_features, target_feature, test_size=0.2, random_state=42
)


param_grid = {
    'n_estimators': [50, 110, 83],
    'max_depth': [4, 5, 15],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3],
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    scoring='r2',
    cv=3,
    n_jobs=-1,
)


grid_search.fit(data_train, target_train)


best_rf_model = grid_search.best_estimator_


rf_predictions = best_rf_model.predict(data_test)


rf_mae = mean_absolute_error(target_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(target_test, rf_predictions))
rf_r2 = r2_score(target_test, rf_predictions)

print("Random Forest Model Metrics (After Applying Weights and Hyperparameter Tuning):")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Mean Absolute Error (MAE): {rf_mae:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rf_rmse:,.2f}")
print(f"R-squared (R2): {rf_r2:.4f}")


feature_importance = pd.DataFrame({
    'feature': features,
    'importance': best_rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("\nFeature Importance:")
for idx, row in feature_importance.iterrows():
    print(f"{row['feature']}: {row['importance']:.4f}")


rf_train_predictions = best_rf_model.predict(data_train)
rf_train_r2 = r2_score(target_train, rf_train_predictions)


print(f"Training R²: {rf_train_r2:.4f}")
print(f"Test R²: {rf_r2:.4f}")

# # ---- Visualization ----

plt.figure(figsize=(12, 10))
correlation_matrix = data[features + [target]].corr()

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    cbar=True,
    linewidths=0.5,
    annot_kws={"size": 4},
    cbar_kws={'shrink':.7}
)

plt.xticks(fontsize=5)
plt.yticks(fontsize=5)

plt.xlabel("Features and Target", fontsize=5) 
plt.ylabel("Features and Target", fontsize=5)

plt.title("Feature Correlation Heatmap", fontsize=14)
plt.show()

plt.figure(figsize=(14, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance')
plt.yticks(fontsize=7)
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(target_test, rf_predictions, color='blue', alpha=0.5)
plt.plot([min(target_test), max(target_test)], [min(target_test), max(target_test)], color='red', linestyle='--')
plt.title('Prediction vs Actual')
plt.xlabel('Actual Contract Value')
plt.ylabel('Predicted Contract Value')
plt.show()

