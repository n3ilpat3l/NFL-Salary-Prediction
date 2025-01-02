import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib import pyplot as plt
import numpy as np

input_data_path = '/Users/neilpatel/Documents/College/4th Year/Semester 1/CS 4641/Project/cs4641-group-48/initial_contract_yrs_data.csv'
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

data['total_passing_yards'] *= 3
data['average_passing_rating'] *= 4
data['total_passing_touchdowns'] *= 5
data['total_rushing_touchdowns'] *= 5
data['total_rushing_yards'] *= 3

data['total_passing_sacks'] *= 0.5
data['total_games_played'] *= 0.5
data['total_rushing_attempts'] *= 0.5


data_features = data[features]
target_feature = data[target]

#split
data_train, data_test, target_train, target_test = train_test_split(data_features, target_feature, test_size=0.2, random_state=42) ##DONT TOUCH RANDOM STATE KEEPS TEST DATA CONSISTENT

model = Ridge(alpha=1600) ##DONT CHANGE ALPHA ITS BEEN TESTED
model.fit(data_train, target_train)

target_pred = model.predict(data_test)

mae = mean_absolute_error(target_test, target_pred)
rmse = np.sqrt(mean_squared_error(target_test, target_pred))
r2 = r2_score(target_test, target_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")

plt.scatter(target_test, target_pred)
plt.xlabel('Actual Contract')
plt.ylabel('Predicted Contract')
plt.title('Actual vs. Predicted Contract')
plt.plot([target_test.min(), target_test.max()], [target_test.min(), target_test.max()], color='black')
plt.show()
