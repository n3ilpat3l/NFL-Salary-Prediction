import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
input_data_path = "/Users/Tai'Re Barashango/CS4641/Project/initial_contract_yrs_data.csv"
data = pd.read_csv(input_data_path)

# Define features and target
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


# Filter out rows where contract_value is near zero
threshold = 1  # Adjust this threshold as needed
data = data[data[target] > threshold]

# Normalize features
scaler = StandardScaler()
data_features = data[features]
data_features_scaled = scaler.fit_transform(data_features)

# Log-transform the target
data['contract_value_log'] = np.log1p(data['contract_value'])
target_feature = data['contract_value_log']

# Train-test split
data_train, data_test, target_train, target_test = train_test_split(data_features_scaled, target_feature, test_size=0.2, random_state=42)

# Define the neural network model
model = Sequential([
    Dense(32, activation='relu', input_dim=data_train.shape[1]),
    Dropout(0.1),
    Dense(16, activation='relu'),
    Dropout(0.1),
    Dense(1)  # Output layer with linear activation for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    data_train,
    target_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    verbose=2,
    callbacks=[early_stopping]
)

# Predict contract values on the test set
target_pred = model.predict(data_test).flatten()

# Metrics
mae = mean_absolute_error(target_test, target_pred)
rmse = np.sqrt(mean_squared_error(target_test, target_pred))
r2 = r2_score(target_test, target_pred)

print("\nTest Set Performance:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R2): {r2:.4f}")


# Visualize Actual vs Predicted Contract Values
plt.figure(figsize=(8, 6))
plt.scatter(target_test, target_pred, alpha=0.7, label='Predicted')
plt.plot([target_test.min(), target_test.max()], [target_test.min(), target_test.max()], color='red', label='Ideal Fit')
plt.xlabel('Actual Contract Values')
plt.ylabel('Predicted Contract Values')
plt.title('Actual vs. Predicted Contract Values')
plt.legend()
plt.tight_layout()
plt.show()

# Additional Visualizations

# 1. Correlation Matrix
corr_matrix = data[features + [target]].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix')
plt.show()

# 2. Residual Plot
residuals = target_test - target_pred
plt.figure(figsize=(8, 6))
plt.scatter(target_pred, residuals, alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Contract Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.tight_layout()
plt.show()

# Target Distribution - Original Values
sns.histplot(data['contract_value'], kde=True, bins=20, color='green')
plt.title('Distribution of Original Contract Value')
plt.xlabel('Contract Value')
plt.ylabel('Frequency')
plt.show()


# 4. Feature Distributions
for feature in features:
    plt.figure(figsize=(6, 4))
    sns.histplot(data[feature], kde=True, bins=20, color='blue')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

# 5. Pair Plot
sns.pairplot(data[features + [target]], diag_kind='kde')
plt.suptitle('Pair Plot of Features and Target', y=1.02)
plt.show()
