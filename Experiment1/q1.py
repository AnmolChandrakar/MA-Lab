import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Prepare Data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Perform Linear Regression
X_reshaped = x.reshape(-1, 1)
model = LinearRegression()
model.fit(X_reshaped, y)
slope = model.coef_[0]
intercept = model.intercept_

print(f"\nCalculated Slope (m): {slope:.4f}")
print(f"Calculated Intercept (b): {intercept:.4f}")

# Calculate Predictions
y_pred = model.predict(X_reshaped)

print("\nPredicted y-values (y_pred) using model.predict():")
print(y_pred)

# Calculate Performance Metrics
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r_squared = r2_score(y, y_pred)
n = len(y)
p = 1
adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
see = np.sqrt(np.sum((y - y_pred)**2) / (n - p - 1))

print(f"\nMean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R^2): {r_squared:.4f}")
print(f"Adjusted R-squared: {adjusted_r_squared:.4f}")
print(f"Standard Error of Estimate (SEE): {see:.4f}")

# Visualize Regression Line
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='Actual Data Points')
plt.plot(x, y_pred, color='red', label='Regression Line')
plt.title('Linear Regression with Data Points')
plt.xlabel('X-values')
plt.ylabel('Y-values')
plt.legend()
plt.grid(True)
plt.show()

# Plot 3: Error Metrics visualization
error_metrics = ['MAE', 'MSE', 'RMSE', 'SEE', 'R-squared']
error_values = [mae, mse, rmse, see, r_squared]

plt.figure()
plt.bar(error_metrics, error_values)
plt.title('Error Metrics Comparison')
plt.ylabel('Error Value')
plt.show()


