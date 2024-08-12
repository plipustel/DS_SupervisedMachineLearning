import numpy as np
import pandas as pd

# Load the dataset
data = pd.read_csv('D:/TheCoder/DataScience/DS_SupervisedMachineLearning/datasets/AmesHousing.csv')

# For simplicity, we'll use only two variables, GrLivArea (input feature) and SalePrice (target)
x = data['Gr Liv Area'].values
y = data['SalePrice'].values
print(x, y)
# Step 1: Calculate the means of x and y
x_mean = np.mean(x)
y_mean = np.mean(y)

# Step 2: Calculate the slope (b1) and intercept (b0)
numerator = 0
denominator = 0
for i in range(len(x)):
    numerator += (x[i] - x_mean) * (y[i] - y_mean)
    denominator += (x[i] - x_mean) ** 2
b1 = numerator / denominator
b0 = y_mean - (b1 * x_mean)

print(f'Intercept (b0): {b0}')
print(f'Slope (b1): {b1}')

# Step 3: Make predictions
y_pred = b0 + b1 * x

# Step 4: Calculate Mean Squared Error (MSE) and R^2 Score
mse = np.mean((y - y_pred) ** 2)
ss_tot = np.sum((y - y_mean) ** 2)
ss_res = np.sum((y - y_pred) ** 2)
r2_score = 1 - (ss_res / ss_tot)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2_score}')

# Optional: Plotting the results
import matplotlib.pyplot as plt

plt.scatter(x, y, color='blue', label='Actual data')
plt.plot(x, y_pred, color='red', label='Regression line')
plt.xlabel('Gr Liv Area')
plt.ylabel('SalePrice')
plt.legend()
plt.show()
