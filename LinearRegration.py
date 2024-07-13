# package need to be installed(numpy matplotlib scikit-learn)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 3 + 4 * X + np.random.randn(100, 1)

# Visualize the data
plt.scatter(X, y, color='blue')
plt.title('Synthetic Data for Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
X_new = np.array([[0], [2]])
y_pred = model.predict(X_new)

# Plot the fitted line
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X_new, y_pred, color='red', label='Linear regression')
plt.title('Linear Regression Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Print the coefficients
print(f'Coefficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')