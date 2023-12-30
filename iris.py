# Import necessary libraries
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Provide the correct absolute path to 'IRIS.csv'
file_path = 'IRIS.csv'
iris = pd.read_csv(file_path, encoding='latin-1')

# Explore the dataset
print(iris.head())
print(iris.info())

# Select features and target variable
features = ['sepal_width', 'petal_length', 'petal_width']  # Use relevant features
X = iris[features]
y = iris['sepal_length']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
print(f'R2 Score: {r2:.2f}')

# Visualize feature coefficients
coefficients = pd.Series(model.coef_, index=features)
sns.barplot(x=coefficients, y=coefficients.index, orient='h')
plt.title('Feature Coefficients')
plt.show()

# Pair plot to visualize relationships between features
sns.pairplot(iris, hue='species', height=2.5)
plt.suptitle('Pair Plot of Iris Dataset', y=1.02)
plt.show()

# Residual plot to analyze the residuals of the Linear Regression model
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Distribution plot of actual vs predicted values
sns.kdeplot(y_test, label='Actual Values', shade=True)
sns.kdeplot(y_pred, label='Predicted Values', shade=True)
plt.xlabel('Sepal Length')
plt.title('Distribution of Actual vs Predicted Sepal Length')
plt.legend()
plt.show()