# Import libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Create dataset
# -------------------------------
data = {
    'Hours_Study': [2, 3, 5, 7, 8, 4, 6, 9, 1, 10],
    'Hours_Sleep': [7, 6, 8, 5, 6, 7, 6, 5, 8, 6],
    'Attendance': [80, 85, 90, 70, 95, 75, 88, 92, 65, 98],
    'Marks': [50, 55, 65, 60, 75, 58, 68, 80, 45, 85]
}
df = pd.DataFrame(data)

# -------------------------------
# Step 2: Prepare features and target
# -------------------------------
X = df[['Hours_Study', 'Hours_Sleep', 'Attendance']]
y = df['Marks']

# -------------------------------
# Step 3: Train Linear Regression model
# -------------------------------
model = LinearRegression()
model.fit(X, y)

# Print model coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# -------------------------------
# Step 4: Make predictions
# -------------------------------
y_pred = model.predict(X)

# -------------------------------
# Step 5: Plot Actual vs Predicted Marks
# -------------------------------
plt.scatter(y, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Perfect prediction line
plt.xlabel('Actual Marks')
plt.ylabel('Predicted Marks')
plt.title('Actual vs Predicted Marks')
plt.grid(True)
plt.show()

# -------------------------------
# Step 6: Compute R² score and Mean Squared Error (MSE)
# -------------------------------
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
print("R² Score:", r2)
print("Mean Squared Error:", mse)
