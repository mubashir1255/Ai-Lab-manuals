# Import libraries
import pandas as pd
from sklearn.linear_model import LinearRegression

# Create toy dataset
data = {
    'Size': [1500, 1800, 2400, 3000, 3500],
    'Bedrooms': [3, 4, 3, 5, 4],
    'Age': [10, 15, 20, 5, 8],
    'Price': [400000, 500000, 600000, 650000, 700000]
}
df = pd.DataFrame(data)

# Features and target
X = df[['Size', 'Bedrooms', 'Age']]
y = df['Price']

# Fit model
model = LinearRegression()
model.fit(X, y)

# Print coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Predict price for a new house using a DataFrame (fixes warning)
new_house = pd.DataFrame({'Size':[2000], 'Bedrooms':[3], 'Age':[10]})
predicted_price = model.predict(new_house)
print("Predicted House Price:", predicted_price[0])
