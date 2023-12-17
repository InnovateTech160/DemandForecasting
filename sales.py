import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load data
data = pd.read_csv('split1234.csv')
data['month'] = pd.to_datetime(data['month'])

# Aggregate sales by month and distribution channel
monthly_sales = data.groupby(['distribution_channel', data['month'].dt.to_period('M')])['net_price'].sum().reset_index()

print(monthly_sales.head(30))

# Pivot data to have months as columns and distribution channels as rows
pivot_sales = monthly_sales.pivot(index='distribution_channel', columns='month', values='net_price')

# Drop any rows with NaN values
pivot_sales.dropna(axis=0, inplace=True)

# Prepare the dataset for linear regression
X = []
y = []
for channel in pivot_sales.index:
    channel_sales = pivot_sales.loc[channel].to_numpy()
    for i in range(len(channel_sales) - 2):
        X.append(channel_sales[i:i+2])
        y.append(channel_sales[i+2])

X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict using the test set
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Metrics Output
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared Score: {r2}")

# Plot Actual vs Predicted Sales
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Diagonal line
plt.show()

# Residual plot
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted Sales')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='red', linestyle='--')
plt.show()

# User input for future prediction
input_channel = input("Enter distribution channel: ")
first_month_sales = float(input("Enter sales for the first month: "))
second_month_sales = float(input("Enter sales for the second month: "))

if input_channel in pivot_sales.index:
    predicted_sales = model.predict([[first_month_sales, second_month_sales]])
    print(f"Predicted sales for the third month: {predicted_sales[0]}")
else:
    print("Distribution channel not found in dataset.")


model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model as a pickle file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as model.pkl")