import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load and preprocess your data
sales_data_path = "supermarket_sales.csv"  # Update with your actual path
sales_data = pd.read_csv(sales_data_path)
sales_data["Date"] = pd.to_datetime(sales_data["Date"], format="%m/%d/%Y")

# Group by branch and date, and aggregate the total sales
branch_sales = (
    sales_data.groupby(["Branch", "Date"]).agg({"Total": "sum"}).reset_index()
)


# Function to create lag features and reshape for LSTM
def create_branch_data_for_lstm(branch_data):
    for lag in range(1, 8):
        branch_data[f"lag_{lag}"] = branch_data["Total"].shift(lag)
    branch_data = branch_data.dropna()

    X = branch_data[[f"lag_{i}" for i in range(1, 8)]]
    y = branch_data["Total"]

    # Reshape the data for LSTM: [samples, timesteps, features]
    X_lstm = X.values.reshape((X.shape[0], 7, 1))
    return X_lstm, y


# Separate data by branch
branches = ["A", "B", "C"]
lstm_predictions = {}
mae_results = {}
accuracy_results = {}

for branch in branches:
    branch_data = branch_sales[branch_sales["Branch"] == branch].copy()
    X_lstm, y = create_branch_data_for_lstm(branch_data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_lstm, y, test_size=0.2, shuffle=False
    )

    # Build the LSTM model
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, activation="relu", input_shape=(7, 1)))
    lstm_model.add(Dense(1))

    # Compile the model
    lstm_model.compile(optimizer="adam", loss="mean_squared_error")

    # Train the model
    lstm_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

    # Predict the next 7 sales
    predictions = lstm_model.predict(X_test[:7])

    # Calculate MAE and Accuracy
    mae = mean_absolute_error(y_test[:7], predictions)
    accuracy = 100 - (mae / y_test[:7].mean() * 100)

    # Store results
    lstm_predictions[branch] = predictions.flatten()
    mae_results[branch] = mae
    accuracy_results[branch] = accuracy

# Display results
for branch in branches:
    print(
        f"Branch {branch}: {lstm_predictions[branch]} MAE: {mae_results[branch]} Accuracy: {accuracy_results[branch]}%"
    )
