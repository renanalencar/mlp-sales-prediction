import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

# Load and preprocess your data
sales_data_path = "supermarket_sales.csv"  # Update with your actual path
sales_data = pd.read_csv(sales_data_path)
sales_data["Date"] = pd.to_datetime(sales_data["Date"], format="%m/%d/%Y")

# Group by branch and date, and aggregate the total sales
branch_sales = (
    sales_data.groupby(["Branch", "Date"]).agg({"Total": "sum"}).reset_index()
)


# Function to create lag features and train/test sets for each branch
def create_branch_data(branch_data):
    for lag in range(1, 8):
        branch_data[f"lag_{lag}"] = branch_data["Total"].shift(lag)
    branch_data = branch_data.dropna()

    X = branch_data[[f"lag_{i}" for i in range(1, 8)]]
    y = branch_data["Total"]
    return X, y


# Separate data by branch
branches = ["A", "B", "C"]
mlp_predictions = {}
mae_results = {}
accuracy_results = {}

for branch in branches:
    branch_data = branch_sales[branch_sales["Branch"] == branch].copy()
    X, y = create_branch_data(branch_data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Normalize the data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the MLP model
    mlp_model = MLPRegressor(
        hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42
    )
    mlp_model.fit(X_train_scaled, y_train)

    # Predict the next 7 sales
    predictions = mlp_model.predict(X_test_scaled[:7])

    # Calculate MAE and Accuracy
    mae = mean_absolute_error(y_test[:7], predictions)
    accuracy = 100 - (mae / y_test[:7].mean() * 100)

    # Store results
    mlp_predictions[branch] = predictions
    mae_results[branch] = mae
    accuracy_results[branch] = accuracy

# Display results
for branch in branches:
    print(
        f"Branch {branch}: {mlp_predictions[branch]} MAE: {mae_results[branch]} Accuracy: {accuracy_results[branch]}%"
    )
