import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


# Function to load data from CSV
@st.cache_data
def load_data():
    # Replace with the correct path to your file
    file_path = "data/supermarket_sales.csv"
    data = pd.read_csv(file_path)
    data["Date"] = pd.to_datetime(data["Date"])
    data["Month"] = data["Date"].dt.to_period("M")
    return data


# Function to aggregate monthly sales
def get_monthly_sales(data):
    monthly_sales = data.groupby("Month")["Total"].sum().reset_index()
    return monthly_sales


# Function to train the MLP model and predict the next three months
def train_mlp_and_predict(monthly_sales):
    X = monthly_sales.index.values.reshape(-1, 1)
    y = monthly_sales["Total"].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train the MLP model
    mlp = MLPRegressor(hidden_layer_sizes=(100, 50), 
                       max_iter=1000, random_state=42)
    mlp.fit(X_train_scaled, y_train)

    # Predict on the test set
    y_pred = mlp.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)

    # Predict the next 3 months
    future_months = np.array(
        [len(monthly_sales), len(monthly_sales) + 1, len(monthly_sales) + 2]
    ).reshape(-1, 1)
    future_months_scaled = scaler.transform(future_months)
    future_sales_pred = mlp.predict(future_months_scaled)

    return future_sales_pred, mae


# Function to create the monthly sales chart
def plot_sales(monthly_sales):
    fig, ax = plt.subplots()
    ax.plot(monthly_sales["Month"].astype(str), 
            monthly_sales["Total"], marker="o")
    ax.set_title("Vendas Mensais")
    ax.set_xlabel("Mês")
    ax.set_ylabel("Total de Vendas")
    plt.xticks(rotation=45)
    st.pyplot(fig)


# Dashboard Title
st.title("Sales Prediction Dashboard - Supermarket")

# Load the data
data = load_data()

# Display the loaded data
st.subheader("Supermarket Raw Data")
st.write(data.head())

# Aggregate monthly sales
monthly_sales = get_monthly_sales(data)

# Display the monthly sales chart
st.subheader("Monthly Sales Total")
plot_sales(monthly_sales)

# Train the model and predict the sales for the next three months
if st.button("Predict Next 3 Months"):
    future_sales_pred, mae = train_mlp_and_predict(monthly_sales)

    st.subheader("Predictions for the Next 3 Months")
    st.write(f"Mean Absolute Error (MAE): {mae}")

    for i, sales in enumerate(future_sales_pred, 1):
        st.write(f"Month {i}: {sales:.2f}")
