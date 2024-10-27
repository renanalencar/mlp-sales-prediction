import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

# Load and preprocess your data
sales_data_path = "data/supermarket_sales.csv"  # Update with your actual path
sales_data = pd.read_csv(sales_data_path)
sales_data["Date"] = pd.to_datetime(sales_data["Date"], format="%m/%d/%Y")

# Group by branch and date, and aggregate the total sales
branch_sales = (
    sales_data.groupby(["Branch", "Date"]).agg({"Total": "sum"}).reset_index()
)

# Streamlit app title
st.title("Daily Sales Prediction Dashboard")

# Display data grid
st.write("### Daily Sales Data")
st.dataframe(branch_sales)


# Plot sales trends for each branch
def plot_sales_trends():
    plt.figure(figsize=(12, 6))
    branches = branch_sales["Branch"].unique()
    for branch in branches:
        branch_data = branch_sales[branch_sales["Branch"] == branch]
        plt.plot(branch_data["Date"], branch_data["Total"], 
                 label=f"Branch {branch}")

    plt.title("Daily Sales Trends for Each Branch")
    plt.xlabel("Date")
    plt.ylabel("Total Sales")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)


# st.write("### Daily Sales Trends")
# plot_sales_trends()

# Plot sales trends for each branch interactively using Plotly
def plot_sales_trends_int():
    # Create a figure
    fig = go.Figure()
    branches = branch_sales["Branch"].unique()
    for branch in branches:
        branch_data = branch_sales[branch_sales["Branch"] == branch]
        # Add trace for branch data
        fig.add_trace(go.Scatter(x=branch_data["Date"], y=branch_data["Total"], 
                                 mode='lines', name=f"Branch {branch}"))
    # Customize layout
    fig.update_layout(title='Sales of branches A, B, and C over time', 
                      xaxis_title='Month', yaxis_title='Sales')
    # Show the plot in Streamlit
    st.plotly_chart(fig)

st.write("### Daily Sales Trends")
plot_sales_trends_int()

# Create a function to handle the model predictions
def create_branch_data(branch_data):
    for lag in range(1, 8):
        branch_data[f"lag_{lag}"] = branch_data["Total"].shift(lag)
    branch_data = branch_data.dropna()

    X = branch_data[[f"lag_{i}" for i in range(1, 8)]]
    y = branch_data["Total"]
    return X, y


def mlp_model(branch, days_to_predict):
    branch_data = branch_sales[branch_sales["Branch"] == branch].copy()
    X, y = create_branch_data(branch_data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlp_model = MLPRegressor(
        hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42
    )
    mlp_model.fit(X_train_scaled, y_train)

    predictions = mlp_model.predict(X_test_scaled[:days_to_predict])

    # Calculate MAE and Accuracy
    mae = mean_absolute_error(y_test[:days_to_predict], predictions)
    accuracy = 100 - (mae / y_test[:days_to_predict].mean() * 100)

    return predictions, mae, accuracy


st.write("### Time Series Prediction")

# Radio buttons to choose branch (A, B, or C)
branch_choice = st.radio("Choose a branch", ("A", "B", "C"))

# Text input to choose how many days to predict (up to 7)
days_to_predict = st.slider(
    "How many days ahead do you want to predict?", 
    min_value=1, max_value=7, value=3
)

# Predict button
if st.button("Predict Daily Sales"):
    predictions, mae, accuracy = mlp_model(branch_choice, days_to_predict)

    st.write(f"### Predictions for Branch {branch_choice}:")
    st.write(predictions)

    st.write(f"Mean Average Error (MAE): {mae:.2f}")
    st.write(f"Accuracy: {accuracy:.2f}%")

st.write("### Project Team:")
st.write("- Beauty Chidinma Iromuanya")
st.write("- Meenaj Modan")
st.write("- Ogonna Silvia Otti")
st.write("- Samuel Cezar Barros Moraes")
