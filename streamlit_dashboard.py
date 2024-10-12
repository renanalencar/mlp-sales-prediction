import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


# Função para carregar os dados do CSV
@st.cache
def load_data():
    file_path = "data/supermarket_sales.csv"  # Substitua pelo caminho correto do seu arquivo
    data = pd.read_csv(file_path)
    data["Date"] = pd.to_datetime(data["Date"])
    data["Month"] = data["Date"].dt.to_period("M")
    return data


# Função para agregar as vendas mensais
def get_monthly_sales(data):
    monthly_sales = data.groupby("Month")["Total"].sum().reset_index()
    return monthly_sales


# Função para treinar o modelo MLP e prever os próximos três meses
def train_mlp_and_predict(monthly_sales):
    X = monthly_sales.index.values.reshape(-1, 1)
    y = monthly_sales["Total"].values

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Padronizar as features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Criar e treinar o modelo MLP
    mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    mlp.fit(X_train_scaled, y_train)

    # Prever no conjunto de teste
    y_pred = mlp.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)

    # Prever os próximos 3 meses
    future_months = np.array(
        [len(monthly_sales), len(monthly_sales) + 1, len(monthly_sales) + 2]
    ).reshape(-1, 1)
    future_months_scaled = scaler.transform(future_months)
    future_sales_pred = mlp.predict(future_months_scaled)

    return future_sales_pred, mae


# Função para criar o gráfico de vendas mensais
def plot_sales(monthly_sales):
    fig, ax = plt.subplots()
    ax.plot(monthly_sales["Month"].astype(str), monthly_sales["Total"], marker="o")
    ax.set_title("Vendas Mensais")
    ax.set_xlabel("Mês")
    ax.set_ylabel("Total de Vendas")
    plt.xticks(rotation=45)
    st.pyplot(fig)


# Título do Dashboard
st.title("Dashboard de Previsão de Vendas - Supermercado")

# Carregar os dados
data = load_data()

# Exibir os dados carregados
st.subheader("Dados Brutos do Supermercado")
st.write(data.head())

# Agregar as vendas mensais
monthly_sales = get_monthly_sales(data)

# Exibir o gráfico de vendas mensais
st.subheader("Total de Vendas Mensais")
plot_sales(monthly_sales)

# Treinar o modelo e prever as vendas dos próximos três meses
if st.button("Prever Próximos 3 Meses"):
    future_sales_pred, mae = train_mlp_and_predict(monthly_sales)

    st.subheader("Previsões para os Próximos 3 Meses")
    st.write(f"Erro Absoluto Médio (MAE): {mae}")

    for i, sales in enumerate(future_sales_pred, 1):
        st.write(f"Mês {i}: {sales:.2f}")
