import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# PAGE CONFIG
st.set_page_config(
    page_title="Smart Inventory Forecasting System",
    layout="wide"
)

# TITLE
st.title("📦 Smart Inventory Forecasting & Analytics System")

# LOAD DATA
df = pd.read_csv("outputs/cleaned_inventory_data.csv")

# LOAD MODEL
with open("models/inventory_forecast_model.pkl", "rb") as file:
    model = pickle.load(file)

# SIDEBAR
st.sidebar.title("Navigation")

option = st.sidebar.radio(
    "Go To",
    [
        "Dashboard",
        "Sales Analysis",
        "Forecast Prediction"
    ]
)

# DASHBOARD
if option == "Dashboard":

    st.subheader("Inventory Dashboard")

    total_sales = df['Sales_Amount'].sum()

    total_products = df['Product_ID'].nunique()

    avg_sales = df['Sales_Amount'].mean()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Sales",
        f"${total_sales:,.2f}"
    )

    col2.metric(
        "Total Products",
        total_products
    )

    col3.metric(
        "Average Sales",
        f"${avg_sales:,.2f}"
    )

# SALES ANALYSIS
elif option == "Sales Analysis":

    st.subheader("Monthly Sales Analysis")

    monthly_sales = df.groupby('MONTH')['Sales_Amount'].sum()

    fig, ax = plt.subplots(figsize=(10,5))

    monthly_sales.plot(
        kind='bar',
        ax=ax
    )

    plt.title("Monthly Sales")

    plt.xlabel("Month")

    plt.ylabel("Sales")

    st.pyplot(fig)

# FORECAST SECTION
elif option == "Forecast Prediction":

    st.subheader("Forecast Prediction")

    quantity = st.number_input(
        "Quantity Sold",
        min_value=1,
        value=10
    )

    price = st.number_input(
        "Unit Price",
        min_value=1.0,
        value=100.0
    )

    discount = st.number_input(
        "Discount",
        min_value=0.0,
        value=5.0
    )

    month = st.slider(
        "Month",
        1,
        12,
        6
    )

    year = st.slider(
        "Year",
        2023,
        2025,
        2024
    )

    product = st.number_input(
        "Product Category Encoded",
        min_value=0,
        value=1
    )

    region = st.number_input(
        "Region Encoded",
        min_value=0,
        value=1
    )

    customer = st.number_input(
        "Customer Type Encoded",
        min_value=0,
        value=1
    )

    channel = st.number_input(
        "Sales Channel Encoded",
        min_value=0,
        value=1
    )

    if st.button("Predict Sales"):

        prediction = model.predict([[
            quantity,
            price,
            discount,
            month,
            year,
            product,
            region,
            customer,
            channel
        ]])

        st.success(
            f"Predicted Sales: ${prediction[0]:,.2f}"
        )