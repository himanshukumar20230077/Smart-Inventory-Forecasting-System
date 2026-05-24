import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="Smart Inventory Forecasting System",
    layout="wide"
)

# ======================================
# TITLE
# ======================================

st.markdown(
    """
    <h1 style='text-align:center;color:#00BFFF;'>
     Smart Inventory Forecasting & Analytics System
    </h1>
    """,
    unsafe_allow_html=True
)

# ======================================
# APP DESCRIPTION
# ======================================

st.markdown(
    """
    ###  AI-Powered Business Analytics Platform

    This platform allows businesses to:

    - Upload inventory CSV files
    - Generate analytics dashboards
    - Forecast future sales demand
    - Identify top & low performing products
    - Download processed reports
    """
)

# ======================================
# FILE UPLOAD
# ======================================

uploaded_file = st.file_uploader(
    " Upload Inventory CSV File",
    type=["csv"]
)

# ======================================
# FILE FORMAT INFO
# ======================================

st.info("""

 Upload CSV file with these required columns:

- Product_ID
- Sale_Date
- Region
- Sales_Amount
- Quantity_Sold
- Product_Category
- Unit_Price
- Discount
- Customer_Type
- Sales_Channel

""")

# ======================================
# FILE CHECK
# ======================================

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("✅ CSV Uploaded Successfully")

else:

    st.warning("⚠ Please upload a CSV file")

    st.stop()

# ======================================
# REQUIRED COLUMNS VALIDATION
# ======================================

required_columns = [

    'Product_ID',
    'Sale_Date',
    'Region',
    'Sales_Amount',
    'Quantity_Sold',
    'Product_Category',
    'Unit_Price',
    'Discount',
    'Customer_Type',
    'Sales_Channel'

]

missing_columns = [

    col for col in required_columns

    if col not in df.columns

]

if missing_columns:

    st.error(
        f"❌ Missing Columns: {missing_columns}"
    )

    st.stop()

# ======================================
# CLEANING
# ======================================

df.columns = df.columns.str.strip()

df['Sale_Date'] = pd.to_datetime(
    df['Sale_Date']
)

df['MONTH'] = df['Sale_Date'].dt.month

df['YEAR'] = df['Sale_Date'].dt.year

df = df.drop_duplicates()

# ======================================
# LABEL ENCODING
# ======================================

le_product = LabelEncoder()

le_region = LabelEncoder()

le_customer = LabelEncoder()

le_channel = LabelEncoder()

df['Product_Category_Encoded'] = le_product.fit_transform(
    df['Product_Category']
)

df['Region_Encoded'] = le_region.fit_transform(
    df['Region']
)

df['Customer_Type_Encoded'] = le_customer.fit_transform(
    df['Customer_Type']
)

df['Sales_Channel_Encoded'] = le_channel.fit_transform(
    df['Sales_Channel']
)

# ======================================
# FEATURES
# ======================================

features = [

    'Quantity_Sold',
    'Unit_Price',
    'Discount',
    'MONTH',
    'YEAR',
    'Product_Category_Encoded',
    'Region_Encoded',
    'Customer_Type_Encoded',
    'Sales_Channel_Encoded'

]

target = 'Sales_Amount'

# ======================================
# CREATE X/y
# ======================================

X = df[features]

y = df[target]

# ======================================
# TRAIN TEST SPLIT
# ======================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42

)

# ======================================
# MODEL
# ======================================

model = RandomForestRegressor(

    n_estimators=100,
    random_state=42

)

model.fit(X_train, y_train)

# ======================================
# SIDEBAR
# ======================================

st.sidebar.title("📦 Inventory Navigation")

st.sidebar.markdown("---")

st.sidebar.success(
    "AI-Powered Inventory Analytics Platform"
)

option = st.sidebar.radio(

    "Go To",

    [

        "Dashboard",
        "Sales Analysis",
        "Forecast Prediction"

    ]

)

# ======================================
# DASHBOARD
# ======================================

if option == "Dashboard":

    st.subheader("📊 Inventory Dashboard")

    total_sales = df['Sales_Amount'].sum()

    total_products = df['Product_ID'].nunique()

    avg_sales = df['Sales_Amount'].mean()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "💰 Total Sales",
        f"${total_sales:,.2f}"
    )

    col2.metric(
        "📦 Total Products",
        total_products
    )

    col3.metric(
        "📈 Average Sales",
        f"${avg_sales:,.2f}"
    )

    # ============================
    # LOW SALES ALERTS
    # ============================

    st.subheader("⚠ Low Sales Product Alerts")

    low_sales = (
        df.groupby('Product_Category')['Sales_Amount']
        .sum()
        .sort_values()
        .head(5)
    )

    st.dataframe(low_sales)

    # ============================
    # TOP PRODUCTS
    # ============================

    st.subheader("🏆 Top Performing Products")

    top_sales = (
        df.groupby('Product_Category')['Sales_Amount']
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )

    st.dataframe(top_sales)

    # ============================
    # DOWNLOAD REPORT
    # ============================

    st.subheader("📥 Download Analytics Report")

    csv = df.to_csv(index=False).encode('utf-8')

    st.download_button(

        label="Download Processed CSV",

        data=csv,

        file_name='processed_inventory_report.csv',

        mime='text/csv'

    )

    # ============================
    # BUSINESS INSIGHTS
    # ============================

    st.subheader("📌 Business Insights")

    highest_month = (
        df.groupby('MONTH')['Sales_Amount']
        .sum()
        .idxmax()
    )

    highest_product = (
        df.groupby('Product_Category')['Sales_Amount']
        .sum()
        .idxmax()
    )

    st.info(
        f"📈 Highest sales occurred in Month {highest_month}"
    )

    st.info(
        f"🏆 Top selling product category: {highest_product}"
    )

    # ============================
    # DATA PREVIEW
    # ============================

    st.subheader("📄 Dataset Preview")

    st.dataframe(df.head())

# ======================================
# SALES ANALYSIS
# ======================================

elif option == "Sales Analysis":

    st.subheader("📊 Monthly Sales Analysis")

    monthly_sales = (
        df.groupby('MONTH')['Sales_Amount']
        .sum()
    )

    fig, ax = plt.subplots(figsize=(10,5))

    monthly_sales.plot(
        kind='bar',
        ax=ax
    )

    plt.title("Monthly Sales")

    plt.xlabel("Month")

    plt.ylabel("Sales")

    st.pyplot(fig)

    # ============================
    # TOP CATEGORY ANALYSIS
    # ============================

    st.subheader("🏆 Top Product Categories")

    top_products = (
        df.groupby('Product_Category')['Sales_Amount']
        .sum()
        .sort_values(ascending=False)
    )

    fig2, ax2 = plt.subplots(figsize=(10,5))

    top_products.plot(
        kind='bar',
        ax=ax2
    )

    plt.title("Top Product Categories")

    plt.xlabel("Product Category")

    plt.ylabel("Sales")

    st.pyplot(fig2)

# ======================================
# FORECAST PREDICTION
# ======================================

elif option == "Forecast Prediction":

    st.subheader("🔮 Sales Forecast Prediction")

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
        2030,
        2025
    )

    # ======================================
    # DROPDOWNS
    # ======================================

    product_category = st.selectbox(
        "Product Category",
        df['Product_Category'].unique()
    )

    region_name = st.selectbox(
        "Region",
        df['Region'].unique()
    )

    customer_type = st.selectbox(
        "Customer Type",
        df['Customer_Type'].unique()
    )

    sales_channel = st.selectbox(
        "Sales Channel",
        df['Sales_Channel'].unique()
    )

    # ======================================
    # ENCODE USER INPUTS
    # ======================================

    product = le_product.transform(
        [product_category]
    )[0]

    region = le_region.transform(
        [region_name]
    )[0]

    customer = le_customer.transform(
        [customer_type]
    )[0]

    channel = le_channel.transform(
        [sales_channel]
    )[0]

    # ======================================
    # PREDICT BUTTON
    # ======================================

    if st.button("🚀 Predict Sales"):

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
            f"💰 Predicted Sales: ${prediction[0]:,.2f}"
        )