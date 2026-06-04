try:
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from prophet import Prophet
except Exception as e:
    raise ImportError(
        "Missing required packages. Please install dependencies: pip install streamlit pandas plotly prophet\n" + str(e)
    )

# -------------------------

# PAGE CONFIG

# -------------------------

st.set_page_config(
page_title="Sales Forecasting System",
page_icon="📈",
layout="wide"
)

# -------------------------

# LOAD DATA

# -------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("data/walmart_sample.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

# -------------------------

# PROPHET MODEL

# -------------------------

@st.cache_resource
def train_prophet(data):
    from prophet import Prophet

    prophet_df = (
        data.groupby("Date")["Weekly_Sales"]
        .sum()
        .reset_index()
    )

    prophet_df.columns = ["ds", "y"]

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
    )

    model.fit(prophet_df)

    future = model.make_future_dataframe(
        periods=26,
        freq="W"
    )

    forecast = model.predict(future)

    return forecast

# -------------------------

# LOAD DATA

# -------------------------

df = load_data()

# -------------------------

# SIDEBAR

# -------------------------

st.sidebar.title("📊 Navigation")
st.sidebar.markdown("---")

selected_store = st.sidebar.selectbox(
    "🏪 Select Store",
    ["All"] + sorted(df["Store"].unique().tolist())
)

page = st.sidebar.radio(
"Go To",
[
"Dashboard",
"Historical Analysis",
"Forecasting",
"Business Insights"
]
)

# -------------------------

# KPI VALUES

# -------------------------

# Filter dataframe based on selected store before computing KPIs
if selected_store == "All":
    filtered_df = df.copy()
else:
    filtered_df = df[df["Store"] == selected_store]

total_sales = filtered_df["Weekly_Sales"].sum()
avg_sales = filtered_df["Weekly_Sales"].mean()
total_stores = filtered_df["Store"].nunique()
total_departments = filtered_df["Dept"].nunique()

# -------------------------

# DASHBOARD PAGE

# -------------------------

if page == "Dashboard":
    st.title("📈 Sales Forecasting & Revenue Prediction System")

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "💰 Total Revenue",
        f"${total_sales:,.0f}"
    )

    col2.metric(
        "📦 Average Weekly Sales",
        f"${avg_sales:,.0f}"
    )

    col3.metric(
        "🏪 Stores",
        total_stores
    )

    col4.metric(
        "🛒 Departments",
        total_departments
    )

    st.markdown("---")

    sales_trend = (
    filtered_df.groupby("Date")["Weekly_Sales"]
    .sum()
    .reset_index()
)
    fig = px.line(
        sales_trend,
        x="Date",
        y="Weekly_Sales",
        title="Weekly Sales Trend"
    )

    st.plotly_chart(fig, use_container_width=True)

# -------------------------

# HISTORICAL ANALYSIS

# -------------------------

elif page == "Historical Analysis":
    st.title("📊 Historical Analysis")

    col1, col2 = st.columns(2)

    top_stores = (
        df.groupby("Store")["Weekly_Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig_store = px.bar(
        top_stores,
        x="Store",
        y="Weekly_Sales",
        title="Top 10 Stores"
    )

    col1.plotly_chart(
        fig_store,
        use_container_width=True
    )

    top_dept = (
        df.groupby("Dept")["Weekly_Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig_dept = px.bar(
        top_dept,
        x="Dept",
        y="Weekly_Sales",
        title="Top Departments"
    )

    col2.plotly_chart(
        fig_dept,
        use_container_width=True
    )

# -------------------------

# FORECASTING

# -------------------------

elif page == "Forecasting":
    st.title("🔮 Sales Forecasting")

    st.subheader("Forecasted Sales Trend")

    # Train forecast on the currently filtered dataframe
    forecast = train_prophet(filtered_df)

    fig = go.Figure()

    # Forecast Line
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat"],
            mode="lines",
            name="Forecast"
        )
    )

    # Lower Bound
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_lower"],
            mode="lines",
            line=dict(width=0),
            showlegend=False
        )
    )

    # Upper Bound
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_upper"],
            mode="lines",
            fill="tonexty",
            name="Confidence Interval"
        )
    )

    fig.update_layout(
        title="Sales Forecast with Confidence Interval",
        xaxis_title="Date",
        yaxis_title="Sales",
        template="plotly_dark",
        height=600
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    future_forecast = forecast.tail(26)

    predicted_sales = future_forecast["yhat"].sum()

    confidence_range = (
        future_forecast["yhat_upper"].sum()
        -
        future_forecast["yhat_lower"].sum()
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "📈 Forecast Revenue (26 Weeks)",
        f"${predicted_sales:,.0f}"
    )

    col2.metric(
        "🎯 Confidence Range",
        f"${confidence_range:,.0f}"
    )

# -------------------------

# BUSINESS INSIGHTS

# -------------------------

elif page == "Business Insights":
    st.title("💡 Business Insights")

    st.success("Store 20 generates the highest revenue.")

    st.success("Department 92 contributes the most sales.")

    st.success("Holiday periods show significant sales spikes.")

    st.success("Overall sales trend is increasing.")
  
