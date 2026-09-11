
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

DATA_FILE = "outputs/dashboard_data.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


# -----------------------------
# Load Data
# -----------------------------
try:
    df = load_data()
except FileNotFoundError:
    st.error("Dashboard data file not found: outputs/dashboard_data.csv")
    st.stop()


# -----------------------------
# Header
# -----------------------------
st.title("📊 Customer Segmentation Dashboard")
st.write(
    "Interactive customer segmentation analysis using K-Means clustering."
)

st.divider()


# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Dashboard Filters")

segments = sorted(df["Customer_Segment"].dropna().unique().tolist())

selected_segments = st.sidebar.multiselect(
    "Customer Segment",
    options=segments,
    default=segments
)

filtered_df = df[
    df["Customer_Segment"].isin(selected_segments)
]


# -----------------------------
# KPI Metrics
# -----------------------------
if len(filtered_df) > 0:

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        len(filtered_df)
    )

    col2.metric(
        "Average Age",
        f"{filtered_df['Age'].mean():.1f}"
    )

    col3.metric(
        "Average Income",
        f"{filtered_df['Annual Income (k$)'].mean():.1f}k"
    )

    col4.metric(
        "Average Spending Score",
        f"{filtered_df['Spending Score (1-100)'].mean():.1f}"
    )

else:
    st.warning("Please select at least one customer segment.")
    st.stop()


st.divider()


# -----------------------------
# Customer Distribution
# -----------------------------
st.subheader("Customer Distribution by Segment")

segment_counts = (
    filtered_df["Customer_Segment"]
    .value_counts()
    .sort_values(ascending=False)
)

fig1, ax1 = plt.subplots(figsize=(10, 5))

ax1.bar(
    segment_counts.index,
    segment_counts.values
)

ax1.set_xlabel("Customer Segment")
ax1.set_ylabel("Number of Customers")
ax1.set_title("Customer Distribution")

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()

st.pyplot(
    fig1,
    use_container_width=True
)

plt.close(fig1)


# -----------------------------
# Income vs Spending
# -----------------------------
st.subheader("Income vs Spending Score")

fig2, ax2 = plt.subplots(figsize=(10, 6))

for segment in filtered_df["Customer_Segment"].unique():

    segment_data = filtered_df[
        filtered_df["Customer_Segment"] == segment
    ]

    ax2.scatter(
        segment_data["Annual Income (k$)"],
        segment_data["Spending Score (1-100)"],
        s=70,
        alpha=0.8,
        label=segment
    )

ax2.set_xlabel("Annual Income (k$)")
ax2.set_ylabel("Spending Score (1-100)")
ax2.set_title("Customer Segmentation")

ax2.legend(
    title="Customer Segment",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()

st.pyplot(
    fig2,
    use_container_width=True
)

plt.close(fig2)


# -----------------------------
# Segment Summary
# -----------------------------
st.subheader("Segment Summary")

summary = (
    filtered_df
    .groupby("Customer_Segment")
    .agg(
        Customers=("CustomerID", "count"),
        Average_Age=("Age", "mean"),
        Average_Income=("Annual Income (k$)", "mean"),
        Average_Spending_Score=("Spending Score (1-100)", "mean")
    )
    .reset_index()
)

summary["Average_Age"] = summary["Average_Age"].round(1)
summary["Average_Income"] = summary["Average_Income"].round(1)
summary["Average_Spending_Score"] = (
    summary["Average_Spending_Score"].round(1)
)

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
)


# -----------------------------
# Customer Data
# -----------------------------
st.subheader("Customer-Level Data")

display_columns = [
    "CustomerID",
    "Genre",
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)",
    "Cluster",
    "Customer_Segment"
]

st.dataframe(
    filtered_df[display_columns],
    use_container_width=True,
    hide_index=True
)


# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption(
    "Customer Segmentation Project | K-Means Clustering | "
    "Interactive Streamlit Dashboard"
)
