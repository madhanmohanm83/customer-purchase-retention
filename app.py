
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# --------------------------------------------------
# 1. PROJECT SETTINGS
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "sales_and_customer_insights.csv"
RESULTS_DIR = PROJECT_DIR / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"

RESULTS_DIR.mkdir(exist_ok=True)
GRAPHS_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="Customer Retention Analytics",
    page_icon="🛍️",
    layout="wide"
)

st.title("Customer Purchase Behavior & Retention Strategy")
st.write(
    "Explore customer purchasing patterns, segment customers, "
    "review churn-risk scores and plan retention actions."
)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)

    for col in ["Launch_Date", "Peak_Sales_Date"]:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors="coerce")

    return data


if not DATA_FILE.exists():
    st.error(f"Dataset not found: {DATA_FILE}")
    st.stop()

df = load_data(str(DATA_FILE))

required_columns = [
    "Customer_ID",
    "Purchase_Frequency",
    "Average_Order_Value",
    "Most_Frequent_Category",
    "Time_Between_Purchases",
    "Region",
    "Churn_Probability",
    "Lifetime_Value",
    "Season",
    "Preferred_Purchase_Times",
    "Retention_Strategy",
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"Required columns missing: {missing_columns}")
    st.stop()

# --------------------------------------------------
# 3. SIDEBAR FILTERS
# --------------------------------------------------
st.sidebar.header("Dashboard Filters")

regions = sorted(df["Region"].dropna().unique().tolist())
categories = sorted(
    df["Most_Frequent_Category"].dropna().unique().tolist()
)
seasons = sorted(df["Season"].dropna().unique().tolist())
strategies = sorted(
    df["Retention_Strategy"].dropna().unique().tolist()
)

selected_regions = st.sidebar.multiselect(
    "Region", regions, default=regions
)
selected_categories = st.sidebar.multiselect(
    "Category", categories, default=categories
)
selected_seasons = st.sidebar.multiselect(
    "Season", seasons, default=seasons
)
selected_strategies = st.sidebar.multiselect(
    "Retention strategy", strategies, default=strategies
)

filtered = df[
    df["Region"].isin(selected_regions)
    & df["Most_Frequent_Category"].isin(selected_categories)
    & df["Season"].isin(selected_seasons)
    & df["Retention_Strategy"].isin(selected_strategies)
].copy()

if filtered.empty:
    st.warning("No records match these filters. Change the sidebar filters.")
    st.stop()

# --------------------------------------------------
# 4. ADVANCED ANALYTICS
# --------------------------------------------------
st.header("1. Business Overview")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Customer records", f"{len(filtered):,}")
k2.metric(
    "Average order value",
    f"{filtered['Average_Order_Value'].mean():,.2f}"
)
k3.metric(
    "Average lifetime value",
    f"{filtered['Lifetime_Value'].mean():,.2f}"
)
k4.metric(
    "Average churn probability",
    f"{filtered['Churn_Probability'].mean():.1%}"
)

st.caption(
    "These figures describe the supplied dataset. "
    "Lifetime value is shown in dataset units; currency is not specified."
)

left, right = st.columns(2)

with left:
    st.subheader("Purchase frequency by category")
    chart_data = (
        filtered.groupby("Most_Frequent_Category")["Purchase_Frequency"]
        .mean()
        .sort_values(ascending=False)
    )
    st.bar_chart(chart_data)

with right:
    st.subheader("Average lifetime value by region")
    chart_data = (
        filtered.groupby("Region")["Lifetime_Value"]
        .mean()
        .sort_values(ascending=False)
    )
    st.bar_chart(chart_data)

left, right = st.columns(2)

with left:
    st.subheader("Retention strategy distribution")
    st.bar_chart(filtered["Retention_Strategy"].value_counts())

with right:
    st.subheader("Seasonal customer records")
    st.bar_chart(filtered["Season"].value_counts())

st.subheader("Purchase frequency vs lifetime value")
st.scatter_chart(
    filtered,
    x="Purchase_Frequency",
    y="Lifetime_Value",
    color="Most_Frequent_Category"
)

# --------------------------------------------------
# 5. CHURN RISK ANALYSIS
# --------------------------------------------------
st.header("2. Churn Risk Analysis")

st.write(
    "Risk bands below are illustrative thresholds applied to the existing "
    "Churn_Probability column. They are not verified churn outcomes."
)

def risk_band(probability):
    if pd.isna(probability):
        return "Unknown"
    if probability < 0.40:
        return "Low"
    if probability < 0.70:
        return "Medium"
    return "High"

filtered["Churn_Risk_Band"] = filtered["Churn_Probability"].apply(risk_band)

risk_order = ["Low", "Medium", "High", "Unknown"]
risk_counts = (
    filtered["Churn_Risk_Band"]
    .value_counts()
    .reindex(risk_order, fill_value=0)
)

r1, r2, r3 = st.columns(3)
r1.metric("Low risk", int(risk_counts["Low"]))
r2.metric("Medium risk", int(risk_counts["Medium"]))
r3.metric("High risk", int(risk_counts["High"]))

st.bar_chart(risk_counts)

high_risk = filtered[
    filtered["Churn_Risk_Band"] == "High"
].sort_values("Churn_Probability", ascending=False)

st.subheader("High-risk customer review list")
st.dataframe(
    high_risk[
        [
            "Customer_ID",
            "Region",
            "Purchase_Frequency",
            "Average_Order_Value",
            "Lifetime_Value",
            "Churn_Probability",
            "Retention_Strategy",
        ]
    ],
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# 6. CUSTOMER SEGMENTATION WITH K-MEANS
# --------------------------------------------------
st.header("3. Customer Segmentation (K-Means)")

st.write(
    "K-Means groups records with similar numeric purchasing characteristics. "
    "Cluster names are descriptive, not pre-existing labels."
)

cluster_features = [
    "Purchase_Frequency",
    "Average_Order_Value",
    "Time_Between_Purchases",
    "Lifetime_Value",
    "Churn_Probability",
]

segment_data = filtered[cluster_features].apply(
    pd.to_numeric, errors="coerce"
)

valid_mask = segment_data.notna().all(axis=1)
valid_data = segment_data.loc[valid_mask].copy()

if len(valid_data) < 3:
    st.warning("Not enough complete records for customer segmentation.")
else:
    max_clusters = min(5, len(valid_data) - 1)
    cluster_count = st.slider(
        "Number of customer segments",
        min_value=2,
        max_value=max(2, max_clusters),
        value=min(3, max_clusters)
    )

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(valid_data)

    model = KMeans(
        n_clusters=cluster_count,
        random_state=42,
        n_init=10
    )
    labels = model.fit_predict(scaled_features)

    segment_results = filtered.loc[valid_data.index].copy()
    segment_results["Customer_Segment"] = [
        f"Segment {label + 1}" for label in labels
    ]

    st.subheader("Segment sizes")
    st.bar_chart(segment_results["Customer_Segment"].value_counts())

    st.subheader("Segment characteristics")
    segment_summary = segment_results.groupby("Customer_Segment")[
        cluster_features
    ].mean().round(2)
    st.dataframe(segment_summary, use_container_width=True)

    if len(valid_data) > cluster_count:
        sample_size = min(2000, len(scaled_features))
        sample_frame = pd.DataFrame(scaled_features).sample(
            n=sample_size, random_state=42
        )
        if sample_size > cluster_count:
            score = silhouette_score(
                sample_frame,
                model.fit_predict(sample_frame)
            )
            st.metric("Sample silhouette score", f"{score:.3f}")
            st.caption(
                "This is a sample-based clustering diagnostic, not a "
                "business success or churn-prediction score."
            )

    # Keep segmentation available for export
    filtered = filtered.join(
        segment_results[["Customer_Segment"]],
        how="left"
    )

# --------------------------------------------------
# 7. RETENTION RECOMMENDATIONS
# --------------------------------------------------
st.header("4. Retention Recommendations")

def recommend_action(row):
    probability = row["Churn_Probability"]
    value = row["Lifetime_Value"]
    frequency = row["Purchase_Frequency"]

    if pd.isna(probability):
        return "Review customer data"

    if probability >= 0.70 and value >= filtered["Lifetime_Value"].median():
        return "Priority personal outreach and loyalty review"
    if probability >= 0.70:
        return "Re-engagement message or relevant offer"
    if probability >= 0.40:
        return "Personalized reminder and category recommendations"
    if frequency >= filtered["Purchase_Frequency"].median():
        return "Reward repeat purchases and loyalty"
    return "Share relevant products and monitor engagement"

filtered["Recommended_Action"] = filtered.apply(
    recommend_action, axis=1
)

st.caption(
    "Recommendations are rule-based examples for review, not automated "
    "marketing actions. Validate thresholds and business policies before use."
)

st.dataframe(
    filtered[
        [
            "Customer_ID",
            "Purchase_Frequency",
            "Lifetime_Value",
            "Churn_Probability",
            "Churn_Risk_Band",
            "Retention_Strategy",
            "Recommended_Action",
        ]
    ].sort_values("Churn_Probability", ascending=False),
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# 8. MACHINE LEARNING STATUS
# --------------------------------------------------
st.header("5. Machine Learning Assessment")

st.info(
    "The dataset contains Churn_Probability, but the inspected columns do "
    "not contain a confirmed actual churn outcome label. Therefore, this "
    "dashboard does not train or claim a supervised churn prediction model."
)

st.write(
    "K-Means segmentation above is an unsupervised machine learning method. "
    "It groups records based on the selected numeric features without "
    "requiring a churn label."
)

st.write(
    "To train and evaluate a genuine churn classifier, obtain a documented "
    "target such as Churned = 0/1 and confirm what each row represents."
)

# --------------------------------------------------
# 9. SEARCH, EXPORT AND SAVE SUMMARY
# --------------------------------------------------
st.header("6. Search and Export")

customer_search = st.text_input("Search by Customer ID")
display_data = filtered.copy()

if customer_search.strip():
    display_data = display_data[
        display_data["Customer_ID"].str.contains(
            customer_search.strip(),
            case=False,
            na=False
        )
    ]

st.dataframe(display_data, use_container_width=True, hide_index=True)

st.download_button(
    "Download filtered customer report (CSV)",
    data=display_data.to_csv(index=False).encode("utf-8"),
    file_name="customer_retention_report.csv",
    mime="text/csv"
)

summary_text = (
    f"Filtered records: {len(filtered)}\n"
    f"Average order value: {filtered['Average_Order_Value'].mean():.2f}\n"
    f"Average lifetime value: {filtered['Lifetime_Value'].mean():.2f}\n"
    f"Average churn probability: "
    f"{filtered['Churn_Probability'].mean():.4f}\n"
    f"High-risk records: {int(risk_counts['High'])}\n"
)

(RESULTS_DIR / "dashboard_summary.txt").write_text(
    summary_text, encoding="utf-8"
)

st.success("Dashboard loaded. Use the sidebar to explore different customer groups.")
