
# Customer Purchase Behavior & Retention Strategy

A Python-based customer analytics project that explores purchasing patterns, customer lifetime value, churn-risk scores, and retention opportunities through an interactive Streamlit dashboard.

---

## 1. Project Overview

Businesses need to understand how customers purchase products, how customer value varies, and which customers may require additional engagement.

This project analyzes customer transaction-related records and provides an interactive dashboard for exploring customer behavior, segmenting customers using Machine Learning, reviewing churn-risk scores, and generating rule-based retention recommendations.

The dashboard helps users explore the supplied dataset and identify patterns that may support business decision-making.

## 2. Problem Statement

Businesses may find it difficult to identify valuable customer groups, understand purchasing patterns, review potential churn risks, and determine suitable retention actions from raw customer data.

The objective of this project is to transform customer data into understandable analytics through data processing, visualization, customer segmentation, risk analysis, and rule-based recommendations.

**Important:** The supplied dataset contains a `Churn_Probability` field, but the inspected columns do not include a confirmed actual churn outcome label. Therefore, this project does not claim to train a supervised churn prediction model.

## 3. Project Objectives

- Analyze customer purchasing behavior.
- Compare average purchase frequency across product categories.
- Explore average customer lifetime value across regions.
- Understand seasonal patterns and retention strategy distributions.
- Review the existing churn probability scores.
- Segment customers using K-Means clustering.
- Generate rule-based retention recommendations.
- Allow users to filter, search, view, and download customer records.
- Present the analysis through an interactive web dashboard.

## 4. Dataset Description

**Dataset file:** `sales_and_customer_insights.csv`

**Dataset size:** 10,000 rows and 15 columns, based on the initial inspection.

The dataset contains the following fields:

| Column | Description |
|---|---|
| `Customer_ID` | Customer identifier |
| `Product_ID` | Product identifier |
| `Transaction_ID` | Transaction identifier |
| `Purchase_Frequency` | Purchase frequency value |
| `Average_Order_Value` | Average order value |
| `Most_Frequent_Category` | Most frequent product category |
| `Time_Between_Purchases` | Time between purchases |
| `Region` | Customer region |
| `Churn_Probability` | Supplied churn probability score |
| `Lifetime_Value` | Customer lifetime value |
| `Launch_Date` | Launch date field |
| `Peak_Sales_Date` | Peak sales date field |
| `Season` | Seasonal category |
| `Preferred_Purchase_Times` | Preferred purchase time |
| `Retention_Strategy` | Retention strategy field |

The currency unit and complete business definitions of every field have not been independently confirmed.

## 5. Technologies Used

- **Python** — Core programming language.
- **Pandas** — Data loading, filtering, grouping, and analysis.
- **Matplotlib** — Data visualization library used in the project analysis.
- **Streamlit** — Interactive dashboard development.
- **Scikit-learn** — Machine Learning utilities.
- **StandardScaler** — Feature scaling before clustering.
- **K-Means** — Unsupervised customer segmentation.
- **Silhouette Score** — Sample-based clustering diagnostic.
- **Git and GitHub** — Version control and project hosting.

## 6. Project Workflow

The project follows this workflow:

1. **Data collection:** Load the supplied CSV dataset.
2. **Data validation:** Check required columns and confirm the dataset can be loaded.
3. **Data preprocessing:** Parse available date columns and prepare numeric features for analysis.
4. **Exploratory data analysis:** Calculate summary metrics and compare customer behavior across categories and regions.
5. **Data visualization:** Display charts for purchase frequency, lifetime value, seasonal records, retention strategies, and customer relationships.
6. **Churn-risk analysis:** Apply illustrative Low, Medium, and High thresholds to the existing `Churn_Probability` field.
7. **Customer segmentation:** Scale selected numerical features and apply K-Means clustering.
8. **Retention recommendations:** Generate rule-based suggestions using churn probability, lifetime value, and purchase frequency.
9. **Dashboard interaction:** Provide filters, customer search, customer tables, and CSV downloads.
10. **Output generation:** Save analytical outputs and a dashboard summary in the results directory.

### Workflow Diagram

```text
Customer Dataset (CSV)
         |
         v
Data Loading and Validation
         |
         v
Data Preparation and Analysis
         |
         v
Exploratory Data Visualization
         |
         +----------------------+
         |                      |
         v                      v
Churn-Risk Analysis      K-Means Segmentation
         |                      |
         v                      v
Retention Recommendations
         |
         v
Interactive Streamlit Dashboard
         |
         v
Customer Search and CSV Export
```

## 7. Dashboard Features

### Business Overview

- Total filtered customer records.
- Average order value.
- Average lifetime value.
- Average supplied churn probability.
- Purchase frequency by category.
- Average lifetime value by region.
- Retention strategy distribution.
- Seasonal customer record distribution.
- Purchase frequency versus lifetime value chart.

### Customer Segmentation

The project uses K-Means, an unsupervised Machine Learning algorithm, to group records based on selected numerical features:

- Purchase frequency
- Average order value
- Time between purchases
- Lifetime value
- Churn probability

StandardScaler is applied to help prevent differences in feature scales from dominating the clustering process.

Users can select the number of clusters and review segment sizes and average characteristics.

The resulting segment numbers are algorithm-generated labels. They do not automatically mean high-value, loyal, or high-risk customers.

### Churn-Risk Analysis

The dashboard applies these illustrative thresholds:

- **Low Risk:** Probability below 0.40.
- **Medium Risk:** Probability from 0.40 to below 0.70.
- **High Risk:** Probability of 0.70 or above.
- **Unknown:** Missing probability value.

The dashboard displays risk counts and a high-risk customer review table.

These thresholds are project rules, not independently validated business thresholds.

### Retention Recommendations

The dashboard generates suggestions based on existing data fields, including:

- Priority personal outreach and loyalty review.
- Re-engagement messages or relevant offers.
- Personalized reminders and category recommendations.
- Rewards for repeat purchases.
- Relevant product suggestions and engagement monitoring.

These are illustrative, rule-based recommendations. They do not automatically contact customers, and their actual effectiveness has not been measured.

### Search and Export

Users can:

- Filter records by region, category, season, and retention strategy.
- Search customer IDs.
- View matching customer records.
- Download filtered customer data as CSV.
- Download customer segmentation results through the dashboard.

## 8. Machine Learning Approach

### K-Means Clustering

K-Means is used for unsupervised customer segmentation.

The algorithm groups records according to similarities in the selected numerical features without requiring predefined customer segment labels.

The project uses feature scaling before clustering and reports a sample-based Silhouette Score as a clustering diagnostic.

The score is not a measure of business success or churn prediction accuracy.

### Supervised Churn Prediction

A supervised churn classifier is not implemented in this project.

The inspected dataset contains `Churn_Probability`, but no confirmed actual churn outcome column was identified. A genuine supervised churn prediction model would require a documented target, such as `Churned = 0/1`, and an appropriate evaluation process.

## 9. Project Structure

```text
customer purchase retention/
|
|-- app.py
|-- requirements.txt
|-- .gitignore
|-- README.md
|
|-- data/
|   |-- sales_and_customer_insights.csv
|
|-- src/
|   |-- data_analysis.py
|
|-- results/
|   |-- graphs/
|   |-- data_summary.txt
|   |-- dashboard_summary.txt
|
|-- .venv/
```

Generated output files may vary depending on which analysis scripts have been run.

The `.venv` folder contains the local Python environment and should not be uploaded to GitHub.

## 10. Installation and Setup

### Prerequisites

- Python
- Visual Studio Code or another Python development environment
- Git, if uploading to GitHub
- The supplied CSV dataset

### Step 1: Clone the Repository

Replace the example URL with the actual GitHub repository URL.

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd "customer purchase retention"
```

Alternatively, download the repository ZIP and extract it.

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 3: Activate the Environment on Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use a session-only policy change:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 5: Start the Dashboard

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal, usually:

`http://localhost:8501`

Make sure `data/sales_and_customer_insights.csv` exists before starting the application.

## 11. Live Dashboard

**Streamlit App:** Add the live Streamlit URL for this Customer Purchase Behavior & Retention Strategy project here.

The live URL must belong to this project. Do not use a URL belonging to a different project.

## 12. Results and Outputs

The project supports:

- Interactive business analytics.
- Category and regional comparisons.
- Churn-risk group counts.
- K-Means segment summaries.
- Rule-based retention suggestions.
- Downloadable filtered customer reports.
- Analytical summary and graph files generated by the analysis script.

Actual clustering results and business outcomes depend on the dataset and selected filters. No measured improvement in customer retention is claimed.

## 13. Limitations

- The dataset's currency unit is unspecified.
- The actual meaning and calibration of `Churn_Probability` have not been independently verified.
- Churn-risk thresholds are illustrative.
- Retention recommendations are rule-based, not experimentally validated.
- K-Means segments are exploratory groups, not verified customer types.
- A supervised churn classifier is not included because a confirmed target label was not identified in the inspected columns.
- The dashboard does not demonstrate causal improvement in retention or revenue.

## 14. Future Enhancements

Potential future improvements include:

- Validating the dataset definitions and probability scores.
- Adding confirmed churn outcome labels if reliable data becomes available.
- Evaluating a supervised churn classifier when suitable labels exist.
- Improving segment interpretation and business reporting.
- Measuring retention campaign outcomes with appropriate data.
- Adding automated tests and dashboard deployment documentation.

## 15. Author

## 15. Author

**Name:** M. Madhan Mohan Reddy

**Project:** Customer Purchase Behavior & Retention Strategy

**Technologies:** Python, Pandas, Streamlit, Scikit-learn, K-Means Clustering

**Purpose:** Customer analytics, customer segmentation, churn-risk analysis, and retention strategy exploration.

**GitHub:** https://github.com/madhanmohanm83


## 16. Disclaimer

This project is intended for educational, analytical, and portfolio demonstration purposes. Its recommendations should be reviewed before being used for real customer decisions. No claim is made that the proposed actions have improved customer retention.
