# Customer-segmentation-and-purchase-analysis-pipeline

An end-to-end data mining pipeline built in PySpark that analyses UK e-commerce transaction data. It cleans the data, builds customer-level RFM features, and applies three data mining techniques: clustering, classification and association rule mining.

## Dataset
This project uses the Online Retail II dataset from the UCI Machine Learning Repository.

Download it here: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci

Place `online_retail_II.xlsx` in the same folder as the script before running.

Pipeline
Start Spark session: local mode with 4 GB driver memory.
Load data: reads the Excel file with pandas, standardises column names and converts to a Spark DataFrame.
Pre-processing: drops rows with missing key fields, removes cancelled invoices and non-positive quantities or prices, and engineers TotalSpend, Month, Hour and DayOfWeek.
RFM features: for each customer, calculates Recency, Frequency and Monetary value, plus average order value and average items per order.
Exploratory analysis: top countries by revenue, best-selling products, monthly revenue trend and RFM distributions.
Feature preparation: assembles and standardises features using a Spark ML Pipeline.
Clustering: tests k = 2 to 8 using silhouette score, fits the final K-Means model and profiles each segment.
Classification: labels customers in the top 25% of spend as "high value", then trains and compares a Decision Tree and a Random Forest (accuracy, F1, confusion matrix, feature importance).
Association rule mining: builds baskets from the 100 most popular products and runs FP-Growth (min support 0.02, min confidence 0.3), ranking rules by confidence and lift.
Summary: prints key results and saves all charts.

Tech stack:

Python
PySpark (Spark SQL and MLlib)
pandas
matplotlib
seaborn

Getting Started:
Prerequisites
Python 3.8+
Java 8 or later (required by PySpark)
Installation
bash
pip install pyspark pandas matplotlib seaborn openpyxl
Run
bash
python DataminingCode.py

The script reads the dataset from the same folder, prints results to the console and saves charts as PNG files. Loading the Excel file can take a minute or two.

Output Files:
File	Description
eda_monthly_revenue.png	Monthly revenue trend
eda_rfm_distributions.png	Distributions of Recency, Frequency and Monetary value
kmeans_silhouette.png	Silhouette score for each value of k
kmeans_cluster_scatter.png	Customer segments (Recency vs Monetary)
classification_feature_importance.png	Random Forest feature importances
