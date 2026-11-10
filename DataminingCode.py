
# If needed, run in your terminal before running the script:
#   pip install pyspark pandas matplotlib seaborn openpyxl
#
# DATASET:
#   The dataset is already provided in the folder, but if some reason it doesn't work, you can download it from: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci


import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType

from pyspark.ml.feature import (
    VectorAssembler, StandardScaler,
    StringIndexer, OneHotEncoder
)
from pyspark.ml.classification import (
    DecisionTreeClassifier, RandomForestClassifier
)
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator, ClusteringEvaluator
)
from pyspark.ml.fpm import FPGrowth
from pyspark.ml import Pipeline


# STEP 1: Start the pyspark session



print("STEP 1: Starting PySpark Session")


spark = SparkSession.builder \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print("Spark session started.\n")


# STEP 2: Load the data via pandas



print("STEP 2: Loading Data")


FILE_PATH = "online_retail_II.xlsx"

if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(
        "online_retail_II.xlsx not found.\n"
        "Download from: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci\n"
        "and place it in the same folder as this script."
    )

print("Reading Excel file (this may take a minute)...")
retail_pd = pd.read_excel(FILE_PATH, sheet_name=0)

# Standardise column names
retail_pd.columns = [c.strip().replace(" ", "_").replace("-", "_")
                       for c in retail_pd.columns]

print(f"Loaded {len(retail_pd):,} rows.")
print(f"Columns: {list(retail_pd.columns)}\n")

# Convert to Spark DataFrame
df = spark.createDataFrame(retail_pd)
df.printSchema()


# STEP 3: Preprocessing



print("STEP 3: Pre-processing")


# Drop rows with missing CustomerID or Description 
before = df.count()
df = df.dropna(subset=["Customer_ID", "Description", "StockCode"])
after = df.count()
print(f"Dropped {before - after:,} rows with nulls in key columns.")
print(f"Remaining rows: {after:,}")

# Remove cancelled transactions (InvoiceNo starts with 'C')
df = df.filter(~F.col("Invoice").cast(StringType()).startswith("C"))
print(f"After removing cancellations: {df.count():,} rows")

# Remove rows with non-positive Quantity or Price
df = df.filter((F.col("Quantity") > 0) & (F.col("Price") > 0))
print(f"After removing invalid quantities/prices: {df.count():,} rows")

# Cast numeric columns
df = df.withColumn("Quantity",    F.col("Quantity").cast(DoubleType()))
df = df.withColumn("Price",       F.col("Price").cast(DoubleType()))
df = df.withColumn("Customer_ID", F.col("Customer_ID").cast(DoubleType()))

# Add TotalSpend column = Quantity * Price
df = df.withColumn("TotalSpend", F.col("Quantity") * F.col("Price")) 

# Extract month and hour from InvoiceDate 
df = df.withColumn("InvoiceDate", F.to_timestamp("InvoiceDate"))
df = df.withColumn("Month", F.month("InvoiceDate"))
df = df.withColumn("Hour",  F.hour("InvoiceDate"))
df = df.withColumn("DayOfWeek", F.dayofweek("InvoiceDate"))

print(f"\nFinal cleaned dataset: {df.count():,} rows\n")


# STEP 4: Build RFM model (Recency, Frequency, Monetary)




print("STEP 4: Building RFM Customer Features")


# Reference date = day after the last transaction in the dataset
max_date = df.agg(F.max("InvoiceDate")).collect()[0][0]
ref_date  = max_date

# Recency  = days since customer's last purchase (lower = more recent)
# Frequency = number of purchases
# Monetary  = total spend
rfm = df.groupBy("Customer_ID").agg(
    F.datediff(F.lit(ref_date), F.max("InvoiceDate")).alias("Recency"),
    F.countDistinct("Invoice").alias("Frequency"),
    F.round(F.sum("TotalSpend"), 2).alias("Monetary")
)

# Also add average order value and average items per order
avg_order = df.groupBy("Customer_ID", "Invoice").agg(
    F.sum("TotalSpend").alias("OrderValue"),
    F.sum("Quantity").alias("OrderItems")
)
avg_stats = avg_order.groupBy("Customer_ID").agg(
    F.round(F.avg("OrderValue"), 2).alias("AvgOrderValue"),
    F.round(F.avg("OrderItems"), 2).alias("AvgItemsPerOrder")
)

rfm = rfm.join(avg_stats, on="Customer_ID", how="left")

print(f"RFM table built for {rfm.count():,} unique customers.")
print("\nRFM Sample:")
rfm.show(10)
print("RFM Summary Statistics:")
rfm.describe().show()


# STEP 5: Exploratory data analysis


print("STEP 5: Exploratory Data Analysis")


# Top 10 countries by revenue
print("Top 10 Countries by Revenue:")
df.groupBy("Country").agg(F.round(F.sum("TotalSpend"), 2).alias("Revenue")) \
  .orderBy(F.desc("Revenue")).show(10)

# Top 10 best-selling products
print("Top 10 Products by Quantity Sold:")
df.groupBy("Description").agg(F.sum("Quantity").alias("TotalQty")) \
  .orderBy(F.desc("TotalQty")).show(10)

# Monthly revenue trend
monthly = df.groupBy("Month") \
            .agg(F.round(F.sum("TotalSpend"), 2).alias("Revenue")) \
            .orderBy("Month").toPandas()

plt.figure(figsize=(9, 4))
sns.barplot(x="Month", y="Revenue", data=monthly, palette="Blues_d")
plt.title("Monthly Revenue (All Years Combined)")
plt.xlabel("Month")
plt.ylabel("Revenue (£)")
plt.tight_layout()
plt.savefig("eda_monthly_revenue.png")
print("Saved: eda_monthly_revenue.png")
plt.close()

# RFM distributions
rfm_pd = rfm.sample(fraction=0.5, seed=42).toPandas()

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, col, colour in zip(axes,
                            ["Recency", "Frequency", "Monetary"],
                            ["steelblue", "seagreen", "tomato"]):
    ax.hist(rfm_pd[col].clip(upper=rfm_pd[col].quantile(0.95)),
            bins=40, color=colour, edgecolor="white")
    ax.set_title(f"{col} Distribution")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("eda_rfm_distributions.png")
print("Saved: eda_rfm_distributions.png\n")
plt.close()


# STEP 6: Prepare RFM model for clustering


print("STEP 6: Preparing RFM Features for Clustering")


rfm_features = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "AvgItemsPerOrder"]

# Drop any rows with null field in RFM features
rfm_clean = rfm.dropna(subset=rfm_features)
print(f"Customers after dropping nulls: {rfm_clean.count():,}")

# Assemble and scale features
assembler_rfm = VectorAssembler(inputCols=rfm_features, outputCol="features_raw")
scaler_rfm    = StandardScaler(inputCol="features_raw", outputCol="features",
                                withMean=True, withStd=True)

prep_pipeline = Pipeline(stages=[assembler_rfm, scaler_rfm])
prep_model    = prep_pipeline.fit(rfm_clean)
rfm_scaled    = prep_model.transform(rfm_clean)

print("RFM features assembled and scaled.\n")


# STEP 7: Clustering — K-means to separate customers into segments based on how much they spend


print("STEP 7: Clustering — K-Means Customer Segmentation")


print("Finding optimal k (2-8) using Silhouette Score")
silhouette_scores = []
k_values = range(2, 9)
sample_scaled = rfm_scaled.sample(fraction=0.4, seed=42)

for k in k_values:
    km = KMeans(featuresCol="features", k=k, seed=42, maxIter=30)
    km_model = km.fit(sample_scaled)
    preds    = km_model.transform(sample_scaled)
    score    = ClusteringEvaluator(featuresCol="features").evaluate(preds)
    silhouette_scores.append(score)
    print(f"  k={k}  Silhouette: {score:.4f}")

best_k = list(k_values)[silhouette_scores.index(max(silhouette_scores))]
print(f"\nBest k: {best_k}")

# Final model on full data
km_final       = KMeans(featuresCol="features", k=best_k, seed=42, maxIter=100)
km_final_model = km_final.fit(rfm_scaled)
rfm_clustered  = km_final_model.transform(rfm_scaled)

final_sil = ClusteringEvaluator(featuresCol="features").evaluate(rfm_clustered)
print(f"Final Silhouette Score (k={best_k}): {final_sil:.4f}")

# Cluster profile: mean RFM values per segment
print(f"\nCluster Profiles (mean RFM values per segment):")
rfm_clustered.groupBy("prediction").agg(
    F.round(F.avg("Recency"),          1).alias("Avg_Recency"),
    F.round(F.avg("Frequency"),        1).alias("Avg_Frequency"),
    F.round(F.avg("Monetary"),         1).alias("Avg_Monetary"),
    F.round(F.avg("AvgOrderValue"),    1).alias("Avg_OrderValue"),
    F.count("*").alias("CustomerCount")
).orderBy("prediction").show()

# --- Plot: Silhouette Scores ---
plt.figure(figsize=(7, 4))
plt.plot(list(k_values), silhouette_scores, marker="o", color="steelblue")
plt.title("K-Means — Silhouette Score vs. k (Customer Segmentation)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.xticks(list(k_values))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("kmeans_silhouette.png")
print("Saved: kmeans_silhouette.png")
plt.close()

# --- Plot: Cluster scatter (Recency vs Monetary) ---
cluster_pd = rfm_clustered.select("Recency", "Monetary", "prediction") \
                           .sample(fraction=0.5, seed=42).toPandas()
lbls = cluster_pd["prediction"].astype(int)
colours = ["steelblue", "tomato", "seagreen", "orange", "purple", "brown", "pink"]

plt.figure(figsize=(8, 5))
for seg in sorted(lbls.unique()):
    mask = lbls == seg
    plt.scatter(cluster_pd["Recency"][mask], cluster_pd["Monetary"][mask],
                c=colours[seg % len(colours)],
                label=f"Segment {seg}", alpha=0.4, s=15)
plt.title(f"K-Means Customer Segments (k={best_k}) — Recency vs Monetary")
plt.xlabel("Recency (days)")
plt.ylabel("Monetary (£)")
plt.legend()
plt.tight_layout()
plt.savefig("kmeans_cluster_scatter.png")
print("Saved: kmeans_cluster_scatter.png\n")
plt.close()


# STEP 8: Classification — This is used to predict customers who are of high-value

# Label customers as High Value (1) if their Monetary spend is above the
# 75th percentile, else Standard (0). Then train classifiers to predict
# high-value status from Recency, Frequency, and order behaviour features.


print("STEP 8: Classification — High-Value Customer Prediction")


monetary_75 = rfm_clean.approxQuantile("Monetary", [0.75], 0.01)[0]
print(f"75th percentile Monetary threshold: £{monetary_75:.2f}")

rfm_labelled = rfm_clean.withColumn(
    "HighValue",
    F.when(F.col("Monetary") >= monetary_75, 1.0).otherwise(0.0)
)

print("Class distribution:")
rfm_labelled.groupBy("HighValue").count().orderBy("HighValue").show()

clf_features = ["Recency", "Frequency", "AvgOrderValue", "AvgItemsPerOrder"]
train_df, test_df = rfm_labelled.randomSplit([0.8, 0.2], seed=42)
print(f"Train: {train_df.count():,}  |  Test: {test_df.count():,}\n")

assembler_clf = VectorAssembler(inputCols=clf_features, outputCol="features_raw")
scaler_clf    = StandardScaler(inputCol="features_raw", outputCol="features",
                                withMean=True, withStd=True)

# Decision Tree 
print("--- Decision Tree ---")
dt = DecisionTreeClassifier(
    featuresCol="features", labelCol="HighValue", maxDepth=10, seed=42
)
dt_pipeline = Pipeline(stages=[assembler_clf, scaler_clf, dt])
dt_model    = dt_pipeline.fit(train_df)
dt_preds    = dt_model.transform(test_df)

acc_eval = MulticlassClassificationEvaluator(labelCol="HighValue", metricName="accuracy")
f1_eval  = MulticlassClassificationEvaluator(labelCol="HighValue", metricName="f1")
dt_acc   = acc_eval.evaluate(dt_preds)
dt_f1    = f1_eval.evaluate(dt_preds)
print(f"  Accuracy: {dt_acc:.4f}  |  F1: {dt_f1:.4f}")

# --- Random Forest ---
print("\n--- Random Forest ---")
rf = RandomForestClassifier(
    featuresCol="features", labelCol="HighValue",
    numTrees=50, maxDepth=10, seed=42
)
rf_pipeline = Pipeline(stages=[assembler_clf, scaler_clf, rf])
rf_model    = rf_pipeline.fit(train_df)
rf_preds    = rf_model.transform(test_df)

rf_acc = acc_eval.evaluate(rf_preds)
rf_f1  = f1_eval.evaluate(rf_preds)
print(f"  Accuracy: {rf_acc:.4f}  |  F1: {rf_f1:.4f}")

print("\nRandom Forest — Confusion Matrix:")
rf_preds.groupBy("HighValue", "prediction") \
        .count().orderBy("HighValue", "prediction").show()

# Feature importances
importances = rf_model.stages[-1].featureImportances.toArray()
feat_imp_pd = pd.DataFrame({
    "Feature": clf_features,
    "Importance": importances
}).sort_values("Importance", ascending=False)

plt.figure(figsize=(7, 4))
sns.barplot(x="Importance", y="Feature", data=feat_imp_pd, palette="Blues_r")
plt.title("Random Forest — Feature Importances (High-Value Prediction)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("classification_feature_importance.png")
print("Saved: classification_feature_importance.png\n")
plt.close()

print("--- Classification Summary ---")
print(f"{'Model':<20} {'Accuracy':>10} {'F1':>10}")
print("-" * 42)
print(f"{'Decision Tree':<20} {dt_acc:>10.4f} {dt_f1:>10.4f}")
print(f"{'Random Forest':<20} {rf_acc:>10.4f} {rf_f1:>10.4f}")


# STEP 9: Association rule mining — used to find out what products are frequently bought together



print("STEP 9: Association Rule Mining — Market Basket Analysis")


print("Building basket (invoice -> list of products)...")

# Keeps the top 100 most popular products to keep FP-tree manageable
top_products = df.groupBy("Description") \
                 .agg(F.count("*").alias("cnt")) \
                 .orderBy(F.desc("cnt")) \
                 .limit(100) \
                 .select("Description") \
                 .rdd.flatMap(lambda x: x).collect()

top_products_set = set(top_products)

# Filter to top products only
df_top = df.filter(F.col("Description").isin(top_products))

# Group by Invoice -> list of products purchased
baskets = df_top.groupBy("Invoice") \
                .agg(F.collect_set("Description").alias("items")) \
                .filter(F.size("items") >= 2)  # at least 2 items per basket

print(f"Number of baskets (invoices with 2+ top products): {baskets.count():,}")
baskets.show(5, truncate=True)

# Run FP-Growth algorithm
fp_model = FPGrowth(
    itemsCol="items",
    minSupport=0.02,
    minConfidence=0.3
).fit(baskets)

print(f"\nTop 15 Frequent Itemsets (most commonly bought together):")
fp_model.freqItemsets \
        .filter(F.size("items") >= 2) \
        .orderBy(F.desc("freq")) \
        .show(15, truncate=False)

print("\nTop 15 Association Rules (by confidence):")
fp_model.associationRules \
        .orderBy(F.desc("confidence")) \
        .show(15, truncate=False)

print("\nTop 10 Rules by Lift (strongest associations):")
fp_model.associationRules \
        .orderBy(F.desc("lift")) \
        .show(10, truncate=False)


# STEP 10: Summary



print("STEP 10: Final Summary")


total_tx   = df.count()
total_cust = rfm_clean.count()

print(f"""
DATASET
  Total transactions (cleaned) : {total_tx:,}
  Unique customers             : {total_cust:,}

CLUSTERING (K-Means Customer Segmentation)
  Optimal k          : {best_k}
  Silhouette Score   : {final_sil:.4f}

CLASSIFICATION (High-Value Customer Prediction)
  Decision Tree  — Accuracy: {dt_acc:.4f}  F1: {dt_f1:.4f}
  Random Forest  — Accuracy: {rf_acc:.4f}  F1: {rf_f1:.4f}

ASSOCIATION RULES (Market Basket Analysis)
  Basket min support    : 0.02
  Min confidence        : 0.30
  

OUTPUT FILES GENERATED
  eda_monthly_revenue.png
  eda_rfm_distributions.png
  kmeans_silhouette.png
  kmeans_cluster_scatter.png
  classification_feature_importance.png
""")

print("Analysis complete.")
spark.stop()