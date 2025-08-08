#!/usr/bin/env python3
# pyspark_dt_basic.py

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from pyspark.mllib.evaluation import MulticlassMetrics

def main():
    spark = (SparkSession.builder
             .appName("DT_FraudDetection")
             .master("local[*]")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # Veriyi oku ve hazırla
    df = (spark.read
          .option("header", True)
          .option("inferSchema", True)
          .csv("creditcard.csv"))
    feat_cols = [f"V{i}" for i in range(1,29)] + ["Amount"]
    assembler = VectorAssembler(inputCols=feat_cols, outputCol="rawFeatures")
    df_feat = assembler.transform(df)
    scaler = StandardScaler(inputCol="rawFeatures",
                            outputCol="features",
                            withMean=True, withStd=True)
    df_scaled = scaler.fit(df_feat).transform(df_feat)
    df_final = df_scaled.select(
        col("features"),
        col("Class").cast(DoubleType()).alias("label")
    )

    # Train/Test ayır
    train_df, test_df = df_final.randomSplit([0.8,0.2], seed=42)

    # Decision Tree
    dt = DecisionTreeClassifier(
        featuresCol="features",
        labelCol="label",
        maxDepth=5,      # basit bir ağaç
        maxBins=32,
        seed=42
    )
    model = dt.fit(train_df)
    preds = model.transform(test_df)

    # Metrikleri hesapla
    rdd = preds.select("prediction","label").rdd.map(lambda r:(r[0],r[1]))
    metrics = MulticlassMetrics(rdd)
    print("=== Decision Tree Metrics ===")
    print(f"Precision (fraud=1): {metrics.precision(1.0):.4f}")
    print(f"Recall    (fraud=1): {metrics.recall(1.0):.4f}")
    print(f"F1-Score  (fraud=1): {metrics.fMeasure(1.0):.4f}")
    print(f"Accuracy overall:    {metrics.accuracy:.4f}")

    spark.stop()

if __name__=="__main__":
    main()
