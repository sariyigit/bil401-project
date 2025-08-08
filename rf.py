#!/usr/bin/env python3
# run_optimal_rf.py

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from pyspark.mllib.evaluation import MulticlassMetrics

def main():
    # 1) Spark başlat
    spark = (SparkSession.builder
             .appName("OptimalRF")
             .master("local[*]")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # 2) Veriyi oku ve hazırla
    df = (spark.read
          .option("header", True)
          .option("inferSchema", True)
          .csv("creditcard.csv"))

    feat_cols = [f"V{i}" for i in range(1,29)] + ["Amount"]
    assembler = VectorAssembler(inputCols=feat_cols, outputCol="rawFeatures")
    df_feat = assembler.transform(df)

    scaler = StandardScaler(
        inputCol="rawFeatures",
        outputCol="features",
        withMean=True,
        withStd=True
    )
    df_scaled = scaler.fit(df_feat).transform(df_feat)

    df_final = df_scaled.select(
        col("features"),
        col("Class").cast(DoubleType()).alias("label")
    )

    # 3) Eğitim/test böl (80/20)
    train_df, test_df = df_final.randomSplit([0.8, 0.2], seed=42)

    # 4) En iyi parametrelerle RandomForest
    #    Burayı tuning’den elde ettiğiniz değerlerle güncelleyin:
    best_num_trees = 500
    best_max_depth = 10
    best_feature_subset = "auto"

    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        predictionCol="prediction",
        probabilityCol="probability",
        numTrees=best_num_trees,
        maxDepth=best_max_depth,
        featureSubsetStrategy=best_feature_subset,
        seed=42
    )

    # 5) Modeli eğit ve test üzerinde değerlendir
    model = rf.fit(train_df)
    preds = model.transform(test_df)

    # 6) Precision / Recall / F1 / Accuracy
    rdd = preds.select("prediction", "label") \
               .rdd.map(lambda r: (r[0], r[1]))
    metrics = MulticlassMetrics(rdd)

    print("=== Optimal Random Forest Metrics ===")
    print(f"Parameters: numTrees={best_num_trees}, maxDepth={best_max_depth}, featureSubset='{best_feature_subset}'")
    print(f"Precision (fraud=1): {metrics.precision(1.0):.4f}")
    print(f"Recall    (fraud=1): {metrics.recall(1.0):.4f}")
    print(f"F1-Score  (fraud=1): {metrics.fMeasure(1.0):.4f}")
    print(f"Accuracy overall:    {metrics.accuracy:.4f}")

    spark.stop()

if __name__ == "__main__":
    main()
