#!/usr/bin/env python3
# pyspark_pca_anomaly_fixed.py

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType
import numpy as np
from pyspark.mllib.evaluation import MulticlassMetrics

def main():
    spark = (SparkSession.builder
             .appName("PCA_AnomalyDetection_Fixed")
             .master("local[*]")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # 1) Veri oku ve ölçekle
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

    # 2) PCA uygula
    k = 5
    pca = PCA(k=k, inputCol="features", outputCol="pcaFeatures")
    pca_model = pca.fit(df_scaled)
    df_pca = pca_model.transform(df_scaled)

    # 3) Spark Vectors → Python list
    df_arr = (df_pca
        .withColumn("featArr", vector_to_array(col("features")))
        .withColumn("pcaArr",  vector_to_array(col("pcaFeatures")))
        .select("featArr", "pcaArr", col("Class").alias("label"))
    )

    # 4) PC matrisi
    pc = pca_model.pc.toArray()  # shape: (numFeatures, k)

    # 5) Rekonstrüksiyon hatası UDF’i
    def rec_error(pca_vals, feat_vals):
        pca_vec = np.array(pca_vals)          # (k,)
        orig_vec = np.array(feat_vals)        # (numFeatures,)
        recon = pc.dot(pca_vec)               # (numFeatures,)
        return float(np.linalg.norm(orig_vec - recon))

    err_udf = udf(rec_error, DoubleType())

    df_err = df_arr.withColumn("reconError",
                               err_udf(col("pcaArr"), col("featArr")))

    # 6) Eşik belirle (%99 persentil)
    errors = np.array(df_err.select("reconError")
                       .rdd.map(lambda r: r[0]).collect())
    thresh = np.percentile(errors, 99.5)

    # 7) Tahmin sütunu
    df_pred = df_err.withColumn(
        "prediction",
        (col("reconError") >= float(thresh)).cast("double")
    )

    # 8) Metrikleri hesapla
    rdd = df_pred.select("prediction","label") \
                  .rdd.map(lambda r: (r[0], float(r[1])))
    metrics = MulticlassMetrics(rdd)

    print("=== PCA Reconstruction Error Anomaly Detection ===")
    print(f"Threshold (99th pctile): {thresh:.4f}")
    print(f"Precision (fraud=1):     {metrics.precision(1.0):.4f}")
    print(f"Recall    (fraud=1):     {metrics.recall(1.0):.4f}")
    print(f"F1-Score  (fraud=1):     {metrics.fMeasure(1.0):.4f}")
    print(f"Accuracy overall:        {metrics.accuracy:.4f}")

    spark.stop()

if __name__ == "__main__":
    main()
