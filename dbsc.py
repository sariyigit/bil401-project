#!/usr/bin/env python3
# dbsc.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, sum as spark_sum
import dbscan
from scipy.spatial import distance

def main():
    # SparkSession başlat
    spark = (SparkSession.builder
             .appName("DBSCAN Fraud Detection")
             .master("local[*]")
             .config("spark.driver.host", "127.0.0.1")
             .getOrCreate())
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    # 1) Ham CSV'yi oku, header çıkar, split et
    rdd = sc.textFile("creditcard.csv")
    header = rdd.first()
    data = rdd.filter(lambda l: l != header).map(lambda l: l.split(","))

    # 2) DataFrame[id, value] hazırla (V1..V28 + Amount)
    pts_df = (data
              .zipWithIndex()
              .map(lambda ri: (
                  ri[1],
                  [float(x) for x in ri[0][1:29]] + [float(ri[0][29])]
              ))
              .toDF(["id", "value"])
              .repartition(20))

    # 3) Aynı zipWithIndex ile gerçek label'ları da al
    labels_df = (data
                 .zipWithIndex()
                 .map(lambda ri: (ri[1], int(float(ri[0][29]))))
                 .toDF(["id", "label"]))

    # 4) DBSCAN parametreleri
    eps, minPts, maxPerPartition = 0.3, 10, 2

    # 5) DBSCAN'i çalıştır
    result = dbscan.process(
        spark, pts_df,
        eps,
        minPts,
        distance.euclidean,
        maxPerPartition,
        "dbscan_checkpoint"
    )
    # result: DataFrame[id, component, core_point]

    # 6) prediction ekle: component == -1 ise outlier=1, aksi=0
    pred = result.withColumn(
        "prediction",
        when(col("component") == -1, 1).otherwise(0)
    )

    # 7) Örnek göster
    print("=== Aykırı Noktalar (prediction = 1) Örnekleri ===")
    pred.filter(col("prediction") == 1).show(10, truncate=False)

    # 8) Gerçek label'larla birleştir
    df = pred.join(labels_df, on="id")

    # 9) Confusion matrix bileşenlerini topla
    stats = df.select(
        spark_sum(when((col("prediction")==1)&(col("label")==1),1).otherwise(0)).alias("TP"),
        spark_sum(when((col("prediction")==1)&(col("label")==0),1).otherwise(0)).alias("FP"),
        spark_sum(when((col("prediction")==0)&(col("label")==1),1).otherwise(0)).alias("FN"),
        spark_sum(when((col("prediction")==0)&(col("label")==0),1).otherwise(0)).alias("TN")
    ).collect()[0]

    TP, FP, FN, TN = stats["TP"], stats["FP"], stats["FN"], stats["TN"]

    # 10) Precision, Recall, F1, Accuracy hesapla
    precision = TP / float(TP + FP) if TP + FP > 0 else 0.0
    recall    = TP / float(TP + FN) if TP + FN > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (TP + TN) / float(TP + TN + FP + FN)

    # 11) Metirkleri bas
    print("\n=== Değerlendirme Metrikleri ===")
    print(f"True Positives:  {TP}")
    print(f"False Positives: {FP}")
    print(f"False Negatives: {FN}")
    print(f"True Negatives:  {TN}\n")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}\n")

    spark.stop()

if __name__ == "__main__":
    main()
