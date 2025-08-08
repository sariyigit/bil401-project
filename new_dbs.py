#!/usr/bin/env python3
# dbsc.py

from pyspark.sql import SparkSession
import dbscan
from scipy.spatial import distance

def main():
    spark = (SparkSession.builder
             .appName("DBSCAN Fraud Detection")
             .master("local[*]")
             .config("spark.driver.host", "127.0.0.1")
             .getOrCreate())
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    # 1) CSV oku ve (row, idx) ikilisini al
    rdd = sc.textFile("creditcard.csv")
    header = rdd.first()
    data = (rdd.filter(lambda l: l != header)
               .map(lambda line: line.split(",")))
    # 2) zipWithIndex + doğru Python3 unpack
    df = (data
          .zipWithIndex()
          .map(lambda row_idx: (
               row_idx[1],                             # idx
               [float(x) for x in row_idx[0][1:29]] +  # V1..V28
               [float(row_idx[0][29])]                 # Amount
          ))
          .toDF(["id", "value"])
          .repartition(20))

    # 3) DBSCAN parametreleri
    eps, minPts, maxPerPart = 0.3, 10, 2

    # 4) Çalıştır
    result = dbscan.process(
        spark, df,
        eps,
        minPts,
        distance.euclidean,
        maxPerPart,
        "dbscan_checkpoint"
    )

    result.show()

    result.write \
          .mode("overwrite") \
          .parquet("dbscan_results.parquet")
    

    spark.stop()

if __name__ == "__main__":
    main()