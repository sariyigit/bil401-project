from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

spark = (SparkSession.builder
         .appName("DBSCAN Metrics (noise-only)")
         .master("local[*]")
         .getOrCreate())
sc = spark.sparkContext
sc.setLogLevel("WARN")

# 1) Tüm orijinal etiketler
orig = (spark.read.option("header",True).csv("creditcard.csv")
        .rdd.zipWithIndex()
        .map(lambda ri: (ri[1], int(float(ri[0]["Class"]))))
        .toDF(["id","label"]))

# 2) DBSCAN sonucu (parquet)
res = (spark.read.parquet("dbscan_results.parquet")
       .withColumnRenamed("point","id"))

# 3) left_outer join → yalnızca noise’ları anomaly yap
df = orig.join(res, on="id", how="left_outer")
df = df.withColumn("prediction",
        when(col("component").isNull(), 1).otherwise(0))

# 4) Confusion
total = df.count()
TP = df.filter((col("prediction")==1)&(col("label")==1)).count()
FP = df.filter((col("prediction")==1)&(col("label")==0)).count()
FN = df.filter((col("prediction")==0)&(col("label")==1)).count()
TN = df.filter((col("prediction")==0)&(col("label")==0)).count()

# 5) Metrikler
prec  = TP/(TP+FP) if TP+FP>0 else 0.0
rec   = TP/(TP+FN) if TP+FN>0 else 0.0
f1    = 2*prec*rec/(prec+rec) if prec+rec>0 else 0.0
acc   = (TP+TN)/total if total>0 else 0.0

print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1:        {f1:.4f}")
print(f"Accuracy:  {acc:.4f}")
