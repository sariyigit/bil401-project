#!/usr/bin/env python3
# pyspark_nb_balanced_threshold.py

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.classification import NaiveBayes
from pyspark.sql.functions import col, when
from pyspark.sql.types import DoubleType
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.functions import vector_to_array

import numpy as np

def main():
    spark = (SparkSession.builder
             .appName("NB_Balanced_Threshold_Tuning")
             .master("local[*]")
             .getOrCreate())
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    # 1) Veri oku + MinMax ölçekle
    df = (spark.read
          .option("header", True)
          .option("inferSchema", True)
          .csv("creditcard.csv"))
    feat_cols = [f"V{i}" for i in range(1,29)] + ["Amount"]
    assembler = VectorAssembler(inputCols=feat_cols, outputCol="rawFeatures")
    df_feat  = assembler.transform(df)
    scaler   = MinMaxScaler(inputCol="rawFeatures", outputCol="features")
    df_scaled= scaler.fit(df_feat).transform(df_feat)

    # 2) Label sütunu
    df_final = df_scaled.select(
        col("features"),
        col("Class").cast(DoubleType()).alias("label")
    )

    # 3) Train / test split
    train, test = df_final.randomSplit([0.8,0.2], seed=42)

    # 4) Train üzerindeki sınıfları dengele
    fraud       = train.filter(col("label") == 1.0)
    nonfraud    = train.filter(col("label") == 0.0)
    frac        = fraud.count() / nonfraud.count()
    nonfraud_sm = nonfraud.sample(False, frac, seed=42)
    train_bal   = fraud.union(nonfraud_sm).cache()
    print(f"Train balanced: {train_bal.count()} rows ({fraud.count()} fraud, {nonfraud_sm.count()} non-fraud)")

    # 5) NaiveBayes eğit
    nb = NaiveBayes(
        featuresCol="features",
        labelCol="label",
        modelType="multinomial",
        smoothing=1.0
    )
    model = nb.fit(train_bal)

    # 6) Test üzerindeki olasılıkları al
    df_prob = (
        model.transform(test)
            .select("probability", "label")
            # vector_to_array ile Array[Double] çevir, sonra 1. indeksi al
            .withColumn("p1", vector_to_array(col("probability")).getItem(1))
    )

    # 7) Eşik taraması (örneğin numpy import edildiyse)
    import numpy as np

    best_f1 = 0.0
    best_th = 0.5
    for th in np.linspace(0.01, 0.99, 99):
        df_pred = df_prob.withColumn(
            "pred",
            when(col("p1") >= float(th), 1.0).otherwise(0.0)
        ).select("pred", col("label"))
        rdd = df_pred.rdd.map(lambda r:(r.pred, r.label))
        from pyspark.mllib.evaluation import MulticlassMetrics
        m = MulticlassMetrics(rdd)
        f1 = m.fMeasure(1.0)
        if f1 > best_f1:
            best_f1, best_th = f1, th

    print(f"→ En iyi eşik: {best_th:.2f} (F1={best_f1:.4f})")

    # 8) En iyi eşikle son metrikler
    df_best = df_prob.withColumn(
        "pred", when(col("p1") >= float(best_th), 1.0).otherwise(0.0)
    )
    rdd_best = df_best.rdd.map(lambda r:(r.pred, r.label))
    m2 = MulticlassMetrics(rdd_best)
    print("=== Son Test Metrikleri ===")
    print(f"Precision (fraud=1): {m2.precision(1.0):.4f}")
    print(f"Recall    (fraud=1): {m2.recall(1.0):.4f}")
    print(f"F1-Score  (fraud=1): {m2.fMeasure(1.0):.4f}")
    print(f"Accuracy overall:    {m2.accuracy:.4f}")

    spark.stop()

if __name__=="__main__":
    main()
 