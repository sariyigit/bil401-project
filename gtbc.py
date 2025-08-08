#!/usr/bin/env python3
# pyspark_gbt_hyperparam_tuning.py

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from pyspark.mllib.evaluation import MulticlassMetrics

def main():
    # 1) SparkSession
    spark = (SparkSession.builder
             .appName("GBT_Hyperparam_Tuning")
             .master("local[*]")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # 2) Veri oku
    df = (spark.read
          .option("header", True)
          .option("inferSchema", True)
          .csv("creditcard.csv"))

    # 3) Özellik hazırlama
    feat_cols = [f"V{i}" for i in range(1,29)] + ["Amount"]
    assembler = VectorAssembler(inputCols=feat_cols, outputCol="rawFeatures")
    df_feat = assembler.transform(df)
    scaler = StandardScaler(
        inputCol="rawFeatures",
        outputCol="features",
        withMean=True, withStd=True
    )
    df_scaled = scaler.fit(df_feat).transform(df_feat)

    # 4) Label seçimi
    df_final = df_scaled.select(
        col("features"),
        col("Class").cast(DoubleType()).alias("label")
    )

    # 5) Train/Test split
    train_df, test_df = df_final.randomSplit([0.8, 0.2], seed=42)

    # 6) GBTClassifier tanımı
    gbt = GBTClassifier(
        featuresCol="features",
        labelCol="label",
        predictionCol="prediction",
        seed=42
    )

    # 7) Evaluator (F1)
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )

    # 8) Parametre grid’i
    paramGrid = (ParamGridBuilder()
        .addGrid(gbt.maxIter,    [20, 50, 100])
        .addGrid(gbt.maxDepth,   [3, 5, 7])
        .addGrid(gbt.stepSize,   [0.05, 0.1, 0.2])
        .addGrid(gbt.maxBins,    [32, 64])
        .build()
    )

    # 9) CrossValidator
    cv = CrossValidator(
        estimator=gbt,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=3,
        parallelism=4
    )

    # 10) Hiperparametre araması
    cvModel = cv.fit(train_df)

    # 11) En iyi parametreler
    best = cvModel.bestModel
    print("=== En İyi GBT Parametreleri ===")
    print(f" maxIter:   {best.getMaxIter()}")
    print(f" maxDepth:  {best.getMaxDepth()}")
    print(f" stepSize:  {best.getStepSize():.3f}")
    print(f" maxBins:   {best.getMaxBins()}")

    # 12) Test set performansı
    preds = cvModel.transform(test_df)
    f1 = evaluator.evaluate(preds)
    print(f"\nCrossValidator F1 on test: {f1:.4f}")

    # 13) Detaylı metrikler (fraud=1 için)
    rdd = preds.select("prediction","label") \
               .rdd.map(lambda r: (r[0], r[1]))
    metrics = MulticlassMetrics(rdd)
    print("=== Test Metrics (GBT) ===")
    print(f"Precision (fraud=1): {metrics.precision(1.0):.4f}")
    print(f"Recall    (fraud=1): {metrics.recall(1.0):.4f}")
    print(f"F1-Score  (fraud=1): {metrics.fMeasure(1.0):.4f}")
    print(f"Accuracy overall:    {metrics.accuracy:.4f}")

    spark.stop()

if __name__ == "__main__":
    main()
