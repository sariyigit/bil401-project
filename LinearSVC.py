#!/usr/bin/env python3
# pyspark_linear_svc_hpt.py

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType

def main():
    spark = (SparkSession.builder
             .appName("LinearSVC_Hyperparam_Tuning")
             .master("local[*]")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # Veri hazırlığı
    df = (spark.read
          .option("header", True)
          .option("inferSchema", True)
          .csv("creditcard.csv"))
    feat_cols = [f"V{i}" for i in range(1,29)] + ["Amount"]
    assembler = VectorAssembler(inputCols=feat_cols, outputCol="rawFeatures")
    df_feat = assembler.transform(df)
    scaler = StandardScaler(inputCol="rawFeatures", outputCol="features",
                            withMean=True, withStd=True)
    df_scaled = scaler.fit(df_feat).transform(df_feat)
    df_final = df_scaled.select(
        col("features"),
        col("Class").cast(DoubleType()).alias("label")
    )
    train_df, test_df = df_final.randomSplit([0.8,0.2], seed=42)

    # Estimator & evaluator
    lsvc = LinearSVC(featuresCol="features", labelCol="label")
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )

    # Param grid
    paramGrid = (ParamGridBuilder()
        .addGrid(lsvc.regParam, [0.001, 0.01, 0.1, 1.0])
        .addGrid(lsvc.maxIter,   [50, 100, 200])
        .build()
    )

    # CrossValidator
    cv = CrossValidator(estimator=lsvc,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator,
                        numFolds=3,
                        parallelism=4)

    # Tuning & en iyi model
    cvModel = cv.fit(train_df)
    best = cvModel.bestModel
    print("=== Best LinearSVC Params ===")
    print(f"  regParam: {best.getRegParam()}")
    print(f"  maxIter:  {best.getMaxIter()}")

    # Test performansı
    preds = cvModel.transform(test_df)
    f1 = evaluator.evaluate(preds)
    print(f"\nCrossValidator F1 on test: {f1:.4f}")

    # Detaylı metrikler
    from pyspark.mllib.evaluation import MulticlassMetrics
    rdd = preds.select("prediction","label").rdd.map(lambda r:(r[0],r[1]))
    metrics = MulticlassMetrics(rdd)
    print("=== Test Metrics (LinearSVC) ===")
    print(f"Precision (fraud=1): {metrics.precision(1.0):.4f}")
    print(f"Recall    (fraud=1): {metrics.recall(1.0):.4f}")
    print(f"F1-Score  (fraud=1): {metrics.fMeasure(1.0):.4f}")
    print(f"Accuracy overall:    {metrics.accuracy:.4f}")

    spark.stop()

if __name__=="__main__":
    main()
