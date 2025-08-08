#!/usr/bin/env python3
# lr_hyperparam_tuning.py

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from pyspark.mllib.evaluation import MulticlassMetrics

def main():
    # 1) SparkSession
    spark = (SparkSession.builder
             .appName("LR_Hyperparam_Tuning")
             .master("local[*]")
             .getOrCreate())
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    # 2) Veri okuma
    df = (spark.read
          .option("header", True)
          .option("inferSchema", True)
          .csv("creditcard.csv"))

    # 3) Özellik hazırlama
    feats = [f"V{i}" for i in range(1,29)] + ["Amount"]
    assembler = VectorAssembler(inputCols=feats, outputCol="rawFeatures")
    df_feat = assembler.transform(df)
    scaler = StandardScaler(inputCol="rawFeatures",
                            outputCol="features",
                            withMean=True, withStd=True)
    df_scaled = scaler.fit(df_feat).transform(df_feat)

    # 4) Label hazırlama
    df_final = df_scaled.select(
        col("features"),
        col("Class").cast(DoubleType()).alias("label")
    )

    # 5) Train/Test split
    train_df, test_df = df_final.randomSplit([0.8,0.2], seed=42)

    # 6) Logistic Regression tanımı
    lr = LogisticRegression(featuresCol="features",
                            labelCol="label")

    # 7) Evaluator (F1)
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )

    # 8) Geniş param grid
    paramGrid = (ParamGridBuilder()
        .addGrid(lr.regParam,        [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 0.5])
        .addGrid(lr.elasticNetParam, [0.0, 0.35, 0.5, 0.75, 1.0])
        .addGrid(lr.maxIter,         [50, 100, 200])
        .addGrid(lr.threshold,       [i * 0.1 for i in range(1,10)])  # 0.1,0.2,...,0.9
        .build()
    )

    # 9) CrossValidator
    cv = CrossValidator(estimator=lr,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator,
                        numFolds=3,
                        parallelism=4)  # aynı anda kaç model eğitilsin

    # 10) Model araması
    cvModel = cv.fit(train_df)

    # 11) En iyi hiperparametreler
    best = cvModel.bestModel
    print("=== En iyi hiperparametreler ===")
    print(f"regParam:       {best.getRegParam()}")
    print(f"elasticNetParam:{best.getElasticNetParam()}")
    print(f"threshold:      {best.getThreshold()}")

    # 12) Test set performansı
    preds = cvModel.transform(test_df)
    # Spark evaluator ile F1
    f1 = evaluator.evaluate(preds)
    print(f"\nCrossValidator F1 on test: {f1:.4f}")

    # Karışıklık matrisi ve diğer metrikler
    rdd = preds.select("prediction","label") \
               .rdd.map(lambda r:(r[0],r[1]))
    metrics = MulticlassMetrics(rdd)
    print("=== Test Metrics ===")
    print(f"Precision (fraud=1): {metrics.precision(1.0):.4f}")
    print(f"Recall    (fraud=1): {metrics.recall(1.0):.4f}")
    print(f"F1-Score  (fraud=1): {metrics.fMeasure(1.0):.4f}")
    print(f"Accuracy overall:   {metrics.accuracy:.4f}")

    spark.stop()

if __name__=="__main__":
    main()
