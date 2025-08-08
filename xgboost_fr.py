#!/usr/bin/env python3
# pyspark_xgboost_hpt_final.py

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from pyspark.mllib.evaluation import MulticlassMetrics

from xgboost.spark import SparkXGBClassifier

def main():
    spark = (SparkSession.builder
             .appName("XGB_Hyperparam_Tuning_Final")
             .master("local[*]")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # 1) Veri oku + hazırla
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
    train_df, test_df = df_final.randomSplit([0.8,0.2], seed=42)

    # 2) SparkXGBClassifier
    xgb = SparkXGBClassifier(
        features_col="features",
        label_col="label",
        prediction_col="prediction",
        num_workers=spark.sparkContext.defaultParallelism
    )
    # Eğer üç parametre adı da emin olmak istersen:
    # print(xgb.explainParams())

    # 3) Evaluator (F1)
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )

    # 4) Param grid (explainParams çıktısındaki snake_case isimlerle)
    paramGrid = (ParamGridBuilder()
        .addGrid(xgb.max_depth,       [4, 6, 8])
        .addGrid(xgb.learning_rate,   [0.05, 0.1, 0.2])
        .addGrid(xgb.subsample,       [0.8, 1.0])
        .addGrid(xgb.colsample_bytree,[0.8, 1.0])
        .addGrid(xgb.n_estimators,    [50, 100])
        .build()
    )

    # 5) 3-Fold CV
    cv = CrossValidator(
        estimator=xgb,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=3,
        parallelism=4
    )

    # 6) Tuning
    cvModel = cv.fit(train_df)

    # 7) En iyi parametreler
    best = cvModel.bestModel
    print("=== Best XGBoost Params ===")
    print(f" max_depth:        {best.getOrDefault(xgb.max_depth)}")
    print(f" learning_rate:    {best.getOrDefault(xgb.learning_rate)}")
    print(f" subsample:        {best.getOrDefault(xgb.subsample)}")
    print(f" colsample_bytree: {best.getOrDefault(xgb.colsample_bytree)}")
    print(f" n_estimators:     {best.getOrDefault(xgb.n_estimators)}")

    # 8) Test set performansı
    preds = cvModel.transform(test_df)
    f1 = evaluator.evaluate(preds)
    print(f"\nCrossValidator F1 on test: {f1:.4f}")

    # 9) Fraud=1 için detaylı metrikler
    rdd = preds.select("prediction","label").rdd.map(lambda r:(r[0], r[1]))
    metrics = MulticlassMetrics(rdd)
    print("=== Test Metrics (XGBoost) ===")
    print(f"Precision (fraud=1): {metrics.precision(1.0):.4f}")
    print(f"Recall    (fraud=1): {metrics.recall(1.0):.4f}")
    print(f"F1-Score  (fraud=1): {metrics.fMeasure(1.0):.4f}")
    print(f"Accuracy overall:    {metrics.accuracy:.4f}")

    spark.stop()

if __name__=="__main__":
    main()
