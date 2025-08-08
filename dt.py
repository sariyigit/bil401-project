#!/usr/bin/env python3
# pyspark_dt_extreme_hyperparam_tuning_fixed.py

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from pyspark.mllib.evaluation import MulticlassMetrics

def main():
    spark = (SparkSession.builder
             .appName("DT_Extreme_HPT_Fixed")
             .master("local[*]")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # Veriyi oku + hazırla
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

    # Train/test split
    train_df, test_df = df_final.randomSplit([0.8, 0.2], seed=42)

    # Decision Tree
    dt = DecisionTreeClassifier(featuresCol="features",
                                labelCol="label",
                                seed=42)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )

    # Düzeltilmiş param grid
    paramGrid = (ParamGridBuilder()
        .addGrid(dt.maxDepth,            [3, 5, 7, 10, 15, 20])
        .addGrid(dt.maxBins,             [16, 32, 64, 128])
        .addGrid(dt.impurity,            ["gini", "entropy"])
        .addGrid(dt.minInstancesPerNode, [1, 5, 10, 20])
        # Sıfır yerine yalnızca >0 değerler
        .addGrid(dt.minInfoGain,         [1e-4, 1e-3, 1e-2])
        .build()
    )

    cv = CrossValidator(estimator=dt,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator,
                        numFolds=3,
                        parallelism=4)

    cvModel = cv.fit(train_df)

    best = cvModel.bestModel
    print("=== En İyi DT Parametreleri (Fixed Grid) ===")
    print(f" maxDepth:            {best.getMaxDepth()}")
    print(f" maxBins:             {best.getMaxBins()}")
    print(f" impurity:            {best.getImpurity()}")
    print(f" minInstancesPerNode: {best.getMinInstancesPerNode()}")
    print(f" minInfoGain:         {best.getMinInfoGain()}")

    preds = cvModel.transform(test_df)
    f1 = evaluator.evaluate(preds)
    print(f"\nCrossValidator F1 on test: {f1:.4f}")

    rdd = preds.select("prediction","label").rdd.map(lambda r:(r[0],r[1]))
    metrics = MulticlassMetrics(rdd)
    print("=== Test Metrikleri (Fixed Grid) ===")
    print(f"Precision (fraud=1): {metrics.precision(1.0):.4f}")
    print(f"Recall    (fraud=1): {metrics.recall(1.0):.4f}")
    print(f"F1-Score  (fraud=1): {metrics.fMeasure(1.0):.4f}")
    print(f"Accuracy overall:    {metrics.accuracy:.4f}")

    spark.stop()

if __name__=="__main__":
    main()
