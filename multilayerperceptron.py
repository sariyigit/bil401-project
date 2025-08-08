#!/usr/bin/env python3
# pyspark_mlp_hyperparam_tuning.py

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from pyspark.mllib.evaluation import MulticlassMetrics

def main():
    spark = (SparkSession.builder
             .appName("MLP_Hyperparam_Tuning")
             .master("local[*]")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # 1) Veri oku
    df = (spark.read
          .option("header", True)
          .option("inferSchema", True)
          .csv("creditcard.csv"))

    # 2) Özellikleri birleştir ve ölçekle
    feat_cols = [f"V{i}" for i in range(1,29)] + ["Amount"]
    assembler = VectorAssembler(inputCols=feat_cols, outputCol="rawFeatures")
    df_feat = assembler.transform(df)
    scaler = StandardScaler(inputCol="rawFeatures",
                            outputCol="features",
                            withMean=True, withStd=True)
    df_scaled = scaler.fit(df_feat).transform(df_feat)

    # 3) Label seçimi
    df_final = df_scaled.select(
        col("features"),
        col("Class").cast(DoubleType()).alias("label")
    )

    # 4) Train/Test split
    train_df, test_df = df_final.randomSplit([0.8, 0.2], seed=42)

    # 5) MLPClassifier tanımı (placeholder parametreler)
    mlp = MultilayerPerceptronClassifier(
        featuresCol="features",
        labelCol="label",
        predictionCol="prediction",
        seed=42
    )

    # 6) Evaluator (fraud=1 için F1)
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )

    # 7) Parametre grid
    paramGrid = (ParamGridBuilder()
        # Farklı gizli katman mimarileri
        .addGrid(mlp.layers, [
            [29, 50, 2],
            [29, 50, 25, 2],
            [29, 100, 50, 2]
        ])
        # İterasyon sayısı
        .addGrid(mlp.maxIter, [50, 100])
        # Blok boyutu
        .addGrid(mlp.blockSize, [64, 128])
        # Solver tipi: 'l-bfgs' veya 'gd'
        .addGrid(mlp.solver, ["l-bfgs", "gd"])
        .build()
    )

    # 8) CrossValidator
    cv = CrossValidator(
        estimator=mlp,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=3,
        parallelism=4
    )

    # 9) Hiperparametre araması
    cvModel = cv.fit(train_df)

    # 10) En iyi parametreler
    best = cvModel.bestModel
    print("=== En İyi MLP Parametreleri ===")
    print(f"  layers:    {best.getLayers()}")
    print(f"  maxIter:   {best.getMaxIter()}")
    print(f"  blockSize: {best.getBlockSize()}")
    print(f"  solver:    {best.getSolver()}")

    # 11) Test set performansı
    preds = cvModel.transform(test_df)
    f1 = evaluator.evaluate(preds)
    print(f"\nCrossValidator F1 on test: {f1:.4f}")

    # 12) Fraud=1 için detaylı metrikler
    rdd = preds.select("prediction","label") \
               .rdd.map(lambda r: (r[0], r[1]))
    metrics = MulticlassMetrics(rdd)
    print("=== Test Metrikleri (MLP) ===")
    print(f"Precision (fraud=1): {metrics.precision(1.0):.4f}")
    print(f"Recall    (fraud=1): {metrics.recall(1.0):.4f}")
    print(f"F1-Score  (fraud=1): {metrics.fMeasure(1.0):.4f}")
    print(f"Accuracy overall:    {metrics.accuracy:.4f}")

    spark.stop()

if __name__=="__main__":
    main()
