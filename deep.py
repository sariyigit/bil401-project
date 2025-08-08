#!/usr/bin/env python3
# pyspark_elephas_classifier_fixed.py

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from elephas.spark_model import SparkModel
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def build_classifier(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(1e-3),
        metrics=['accuracy']
    )
    return model

def main():
    spark = (SparkSession.builder
             .appName("Elephas DNN Classifier Fixed")
             .master("local[*]")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # --- 1) Veri Hazırlığı ---
    df = (spark.read
          .option("header", True)
          .option("inferSchema", True)
          .csv("creditcard.csv"))

    feat_cols = [f"V{i}" for i in range(1,29)] + ["Amount"]
    assembler = VectorAssembler(inputCols=feat_cols, outputCol="rawFeatures")
    df_feat  = assembler.transform(df)

    scaler = StandardScaler(inputCol="rawFeatures",
                            outputCol="features",
                            withMean=True, withStd=True)
    df_scaled = scaler.fit(df_feat).transform(df_feat)

    df_final = df_scaled.select(
        col("features"),
        col("Class").cast(DoubleType()).alias("label")
    )

    # --- 2) DataFrame ile train/test split ---
    train_df, test_df = df_final.randomSplit([0.8, 0.2], seed=42)

    # --- 3) İlgili RDD’leri oluştur (features array, label) ---
    train_rdd = train_df.rdd.map(lambda row: (row["features"].toArray(), row["label"]))
    test_rdd  = test_df.rdd .map(lambda row: (row["features"].toArray(), row["label"]))

    # --- 4) Elephas DNN Modeli Tanımı ---
    input_dim   = len(feat_cols)
    keras_model = build_classifier(input_dim)
    spark_model = SparkModel(
        model=keras_model,
        frequency='epoch',
        mode='asynchronous',
        num_workers=spark.sparkContext.defaultParallelism
    )

    # --- 5) Modeli Eğit ---
    spark_model.fit(train_rdd, epochs=20, batch_size=256, verbose=1)

    # --- 6) Driver’daki Keras Modelini Al ---
    driver_model = spark_model.master_network

    # --- 7) Test Setinde Tahmin ---
    preds_labels = test_rdd.map(lambda tpl: (
        float(driver_model.predict(tpl[0].reshape(1,-1))[0][0] > 0.5),
        tpl[1]
    )).collect()

    y_pred = np.array([p for p, l in preds_labels])
    y_true = np.array([l for p, l in preds_labels])

    # --- 8) Metrikler ---
    print("=== Elephas DNN Classifier (Fixed) ===")
    print(f"Precision (fraud=1): {precision_score(y_true, y_pred):.4f}")
    print(f"Recall    (fraud=1): {recall_score   (y_true, y_pred):.4f}")
    print(f"F1-Score  (fraud=1): {f1_score       (y_true, y_pred):.4f}")
    print(f"Accuracy overall:    {accuracy_score(y_true, y_pred):.4f}")

    spark.stop()

if __name__=="__main__":
    main()
