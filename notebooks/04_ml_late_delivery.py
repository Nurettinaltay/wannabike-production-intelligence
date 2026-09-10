# Databricks notebook source


# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import Window

fact = spark.table("workspace.gold.fact_workorder")
prod = spark.table("workspace.gold.dim_product")

# Workload feature: how many work orders were already open on the day this one started
daily_load = (fact.groupBy("StartDate")
    .agg(F.count("*").alias("OrdersStartedSameDay")))

ml_base = (fact
    .join(prod.select("ProductID","ProductGroup","CategoryName",
                      "DaysToManufacture","StandardCost","ListPrice"),
          on="ProductID", how="left")
    .join(daily_load, on="StartDate", how="left")
    .select(
        "WorkOrderID",
        "ProductGroup", "CategoryName",
        F.col("OrderQty").cast("double"),
        F.col("DaysToManufacture").cast("double"),
        F.col("StandardCost").cast("double"),
        F.col("PlannedLeadDays").cast("double"),
        F.col("OrdersStartedSameDay").cast("double"),
        F.month("StartDate").alias("StartMonth").cast("double"),
        F.dayofweek("StartDate").alias("StartDayOfWeek").cast("double"),
        F.year("StartDate").alias("StartYear"),
        "StartDate",
        F.col("IsLate").cast("int").alias("label")
    )
)

ml_base.write.mode("overwrite").option("overwriteSchema","true").saveAsTable("workspace.gold.ml_workorder_features")

print("rows:", ml_base.count())
ml_base.groupBy("label").count().show()

# COMMAND ----------

train = ml_base.filter(F.col("StartDate") < "2025-01-01").drop("StartDate","StartYear")
test  = ml_base.filter(F.col("StartDate") >= "2025-01-01").drop("StartDate","StartYear")

print("train:", train.count(), " test:", test.count())
train.groupBy("label").count().show()
test.groupBy("label").count().show()

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import GBTClassifier

categorical = ["ProductGroup", "CategoryName"]
numeric = ["OrderQty", "DaysToManufacture", "StandardCost",
           "PlannedLeadDays", "OrdersStartedSameDay",
           "StartMonth", "StartDayOfWeek"]

indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
            for c in categorical]
encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec")
            for c in categorical]

assembler = VectorAssembler(
    inputCols=[f"{c}_vec" for c in categorical] + numeric,
    outputCol="features",
    handleInvalid="skip"
)

gbt = GBTClassifier(featuresCol="features", labelCol="label", maxIter=50, maxDepth=5, seed=42)

pipeline = Pipeline(stages=indexers + encoders + [assembler, gbt])
model = pipeline.fit(train)

print("trained")

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

pred = model.transform(test)

auc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC").evaluate(pred)

acc  = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy").evaluate(pred)
prec = MulticlassClassificationEvaluator(labelCol="label", metricName="precisionByLabel", metricLabel=1.0).evaluate(pred)
rec  = MulticlassClassificationEvaluator(labelCol="label", metricName="recallByLabel",    metricLabel=1.0).evaluate(pred)
f1   = MulticlassClassificationEvaluator(labelCol="label", metricName="fMeasureByLabel",  metricLabel=1.0).evaluate(pred)

baseline = pred.filter(F.col("label") == 0).count() / pred.count()

print(f"Baseline (always predict on-time) : {baseline:.3f}")
print(f"Accuracy                          : {acc:.3f}")
print(f"AUC                               : {auc:.3f}")
print(f"Precision (late)                  : {prec:.3f}")
print(f"Recall (late)                     : {rec:.3f}")
print(f"F1 (late)                         : {f1:.3f}")

# COMMAND ----------

pred.groupBy("label", "prediction").count().orderBy("label", "prediction").show()

# COMMAND ----------

gbt_model = model.stages[-1]
assembler_stage = [s for s in model.stages if s.__class__.__name__ == "VectorAssembler"][0]

import pandas as pd
attrs = pred.schema["features"].metadata["ml_attr"]["attrs"]
names = [None] * gbt_model.numFeatures
for kind in attrs:
    for a in attrs[kind]:
        names[a["idx"]] = a["name"]

imp = pd.DataFrame({
    "feature": names,
    "importance": gbt_model.featureImportances.toArray()
}).sort_values("importance", ascending=False)

print(imp.to_string(index=False))

# COMMAND ----------

import mlflow
import mlflow.spark

mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run(run_name="gbt_late_delivery_v1") as run:

    # Parameters — what did we train with
    mlflow.log_param("algorithm",   "GBTClassifier")
    mlflow.log_param("maxIter",     50)
    mlflow.log_param("maxDepth",    5)
    mlflow.log_param("split",       "temporal: train <2025, test >=2025")
    mlflow.log_param("train_rows",  train.count())
    mlflow.log_param("test_rows",   test.count())

    # Metrics — how did it perform
    mlflow.log_metric("baseline",  baseline)
    mlflow.log_metric("accuracy",  acc)
    mlflow.log_metric("auc",       auc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall",    rec)
    mlflow.log_metric("f1",        f1)

    # Feature importance as an artifact
    imp.to_csv("/tmp/feature_importance.csv", index=False)
    mlflow.log_artifact("/tmp/feature_importance.csv")

    mlflow.spark.log_model(
        model,
        artifact_path="model",
        registered_model_name="workspace.gold.late_delivery_gbt",
        dfs_tmpdir="/Volumes/workspace/gold/ml_tmp",
        input_example=train.limit(5).toPandas()
    )

    print("run_id:", run.info.run_id)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE VOLUME IF NOT EXISTS workspace.gold.ml_tmp;

# COMMAND ----------

import mlflow
import mlflow.spark

mlflow.set_registry_uri("databricks-uc")

TMP_VOLUME = "/Volumes/workspace/gold/ml_tmp"

with mlflow.start_run(run_name="gbt_late_delivery_v1") as run:

    mlflow.log_param("algorithm",   "GBTClassifier")
    mlflow.log_param("maxIter",     50)
    mlflow.log_param("maxDepth",    5)
    mlflow.log_param("split",       "temporal: train <2025, test >=2025")
    mlflow.log_param("train_rows",  train.count())
    mlflow.log_param("test_rows",   test.count())

    mlflow.log_metric("baseline",  baseline)
    mlflow.log_metric("accuracy",  acc)
    mlflow.log_metric("auc",       auc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall",    rec)
    mlflow.log_metric("f1",        f1)

    imp.to_csv("/tmp/feature_importance.csv", index=False)
    mlflow.log_artifact("/tmp/feature_importance.csv")

    mlflow.spark.log_model(
        model,
        artifact_path="model",
        registered_model_name="workspace.gold.late_delivery_gbt",
        dfs_tmpdir=TMP_VOLUME,
        input_example=train.limit(5).toPandas()
    )

    print("run_id:", run.info.run_id)

# COMMAND ----------

import mlflow
import mlflow.spark

mlflow.set_registry_uri("databricks-uc")

TMP_VOLUME = "/Volumes/workspace/gold/ml_tmp"

# A few rows the model can inspect to infer the input schema
input_example = train.drop("label").limit(5).toPandas()

with mlflow.start_run(run_name="gbt_late_delivery_v1") as run:

    mlflow.log_param("algorithm",   "GBTClassifier")
    mlflow.log_param("maxIter",     50)
    mlflow.log_param("maxDepth",    5)
    mlflow.log_param("split",       "temporal: train <2025, test >=2025")
    mlflow.log_param("train_rows",  train.count())
    mlflow.log_param("test_rows",   test.count())

    mlflow.log_metric("baseline",  baseline)
    mlflow.log_metric("accuracy",  acc)
    mlflow.log_metric("auc",       auc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall",    rec)
    mlflow.log_metric("f1",        f1)

    imp.to_csv("/tmp/feature_importance.csv", index=False)
    mlflow.log_artifact("/tmp/feature_importance.csv")

    mlflow.spark.log_model(
        model,
        artifact_path="model",
        registered_model_name="workspace.gold.late_delivery_gbt",
        dfs_tmpdir=TMP_VOLUME,
        input_example=input_example
    )

    print("run_id:", run.info.run_id)

# COMMAND ----------

from pyspark.ml.functions import vector_to_array

scored = (model.transform(ml_base.drop("label"))
    .withColumn("LateRiskScore", F.round(vector_to_array("probability")[1], 4))
    .withColumn("RiskBand",
        F.when(F.col("LateRiskScore") >= 0.70, "High")
         .when(F.col("LateRiskScore") >= 0.40, "Medium")
         .otherwise("Low"))
    .select("WorkOrderID", "LateRiskScore", "RiskBand",
            F.col("prediction").cast("int").alias("PredictedLate"))
)

scored.write.mode("overwrite").option("overwriteSchema","true").saveAsTable("workspace.gold.fact_workorder_risk")

print("scored:", scored.count())
scored.groupBy("RiskBand").count().orderBy("RiskBand").show()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   r.RiskBand,
# MAGIC   COUNT(*)                              AS work_orders,
# MAGIC   SUM(f.IsLate)                         AS actually_late,
# MAGIC   ROUND(100 * AVG(f.IsLate), 1)         AS actual_late_pct,
# MAGIC   ROUND(100 * AVG(r.LateRiskScore), 1)  AS avg_predicted_pct
# MAGIC FROM workspace.gold.fact_workorder_risk r
# MAGIC JOIN workspace.gold.fact_workorder f ON r.WorkOrderID = f.WorkOrderID
# MAGIC GROUP BY r.RiskBand
# MAGIC ORDER BY avg_predicted_pct DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT ai_query(
# MAGIC   'databricks-meta-llama-3-3-70b-instruct',
# MAGIC   'Reply with exactly one word: OK'
# MAGIC ) AS test;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   f.WorkOrderID,
# MAGIC   p.ProductName,
# MAGIC   p.ProductGroup,
# MAGIC   f.OrderQty,
# MAGIC   f.ScrappedQty,
# MAGIC   ROUND(100 * f.ScrapRate, 2) AS scrap_pct,
# MAGIC   s.ScrapReasonName,
# MAGIC   f.StartDate,
# MAGIC   f.EndDate,
# MAGIC   ai_query(
# MAGIC     'databricks-meta-llama-3-3-70b-instruct',
# MAGIC     CONCAT(
# MAGIC       'You are a manufacturing analyst. Write ONE short sentence (max 25 words) ',
# MAGIC       'summarising this scrap incident for a production manager. ',
# MAGIC       'Use only the facts given. Do not speculate about causes beyond the stated reason. ',
# MAGIC       'Facts: product=', p.ProductName,
# MAGIC       ', group=', p.ProductGroup,
# MAGIC       ', ordered=', CAST(f.OrderQty AS STRING),
# MAGIC       ', scrapped=', CAST(f.ScrappedQty AS STRING),
# MAGIC       ', scrap_rate=', CAST(ROUND(100*f.ScrapRate,2) AS STRING), '%',
# MAGIC       ', reason=', s.ScrapReasonName
# MAGIC     )
# MAGIC   ) AS incident_summary
# MAGIC FROM workspace.gold.fact_workorder f
# MAGIC JOIN workspace.gold.dim_product      p ON f.ProductID     = p.ProductID
# MAGIC JOIN workspace.gold.dim_scrap_reason s ON f.ScrapReasonID = s.ScrapReasonID
# MAGIC WHERE f.HasScrap = 1
# MAGIC ORDER BY f.ScrappedQty DESC
# MAGIC LIMIT 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   f.WorkOrderID,
# MAGIC   p.ProductName,
# MAGIC   f.OrderQty,
# MAGIC   f.ScrappedQty,
# MAGIC   ROUND(100 * f.ScrapRate, 2) AS scrap_pct,
# MAGIC   s.ScrapReasonName,
# MAGIC   ai_query(
# MAGIC     'databricks-meta-llama-3-3-70b-instruct',
# MAGIC     CONCAT(
# MAGIC       'Write exactly one sentence in this format: ',
# MAGIC       '"<qty> units of <product> were scrapped out of <ordered> ordered (<rate>%) due to <reason>." ',
# MAGIC       'Do not add anything else. Use the exact numbers given. ',
# MAGIC       'qty=',      CAST(f.ScrappedQty AS STRING),
# MAGIC       ', product=', p.ProductName,
# MAGIC       ', ordered=', CAST(f.OrderQty AS STRING),
# MAGIC       ', rate=',    CAST(ROUND(100*f.ScrapRate,2) AS STRING),
# MAGIC       ', reason=',  s.ScrapReasonName
# MAGIC     )
# MAGIC   ) AS incident_summary
# MAGIC FROM workspace.gold.fact_workorder f
# MAGIC JOIN workspace.gold.dim_product      p ON f.ProductID     = p.ProductID
# MAGIC JOIN workspace.gold.dim_scrap_reason s ON f.ScrapReasonID = s.ScrapReasonID
# MAGIC WHERE f.HasScrap = 1
# MAGIC ORDER BY f.ScrappedQty DESC
# MAGIC LIMIT 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   f.WorkOrderID,
# MAGIC   CONCAT(
# MAGIC     CAST(f.ScrappedQty AS STRING), ' units of ', p.ProductName,
# MAGIC     ' were scrapped out of ', CAST(f.OrderQty AS STRING), ' ordered (',
# MAGIC     CAST(ROUND(100*f.ScrapRate,2) AS STRING), '%) due to ',
# MAGIC     LOWER(s.ScrapReasonName), '.'
# MAGIC   ) AS incident_summary
# MAGIC FROM workspace.gold.fact_workorder f
# MAGIC JOIN workspace.gold.dim_product      p ON f.ProductID     = p.ProductID
# MAGIC JOIN workspace.gold.dim_scrap_reason s ON f.ScrapReasonID = s.ScrapReasonID
# MAGIC WHERE f.HasScrap = 1
# MAGIC ORDER BY f.ScrappedQty DESC
# MAGIC LIMIT 10;

# COMMAND ----------

