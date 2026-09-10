# Databricks notebook source
from pyspark.sql.functions import current_timestamp, lit

VOLUME_PATH = "/Volumes/workspace/bronze/raw_files"

tables = {
    "Production_WorkOrder.csv":          "workorder",
    "Production_WorkOrderRouting.csv":   "workorder_routing",
    "Production_Product.csv":            "product",
    "Production_ProductCategory.csv":    "product_category",
    "Production_ProductSubCategory.csv": "product_subcategory",
    "Production_BillOfMaterials.csv":    "bill_of_materials",
    "Production_Location.csv":           "location",
    "Production_ScrapReason.csv":        "scrap_reason",
}

for file_name, table_name in tables.items():
    df = (spark.read
          .option("header", "true")
          .option("inferSchema", "false")
          .option("multiLine", "true")
          .option("quote", '"')
          .option("escape", '"')
          .csv(f"{VOLUME_PATH}/{file_name}"))

    df = (df
          .withColumn("_source_file", lit(file_name))
          .withColumn("_ingested_at", current_timestamp()))

    (df.write
       .mode("overwrite")
       .option("overwriteSchema", "true")
       .saveAsTable(f"workspace.bronze.{table_name}"))

    print(f"{table_name:22} {df.count():>7,} satır")

# COMMAND ----------

