# Databricks notebook source
# MAGIC %sql
# MAGIC SELECT StartDate, EndDate, DueDate FROM workspace.bronze.workorder LIMIT 3;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT ScheduledStartDate, ActualStartDate, ActualEndDate 
# MAGIC FROM workspace.bronze.workorder_routing LIMIT 3;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   concat('[', UnitMeasureCode, ']') AS unit_kod,
# MAGIC   length(UnitMeasureCode)           AS uzunluk
# MAGIC FROM workspace.bronze.bill_of_materials
# MAGIC LIMIT 3;

# COMMAND ----------

from pyspark.sql import functions as F

ISO_FMT = "yyyy-MM-dd HH:mm"
US_FMT  = "M/d/yyyy H:mm"

def clean_strings(df):
    """Tüm string kolonlardaki baş/son boşlukları kırpar, boş string'i null yapar."""
    for c, t in df.dtypes:
        if t == "string" and not c.startswith("_"):
            df = df.withColumn(c, F.nullif(F.trim(F.col(c)), F.lit("")))
    return df

def check_nulls(df, cols, label):
    """Dönüşüm sonrası null sayılarını raporlar."""
    total = df.count()
    print(f"\n{label}  (toplam {total:,} satır)")
    for c in cols:
        n = df.filter(F.col(c).isNull()).count()
        pct = 100 * n / total if total else 0
        flag = "  <-- DİKKAT" if pct > 5 else ""
        print(f"  {c:22} null: {n:>6,}  ({pct:5.2f}%){flag}")

# COMMAND ----------

wo = spark.table("workspace.bronze.workorder")
wo = clean_strings(wo)

wo_silver = (wo
    .withColumn("WorkOrderID",   F.col("WorkOrderID").cast("int"))
    .withColumn("ProductID",     F.col("ProductID").cast("int"))
    .withColumn("OrderQty",      F.col("OrderQty").cast("int"))
    .withColumn("StockedQty",    F.col("StockedQty").cast("int"))
    .withColumn("ScrappedQty",   F.col("ScrappedQty").cast("int"))
    .withColumn("ScrapReasonID", F.col("ScrapReasonID").cast("int"))
    .withColumn("StartDate",     F.to_timestamp("StartDate", ISO_FMT).cast("date"))
    .withColumn("EndDate",       F.to_timestamp("EndDate",   ISO_FMT).cast("date"))
    .withColumn("DueDate",       F.to_timestamp("DueDate",   ISO_FMT).cast("date"))
)

check_nulls(wo_silver, ["WorkOrderID","ProductID","OrderQty","StartDate","EndDate","DueDate","ScrapReasonID"], "workorder")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   COUNT(*)                                              AS toplam_is_emri,
# MAGIC   SUM(CASE WHEN ScrappedQty > 0 THEN 1 ELSE 0 END)      AS hurda_cikan,
# MAGIC   SUM(CASE WHEN ScrapReasonID IS NOT NULL THEN 1 ELSE 0 END) AS sebep_dolu
# MAGIC FROM workspace.bronze.workorder;
# MAGIC

# COMMAND ----------

routing = spark.table("workspace.bronze.workorder_routing")
routing = clean_strings(routing)

routing_silver = (routing
    .withColumn("WorkOrderID",        F.col("WorkOrderID").cast("int"))
    .withColumn("ProductID",          F.col("ProductID").cast("int"))
    .withColumn("OperationSequence",  F.col("OperationSequence").cast("int"))
    .withColumn("LocationID",         F.col("LocationID").cast("int"))
    .withColumn("ActualResourceHrs",  F.col("ActualResourceHrs").cast("double"))
    .withColumn("PlannedCost",        F.col("PlannedCost").cast("decimal(18,4)"))
    .withColumn("ActualCost",         F.col("ActualCost").cast("decimal(18,4)"))
    .withColumn("ScheduledStartDate", F.to_timestamp("ScheduledStartDate", US_FMT).cast("date"))
    .withColumn("ScheduledEndDate",   F.to_timestamp("ScheduledEndDate",   US_FMT).cast("date"))
    .withColumn("ActualStartDate",    F.to_timestamp("ActualStartDate",    US_FMT).cast("date"))
    .withColumn("ActualEndDate",      F.to_timestamp("ActualEndDate",      US_FMT).cast("date"))
)

check_nulls(routing_silver,
    ["WorkOrderID","ProductID","OperationSequence","LocationID",
     "ActualResourceHrs","PlannedCost","ActualCost",
     "ScheduledStartDate","ScheduledEndDate","ActualStartDate","ActualEndDate"],
    "workorder_routing")

# COMMAND ----------

routing_silver.selectExpr(
    "min(ActualStartDate) as earliest",
    "max(ActualEndDate)   as latest"
).show()

wo_silver.selectExpr(
    "min(StartDate) as earliest",
    "max(EndDate)   as latest"
).show()

# COMMAND ----------

# Product — the widest table, has the trailing-space problem in ProductLine/Class/Style
product_silver = (clean_strings(spark.table("workspace.bronze.product"))
    .withColumn("ProductID",            F.col("ProductID").cast("int"))
    .withColumn("MakeFlag",             F.col("MakeFlag").cast("boolean"))
    .withColumn("FinishedGoodsFlag",    F.col("FinishedGoodsFlag").cast("boolean"))
    .withColumn("SafetyStockLevel",     F.col("SafetyStockLevel").cast("int"))
    .withColumn("ReorderPoint",         F.col("ReorderPoint").cast("int"))
    .withColumn("StandardCost",         F.col("StandardCost").cast("decimal(18,4)"))
    .withColumn("ListPrice",            F.col("ListPrice").cast("decimal(18,4)"))
    .withColumn("Weight",               F.col("Weight").cast("double"))
    .withColumn("DaysToManufacture",    F.col("DaysToManufacture").cast("int"))
    .withColumn("ProductSubcategoryID", F.col("ProductSubcategoryID").cast("int"))
    .withColumn("ProductModelID",       F.col("ProductModelID").cast("int"))
    .withColumn("SellStartDate",        F.to_timestamp("SellStartDate", ISO_FMT).cast("date"))
    .withColumn("SellEndDate",          F.to_timestamp("SellEndDate",   ISO_FMT).cast("date"))
    .drop("rowguid")
)

subcategory_silver = (clean_strings(spark.table("workspace.bronze.product_subcategory"))
    .withColumn("ProductSubcategoryID", F.col("ProductSubcategoryID").cast("int"))
    .withColumn("ProductCategoryID",    F.col("ProductCategoryID").cast("int"))
    .withColumnRenamed("Name", "SubcategoryName")
    .drop("rowguid")
)

category_silver = (clean_strings(spark.table("workspace.bronze.product_category"))
    .withColumn("ProductCategoryID", F.col("ProductCategoryID").cast("int"))
    .withColumnRenamed("Name", "CategoryName")
)

location_silver = (clean_strings(spark.table("workspace.bronze.location"))
    .withColumn("LocationID",   F.col("LocationID").cast("int"))
    .withColumn("CostRate",     F.col("CostRate").cast("decimal(18,4)"))
    .withColumn("Availability", F.col("Availability").cast("double"))
    .withColumnRenamed("Name", "LocationName")
    .drop("ModifiedDate")
)

scrap_reason_silver = (clean_strings(spark.table("workspace.bronze.scrap_reason"))
    .withColumn("ScrapReasonID", F.col("ScrapReasonID").cast("int"))
    .withColumnRenamed("Name", "ScrapReasonName")
)

bom_silver = (clean_strings(spark.table("workspace.bronze.bill_of_materials"))
    .withColumn("BillOfMaterialsID",  F.col("BillOfMaterialsID").cast("int"))
    .withColumn("ProductAssemblyID",  F.col("ProductAssemblyID").cast("int"))
    .withColumn("ComponentID",        F.col("ComponentID").cast("int"))
    .withColumn("BOMLevel",           F.col("BOMLevel").cast("int"))
    .withColumn("PerAssemblyQty",     F.col("PerAssemblyQty").cast("decimal(18,2)"))
)

print("done")

# COMMAND ----------

product_silver.selectExpr(
    "concat('[', ProductLine, ']') as line",
    "concat('[', Class, ']')       as cls",
    "concat('[', Style, ']')       as style"
).show(5)

# COMMAND ----------

silver_tables = {
    "workorder":           wo_silver,
    "workorder_routing":   routing_silver,
    "product":             product_silver,
    "product_subcategory": subcategory_silver,
    "product_category":    category_silver,
    "location":            location_silver,
    "scrap_reason":        scrap_reason_silver,
    "bill_of_materials":   bom_silver,
}

for name, df in silver_tables.items():
    (df.write
       .mode("overwrite")
       .option("overwriteSchema", "true")
       .saveAsTable(f"workspace.silver.{name}"))
    print(f"{name:22} {df.count():>7,} rows")

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE workspace.silver.workorder;

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE workspace.silver.workorder;

# COMMAND ----------

