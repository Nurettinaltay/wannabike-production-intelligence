# Databricks notebook source
from pyspark.sql import functions as F

product   = spark.table("workspace.silver.product")
subcat    = spark.table("workspace.silver.product_subcategory")
category  = spark.table("workspace.silver.product_category")

dim_product = (product
    .join(subcat,   on="ProductSubcategoryID", how="left")
    .join(category, on="ProductCategoryID",    how="left")
    .select(
        F.col("ProductID"),
        F.col("Name").alias("ProductName"),
        F.col("ProductNumber"),
        F.coalesce(F.col("CategoryName"),    F.lit("Uncategorized")).alias("CategoryName"),
        F.coalesce(F.col("SubcategoryName"), F.lit("Uncategorized")).alias("SubcategoryName"),
        F.col("ProductLine"),
        F.col("Class"),
        F.col("Style"),
        F.col("StandardCost"),
        F.col("ListPrice"),
        F.col("DaysToManufacture"),
        F.col("SafetyStockLevel"),
        F.col("ReorderPoint"),
        F.col("MakeFlag"),
        F.col("FinishedGoodsFlag"),
    )
)

print("dim_product rows:", dim_product.count())

# COMMAND ----------

print("silver.product :", product.count())
print("dim_product    :", dim_product.count())

# COMMAND ----------

dim_product.groupBy("CategoryName", "SubcategoryName").count().orderBy("CategoryName", "SubcategoryName").show(50, truncate=False)

# COMMAND ----------

dim_product.groupBy("CategoryName", "MakeFlag", "FinishedGoodsFlag").count().orderBy("CategoryName").show(50, truncate=False)

# COMMAND ----------

dim_product = dim_product.withColumn(
    "ProductGroup",
    F.when(F.col("CategoryName") == "Bikes", "Finished Bikes")
     .when((F.col("CategoryName") == "Components") & (F.col("MakeFlag") == True), "Manufactured Components")
     .when((F.col("CategoryName") == "Uncategorized") & (F.col("MakeFlag") == True), "Subassemblies")
     .when((F.col("MakeFlag") == False) & (F.col("FinishedGoodsFlag") == False), "Raw Materials")
     .otherwise("Purchased Goods")
)

dim_product.groupBy("ProductGroup").count().orderBy(F.desc("count")).show(truncate=False)

# COMMAND ----------

wo = spark.table("workspace.silver.workorder")
(wo.join(dim_product, "ProductID", "left")
   .groupBy("ProductGroup")
   .agg(F.count("*").alias("work_orders"), F.sum("OrderQty").alias("total_qty"))
   .orderBy(F.desc("work_orders"))
   .show(truncate=False))

# COMMAND ----------

wo = spark.table("workspace.silver.workorder")

(wo.join(dim_product, "ProductID", "left")
   .groupBy("ProductGroup")
   .agg(
       F.count("*").alias("work_orders"),
       F.sum("OrderQty").alias("total_qty"),
       F.round(F.avg("OrderQty"), 1).alias("avg_qty")
   )
   .orderBy(F.desc("work_orders"))
   .show(truncate=False))

# COMMAND ----------

(dim_product.write
   .mode("overwrite")
   .option("overwriteSchema", "true")
   .saveAsTable("workspace.gold.dim_product"))

print("dim_product:", spark.table("workspace.gold.dim_product").count())

# COMMAND ----------

# dim_location — with capacity utilisation baseline
location = spark.table("workspace.silver.location")

dim_location = location.select(
    "LocationID",
    "LocationName",
    "CostRate",
    F.col("Availability").alias("DailyCapacityHrs")
)

# dim_scrap_reason — plus an explicit "no scrap" member
scrap = spark.table("workspace.silver.scrap_reason")

dim_scrap_reason = (scrap
    .select("ScrapReasonID", "ScrapReasonName")
    .union(spark.createDataFrame([(0, "No Scrap")], "ScrapReasonID INT, ScrapReasonName STRING"))
)

for name, df in [("dim_location", dim_location), ("dim_scrap_reason", dim_scrap_reason)]:
    (df.write.mode("overwrite").option("overwriteSchema", "true")
       .saveAsTable(f"workspace.gold.{name}"))
    print(f"{name:20} {df.count():>4} rows")

# COMMAND ----------

from pyspark.sql import functions as F

wo = spark.table("workspace.silver.workorder")
routing = spark.table("workspace.silver.workorder_routing")

bounds = (wo.select(F.min("StartDate").alias("lo"), F.max("EndDate").alias("hi"))
            .union(routing.select(F.min("ActualStartDate"), F.max("ActualEndDate")))
            .agg(F.min("lo").alias("lo"), F.max("hi").alias("hi"))
            .collect()[0])

start_date, end_date = bounds["lo"], bounds["hi"]
print("range:", start_date, "->", end_date)

week_str = F.concat(F.lit("W"), F.lpad(F.weekofyear("DateKey").cast("string"), 2, "0"))

dim_date = (spark.sql(f"""
    SELECT explode(sequence(
        to_date('{start_date}'),
        to_date('{end_date}'),
        interval 1 day
    )) AS DateKey
""")
    .withColumn("Year",       F.year("DateKey"))
    .withColumn("Quarter",    F.concat(F.lit("Q"), F.quarter("DateKey")))
    .withColumn("Month",      F.month("DateKey"))
    .withColumn("MonthName",  F.date_format("DateKey", "MMMM"))
    .withColumn("YearMonth",  F.date_format("DateKey", "yyyy-MM"))
    .withColumn("WeekOfYear", F.weekofyear("DateKey"))
    .withColumn("YearWeek",   F.concat_ws("-", F.year("DateKey"), week_str))
    .withColumn("DayOfWeek",  F.dayofweek("DateKey"))
    .withColumn("DayName",    F.date_format("DateKey", "EEEE"))
    .withColumn("IsWeekend",  F.dayofweek("DateKey").isin([1, 7]))
)

(dim_date.write.mode("overwrite").option("overwriteSchema", "true")
   .saveAsTable("workspace.gold.dim_date"))

print("dim_date:", dim_date.count(), "rows")
dim_date.show(3, truncate=False)

# COMMAND ----------

wo = spark.table("workspace.silver.workorder")

fact_workorder = (wo
    .withColumn("ScrapReasonID", F.coalesce(F.col("ScrapReasonID"), F.lit(0)))
    .withColumn("PlannedLeadDays", F.datediff("DueDate", "StartDate"))
    .withColumn("ActualLeadDays",  F.datediff("EndDate",  "StartDate"))
    .withColumn("DelayDays",       F.datediff("EndDate",  "DueDate"))
    .withColumn("IsLate",          (F.col("EndDate") > F.col("DueDate")).cast("int"))
    .withColumn("ScrapRate",       F.round(F.col("ScrappedQty") / F.col("OrderQty"), 4))
    .withColumn("HasScrap",        (F.col("ScrappedQty") > 0).cast("int"))
    .select(
        "WorkOrderID", "ProductID", "ScrapReasonID",
        "OrderQty", "StockedQty", "ScrappedQty",
        "StartDate", "EndDate", "DueDate",
        "PlannedLeadDays", "ActualLeadDays", "DelayDays",
        "IsLate", "ScrapRate", "HasScrap"
    )
)

fact_workorder.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("workspace.gold.fact_workorder")

print("fact_workorder:", fact_workorder.count())

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   COUNT(*)                              AS total_work_orders,
# MAGIC   SUM(IsLate)                           AS late_orders,
# MAGIC   ROUND(100 * AVG(IsLate), 1)           AS late_pct,
# MAGIC   ROUND(100 * (1 - AVG(IsLate)), 1)     AS on_time_pct,
# MAGIC   ROUND(AVG(CASE WHEN IsLate = 1 THEN DelayDays END), 1) AS avg_delay_when_late,
# MAGIC   SUM(HasScrap)                         AS orders_with_scrap,
# MAGIC   ROUND(100 * SUM(ScrappedQty) / SUM(OrderQty), 3) AS scrap_pct
# MAGIC FROM workspace.gold.fact_workorder;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   p.ProductGroup,
# MAGIC   COUNT(*)                          AS work_orders,
# MAGIC   ROUND(100 * AVG(f.IsLate), 1)     AS late_pct,
# MAGIC   ROUND(AVG(f.PlannedLeadDays), 1)  AS avg_planned_days,
# MAGIC   ROUND(AVG(f.ActualLeadDays), 1)   AS avg_actual_days
# MAGIC FROM workspace.gold.fact_workorder f
# MAGIC JOIN workspace.gold.dim_product p ON f.ProductID = p.ProductID
# MAGIC GROUP BY p.ProductGroup
# MAGIC ORDER BY late_pct DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Primary keys require NOT NULL first
# MAGIC ALTER TABLE workspace.gold.dim_product      ALTER COLUMN ProductID     SET NOT NULL;
# MAGIC ALTER TABLE workspace.gold.dim_date         ALTER COLUMN DateKey       SET NOT NULL;
# MAGIC ALTER TABLE workspace.gold.dim_scrap_reason ALTER COLUMN ScrapReasonID SET NOT NULL;
# MAGIC ALTER TABLE workspace.gold.dim_location     ALTER COLUMN LocationID    SET NOT NULL;
# MAGIC ALTER TABLE workspace.gold.fact_workorder   ALTER COLUMN WorkOrderID   SET NOT NULL;
# MAGIC
# MAGIC ALTER TABLE workspace.gold.dim_product      ADD CONSTRAINT pk_dim_product      PRIMARY KEY (ProductID);
# MAGIC ALTER TABLE workspace.gold.dim_date         ADD CONSTRAINT pk_dim_date         PRIMARY KEY (DateKey);
# MAGIC ALTER TABLE workspace.gold.dim_scrap_reason ADD CONSTRAINT pk_dim_scrap_reason PRIMARY KEY (ScrapReasonID);
# MAGIC ALTER TABLE workspace.gold.dim_location     ADD CONSTRAINT pk_dim_location     PRIMARY KEY (LocationID);
# MAGIC ALTER TABLE workspace.gold.fact_workorder   ADD CONSTRAINT pk_fact_workorder   PRIMARY KEY (WorkOrderID);
# MAGIC
# MAGIC ALTER TABLE workspace.gold.fact_workorder
# MAGIC   ADD CONSTRAINT fk_wo_product FOREIGN KEY (ProductID) REFERENCES workspace.gold.dim_product;
# MAGIC
# MAGIC ALTER TABLE workspace.gold.fact_workorder
# MAGIC   ADD CONSTRAINT fk_wo_scrap FOREIGN KEY (ScrapReasonID) REFERENCES workspace.gold.dim_scrap_reason;

# COMMAND ----------

(spark.table("workspace.gold.dim_scrap_reason")
   .withColumn("ScrapReasonID", F.col("ScrapReasonID").cast("int"))
   .write.mode("overwrite").option("overwriteSchema", "true")
   .saveAsTable("workspace.gold.dim_scrap_reason"))

spark.table("workspace.gold.dim_scrap_reason").printSchema()

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE workspace.gold.dim_scrap_reason ALTER COLUMN ScrapReasonID SET NOT NULL;
# MAGIC ALTER TABLE workspace.gold.dim_scrap_reason ADD CONSTRAINT pk_dim_scrap_reason PRIMARY KEY (ScrapReasonID);
# MAGIC ALTER TABLE workspace.gold.fact_workorder ADD CONSTRAINT fk_wo_scrap FOREIGN KEY (ScrapReasonID) REFERENCES workspace.gold.dim_scrap_reason;

# COMMAND ----------

no_scrap = spark.createDataFrame(
    [(0, "No Scrap")],
    "ScrapReasonID INT, ScrapReasonName STRING"
)

dim_scrap_reason = scrap.select("ScrapReasonID", "ScrapReasonName").union(no_scrap)

# COMMAND ----------

