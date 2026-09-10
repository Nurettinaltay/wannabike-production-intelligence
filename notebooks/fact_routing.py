# Databricks notebook source
from pyspark.sql import functions as F

routing = spark.table("workspace.silver.workorder_routing")

fact_routing = (routing
    .withColumn("ScheduledDurationDays", F.datediff("ScheduledEndDate", "ScheduledStartDate"))
    .withColumn("ActualDurationDays",    F.datediff("ActualEndDate",    "ActualStartDate"))
    .withColumn("DurationVarianceDays",
        F.datediff("ActualEndDate", "ActualStartDate") - F.datediff("ScheduledEndDate", "ScheduledStartDate"))
    .withColumn("StartDelayDays",  F.datediff("ActualStartDate", "ScheduledStartDate"))
    .withColumn("FinishDelayDays", F.datediff("ActualEndDate",   "ScheduledEndDate"))
    .withColumn("IsOperationLate", (F.col("ActualEndDate") > F.col("ScheduledEndDate")).cast("int"))
    .select(
        "WorkOrderID", "ProductID", "OperationSequence", "LocationID",
        "ScheduledStartDate", "ScheduledEndDate",
        "ActualStartDate", "ActualEndDate",
        "ActualResourceHrs", "PlannedCost", "ActualCost",
        "ScheduledDurationDays", "ActualDurationDays", "DurationVarianceDays",
        "StartDelayDays", "FinishDelayDays", "IsOperationLate"
    )
)

fact_routing.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("workspace.gold.fact_routing")

print("fact_routing:", fact_routing.count())

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   l.LocationName,
# MAGIC   COUNT(*)                                   AS operations,
# MAGIC   ROUND(SUM(r.ActualResourceHrs), 0)         AS total_hours,
# MAGIC   ROUND(100 * AVG(r.IsOperationLate), 1)     AS late_pct,
# MAGIC   ROUND(AVG(r.StartDelayDays), 2)            AS avg_start_delay,
# MAGIC   ROUND(AVG(r.DurationVarianceDays), 2)      AS avg_duration_variance,
# MAGIC   l.DailyCapacityHrs,
# MAGIC   l.CostRate
# MAGIC FROM workspace.gold.fact_routing r
# MAGIC JOIN workspace.gold.dim_location l ON r.LocationID = l.LocationID
# MAGIC GROUP BY l.LocationName, l.DailyCapacityHrs, l.CostRate
# MAGIC ORDER BY total_hours DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   l.LocationName,
# MAGIC   r.OperationSequence,
# MAGIC   COUNT(*)                                AS operations,
# MAGIC   ROUND(AVG(r.StartDelayDays), 2)         AS avg_start_delay,
# MAGIC   ROUND(AVG(r.DurationVarianceDays), 2)   AS avg_duration_variance,
# MAGIC   ROUND(100 * AVG(r.IsOperationLate), 1)  AS late_pct
# MAGIC FROM workspace.gold.fact_routing r
# MAGIC JOIN workspace.gold.dim_location l ON r.LocationID = l.LocationID
# MAGIC JOIN workspace.gold.dim_product  p ON r.ProductID  = p.ProductID
# MAGIC WHERE p.ProductGroup = 'Finished Bikes'
# MAGIC GROUP BY l.LocationName, r.OperationSequence
# MAGIC ORDER BY r.OperationSequence;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   d.Year,
# MAGIC   d.Quarter,
# MAGIC   COUNT(*)                              AS operations,
# MAGIC   ROUND(AVG(r.StartDelayDays), 2)       AS avg_start_delay,
# MAGIC   ROUND(AVG(r.DurationVarianceDays), 2) AS avg_duration_variance,
# MAGIC   ROUND(100 * AVG(r.IsOperationLate),1) AS late_pct
# MAGIC FROM workspace.gold.fact_routing r
# MAGIC JOIN workspace.gold.dim_date d ON r.ActualStartDate = d.DateKey
# MAGIC GROUP BY d.Year, d.Quarter
# MAGIC ORDER BY d.Year, d.Quarter;

# COMMAND ----------

fact_routing = fact_routing.withColumn(
    "IsSameWeek",
    (F.weekofyear("ScheduledEndDate") == F.weekofyear("ActualEndDate")) &
    (F.year("ScheduledEndDate") == F.year("ActualEndDate"))
).withColumn("IsSameWeek", F.col("IsSameWeek").cast("int"))

# COMMAND ----------

from pyspark.sql import functions as F

fact_routing = (spark.table("workspace.gold.fact_routing")
    .withColumn("IsSameWeek",
        ((F.weekofyear("ScheduledEndDate") == F.weekofyear("ActualEndDate")) &
         (F.year("ScheduledEndDate")       == F.year("ActualEndDate"))).cast("int"))
)

fact_routing.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("workspace.gold.fact_routing")
print("fact_routing:", fact_routing.count())

# COMMAND ----------

# 2% scrap threshold — acceptance limit defined by production management (user story 2)
fact_workorder = (spark.table("workspace.gold.fact_workorder")
    .withColumn("ExceedsScrapThreshold", (F.col("ScrapRate") > 0.02).cast("int"))
)

fact_workorder.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("workspace.gold.fact_workorder")
print("fact_workorder:", fact_workorder.count())

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   d.Year,
# MAGIC   d.YearWeek,
# MAGIC   COUNT(*)                                  AS operations,
# MAGIC   SUM(r.IsSameWeek)                         AS delivered_same_week,
# MAGIC   SUM(CASE WHEN r.IsSameWeek = 0 THEN 1 ELSE 0 END) AS delayed_operations,
# MAGIC   ROUND(100 * AVG(r.IsSameWeek), 1)         AS same_week_pct
# MAGIC FROM workspace.gold.fact_routing r
# MAGIC JOIN workspace.gold.dim_date d ON r.ActualEndDate = d.DateKey
# MAGIC GROUP BY d.Year, d.YearWeek
# MAGIC ORDER BY d.Year, d.YearWeek
# MAGIC LIMIT 20;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   COUNT(*)                                          AS total_work_orders,
# MAGIC   SUM(ExceedsScrapThreshold)                        AS above_2pct,
# MAGIC   ROUND(100 * AVG(ExceedsScrapThreshold), 2)        AS above_2pct_share,
# MAGIC   ROUND(100 * SUM(ScrappedQty) / SUM(OrderQty), 3)  AS overall_scrap_pct
# MAGIC FROM workspace.gold.fact_workorder;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   s.ScrapReasonName,
# MAGIC   COUNT(*)                            AS work_orders,
# MAGIC   SUM(f.ScrappedQty)                  AS scrapped_units,
# MAGIC   ROUND(100 * AVG(f.ScrapRate), 2)    AS avg_scrap_rate_pct
# MAGIC FROM workspace.gold.fact_workorder f
# MAGIC JOIN workspace.gold.dim_scrap_reason s ON f.ScrapReasonID = s.ScrapReasonID
# MAGIC WHERE f.ScrapReasonID <> 0
# MAGIC GROUP BY s.ScrapReasonName
# MAGIC ORDER BY scrapped_units DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   d.Year,
# MAGIC   d.Quarter,
# MAGIC   COUNT(*)              AS completed_work_orders,
# MAGIC   SUM(f.OrderQty)       AS ordered_qty,
# MAGIC   SUM(f.StockedQty)     AS stocked_qty,
# MAGIC   SUM(f.ScrappedQty)    AS scrapped_qty
# MAGIC FROM workspace.gold.fact_workorder f
# MAGIC JOIN workspace.gold.dim_date d ON f.EndDate = d.DateKey
# MAGIC GROUP BY d.Year, d.Quarter
# MAGIC ORDER BY d.Year, d.Quarter;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   d.Month,
# MAGIC   d.MonthName,
# MAGIC   COUNT(*)                                          AS work_orders,
# MAGIC   SUM(f.ScrappedQty)                                AS scrapped_units,
# MAGIC   ROUND(100 * SUM(f.ScrappedQty) / SUM(f.OrderQty), 3) AS scrap_pct,
# MAGIC   ROUND(100 * AVG(f.IsLate), 1)                     AS late_pct
# MAGIC FROM workspace.gold.fact_workorder f
# MAGIC JOIN workspace.gold.dim_date d ON f.EndDate = d.DateKey
# MAGIC GROUP BY d.Month, d.MonthName
# MAGIC ORDER BY d.Month;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   d.Year,
# MAGIC   d.MonthName,
# MAGIC   COUNT(*)                                          AS work_orders,
# MAGIC   SUM(f.HasScrap)                                   AS orders_with_scrap,
# MAGIC   SUM(f.ScrappedQty)                                AS scrapped_units,
# MAGIC   MAX(f.ScrappedQty)                                AS largest_single_scrap,
# MAGIC   ROUND(100 * SUM(f.ScrappedQty)/SUM(f.OrderQty),3) AS scrap_pct
# MAGIC FROM workspace.gold.fact_workorder f
# MAGIC JOIN workspace.gold.dim_date d ON f.EndDate = d.DateKey
# MAGIC WHERE d.Month = 9
# MAGIC GROUP BY d.Year, d.MonthName
# MAGIC ORDER BY d.Year;

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH ranked AS (
# MAGIC   SELECT
# MAGIC     ScrappedQty,
# MAGIC     SUM(ScrappedQty) OVER ()                                  AS total_scrap,
# MAGIC     SUM(ScrappedQty) OVER (ORDER BY ScrappedQty DESC
# MAGIC                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_scrap,
# MAGIC     ROW_NUMBER()     OVER (ORDER BY ScrappedQty DESC)         AS rn
# MAGIC   FROM workspace.gold.fact_workorder
# MAGIC   WHERE HasScrap = 1
# MAGIC )
# MAGIC SELECT
# MAGIC   rn                                              AS top_n_orders,
# MAGIC   running_scrap,
# MAGIC   total_scrap,
# MAGIC   ROUND(100 * running_scrap / total_scrap, 1)     AS cumulative_pct
# MAGIC FROM ranked
# MAGIC WHERE rn IN (10, 25, 50, 100, 200, 729)
# MAGIC ORDER BY rn;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   ROUND(100 * SUM(StockedQty) / SUM(OrderQty), 2) AS production_efficiency_pct,
# MAGIC   SUM(OrderQty)   AS ordered,
# MAGIC   SUM(StockedQty) AS stocked,
# MAGIC   SUM(ScrappedQty) AS scrapped,
# MAGIC   SUM(OrderQty) - SUM(StockedQty) - SUM(ScrappedQty) AS unexplained_gap
# MAGIC FROM workspace.gold.fact_workorder;

# COMMAND ----------

