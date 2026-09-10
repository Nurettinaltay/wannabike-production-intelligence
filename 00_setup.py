# Databricks notebook source
# MAGIC %sql
# MAGIC CREATE SCHEMA IF NOT EXISTS workspace.bronze;
# MAGIC CREATE SCHEMA IF NOT EXISTS workspace.silver;
# MAGIC CREATE SCHEMA IF NOT EXISTS workspace.gold;
# MAGIC
# MAGIC CREATE VOLUME IF NOT EXISTS workspace.bronze.raw_files;

# COMMAND ----------

