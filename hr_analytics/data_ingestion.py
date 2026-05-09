# =============================================================
# HR Analytics AI System
# Module 1: Data Ingestion
# Responsible: yousef hany - 240100780
# =============================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def create_spark_session():
    """Initialize and return a Spark Session."""
    spark = SparkSession.builder \
        .appName("HR Analytics AI System") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()
    try:
        spark.sparkContext.setLogLevel("ERROR")
    except Exception:
        pass
    return spark


def load_data(spark, filepath):
    """
    Load the IBM HR Analytics Dataset using Apache Spark.

    Args:
        spark: Active SparkSession
        filepath: Path to the CSV dataset

    Returns:
        Spark DataFrame containing the raw dataset
    """
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    return df


def explore_data(df):
    """
    Display basic information about the dataset.

    Args:
        df: Spark DataFrame
    """
    print("=" * 60)
    print("DATA INGESTION - SUMMARY")
    print("=" * 60)
    print(f"Total Rows:    {df.count()}")
    print(f"Total Columns: {len(df.columns)}")
    print("-" * 60)
    print("Schema:")
    df.printSchema()
    print("-" * 60)
    print("First 5 Rows:")
    df.show(5, truncate=True)
    print("=" * 60)
    print("Data Ingestion Completed Successfully.")
    print("=" * 60)


def save_as_parquet(df, output_path):
    """
    Save the DataFrame in Parquet format.

    Args:
        df: Spark DataFrame
        output_path: Path to save the Parquet file
    """
    df.write.mode("overwrite").parquet(output_path)
    print(f"Data saved as Parquet: {output_path}")