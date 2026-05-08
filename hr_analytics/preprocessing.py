# =============================================================
# HR Analytics AI System
# Module 2: Data Preprocessing
# Responsible: Omar Ahmed Wafik - 240101244
# =============================================================

from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline


def check_missing_values(df):
    """
    Check for missing values in all columns.

    Args:
        df: Spark DataFrame

    Returns:
        None
    """
    print("=" * 60)
    print("MISSING VALUES CHECK")
    print("=" * 60)
    missing = df.select([
        F.count(F.when(F.col(c).isNull(), c)).alias(c)
        for c in df.columns
    ])
    missing.show(vertical=True)


def check_duplicates(df):
    """
    Check and remove duplicate records.

    Args:
        df: Spark DataFrame

    Returns:
        Spark DataFrame with duplicates removed
    """
    total = df.count()
    unique = df.distinct().count()
    duplicates = total - unique

    print("=" * 60)
    print("DUPLICATE RECORDS CHECK")
    print("=" * 60)
    print(f"Total Rows:   {total}")
    print(f"Unique Rows:  {unique}")
    print(f"Duplicates:   {duplicates}")

    if duplicates > 0:
        df = df.distinct()
        print(f"Duplicates removed. Remaining Rows: {df.count()}")
    else:
        print("No duplicate records found. Dataset is clean.")

    return df


def drop_irrelevant_columns(df):
    """
    Drop columns that have no predictive value.

    Args:
        df: Spark DataFrame

    Returns:
        Spark DataFrame with irrelevant columns removed
    """
    cols_to_drop = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]
    df = df.drop(*cols_to_drop)

    print("=" * 60)
    print("IRRELEVANT COLUMNS DROPPED")
    print("=" * 60)
    print(f"Dropped Columns: {cols_to_drop}")
    print(f"Remaining Columns: {len(df.columns)}")
    return df


def encode_categorical_columns(df):
    """
    Encode categorical columns using StringIndexer.

    Args:
        df: Spark DataFrame

    Returns:
        Spark DataFrame with encoded categorical columns
    """
    categorical_cols = [
        "Attrition", "OverTime", "Gender",
        "BusinessTravel", "Department",
        "EducationField", "JobRole", "MaritalStatus"
    ]

    indexers = [
        StringIndexer(
            inputCol=col,
            outputCol=col + "_encoded",
            handleInvalid="keep"
        )
        for col in categorical_cols
    ]

    pipeline = Pipeline(stages=indexers)
    df_encoded = pipeline.fit(df).transform(df)
    df_clean = df_encoded.drop(*categorical_cols)

    print("=" * 60)
    print("CATEGORICAL ENCODING COMPLETED")
    print("=" * 60)
    for col in categorical_cols:
        print(f"  {col} → {col}_encoded")
    print(f"Final Columns: {len(df_clean.columns)}")
    return df_clean


def preprocess_data(df):
    """
    Run the full preprocessing pipeline.

    Args:
        df: Raw Spark DataFrame

    Returns:
        Preprocessed Spark DataFrame
    """
    check_missing_values(df)
    df = check_duplicates(df)
    df = drop_irrelevant_columns(df)
    df = encode_categorical_columns(df)
    print("=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print(f"Final Shape: {df.count()} rows x {len(df.columns)} columns")
    print("=" * 60)
    return df