# =============================================================
# HR Analytics AI System
# Module 4: Feature Engineering
# Responsible: Omar Ahmed Wafik - 240101244
# =============================================================

from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline


def create_engagement_score(df):
    """
    Create Engagement Score from JobInvolvement, JobSatisfaction,
    and RelationshipSatisfaction.

    Args:
        df: Spark DataFrame

    Returns:
        Spark DataFrame with EngagementScore column added
    """
    df = df.withColumn(
        "EngagementScore",
        F.round(
            (F.col("JobInvolvement") +
             F.col("JobSatisfaction") +
             F.col("RelationshipSatisfaction")) / 3, 2
        )
    )
    return df


def create_satisfaction_score(df):
    """
    Create Satisfaction Score from EnvironmentSatisfaction,
    JobSatisfaction, RelationshipSatisfaction, and WorkLifeBalance.

    Args:
        df: Spark DataFrame

    Returns:
        Spark DataFrame with SatisfactionScore column added
    """
    df = df.withColumn(
        "SatisfactionScore",
        F.round(
            (F.col("EnvironmentSatisfaction") +
             F.col("JobSatisfaction") +
             F.col("RelationshipSatisfaction") +
             F.col("WorkLifeBalance")) / 4, 2
        )
    )
    return df


def create_experience_level(df):
    """
    Categorize employees based on total working years.

    Args:
        df: Spark DataFrame

    Returns:
        Spark DataFrame with ExperienceLevel column added
    """
    df = df.withColumn(
        "ExperienceLevel",
        F.when(F.col("TotalWorkingYears") <= 2, "Entry Level")
         .when((F.col("TotalWorkingYears") > 2) &
               (F.col("TotalWorkingYears") <= 7), "Junior")
         .when((F.col("TotalWorkingYears") > 7) &
               (F.col("TotalWorkingYears") <= 15), "Mid-Level")
         .when((F.col("TotalWorkingYears") > 15) &
               (F.col("TotalWorkingYears") <= 25), "Senior")
         .otherwise("Executive")
    )
    return df


def create_income_level(df):
    """
    Categorize employees based on monthly income.

    Args:
        df: Spark DataFrame

    Returns:
        Spark DataFrame with IncomeLevel column added
    """
    df = df.withColumn(
        "IncomeLevel",
        F.when(F.col("MonthlyIncome") <= 3000, "Low")
         .when((F.col("MonthlyIncome") > 3000) &
               (F.col("MonthlyIncome") <= 7000), "Medium")
         .when((F.col("MonthlyIncome") > 7000) &
               (F.col("MonthlyIncome") <= 13000), "High")
         .otherwise("Very High")
    )
    return df


def create_risk_score(df):
    """
    Create Risk Score based on key attrition predictors.

    Args:
        df: Spark DataFrame

    Returns:
        Spark DataFrame with RiskScore column added
    """
    df = df.withColumn(
        "RiskScore",
        F.round(
            (
                (F.col("OverTime_encoded") * 0.30) +
                ((4 - F.col("EngagementScore")) / 4 * 0.25) +
                ((4 - F.col("SatisfactionScore")) / 4 * 0.20) +
                (F.when(F.col("YearsAtCompany") <= 2, 1.0)
                  .when(F.col("YearsAtCompany") <= 5, 0.7)
                  .when(F.col("YearsAtCompany") <= 10, 0.4)
                  .otherwise(0.1) * 0.15) +
                (F.when(F.col("MonthlyIncome") <= 3000, 1.0)
                  .when(F.col("MonthlyIncome") <= 7000, 0.6)
                  .when(F.col("MonthlyIncome") <= 13000, 0.3)
                  .otherwise(0.1) * 0.10)
            ), 4
        )
    )
    return df


def encode_new_features(df):
    """
    Encode the newly created categorical features.

    Args:
        df: Spark DataFrame

    Returns:
        Spark DataFrame with encoded new features
    """
    indexers = [
        StringIndexer(
            inputCol="ExperienceLevel",
            outputCol="ExperienceLevel_encoded",
            handleInvalid="keep"
        ),
        StringIndexer(
            inputCol="IncomeLevel",
            outputCol="IncomeLevel_encoded",
            handleInvalid="keep"
        )
    ]
    pipeline = Pipeline(stages=indexers)
    df = pipeline.fit(df).transform(df)
    df = df.drop("ExperienceLevel", "IncomeLevel")
    return df


def engineer_features(df):
    """
    Run the full feature engineering pipeline.

    Args:
        df: Preprocessed Spark DataFrame

    Returns:
        Spark DataFrame with all engineered features
    """
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    df = create_engagement_score(df)
    print("EngagementScore created.")
    df = create_satisfaction_score(df)
    print("SatisfactionScore created.")
    df = create_experience_level(df)
    print("ExperienceLevel created.")
    df = create_income_level(df)
    print("IncomeLevel created.")
    df = create_risk_score(df)
    print("RiskScore created.")
    df = encode_new_features(df)
    print("New features encoded.")
    print("=" * 60)
    print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
    print(f"Total Columns: {len(df.columns)}")
    print("=" * 60)
    return df