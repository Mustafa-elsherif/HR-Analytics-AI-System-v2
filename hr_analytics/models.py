# =============================================================
# HR Analytics AI System
# Module 5: Model Development
# Responsible: Mustafa Nabil - 240103415
#              Mahmoud Hesham - 240101375
# =============================================================

from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    RegressionEvaluator
)


FEATURE_COLS = [
    "Age", "DailyRate", "DistanceFromHome", "Education",
    "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement",
    "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate",
    "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
    "OverTime_encoded", "Gender_encoded", "BusinessTravel_encoded",
    "Department_encoded", "EducationField_encoded", "JobRole_encoded",
    "MaritalStatus_encoded", "EngagementScore", "SatisfactionScore",
    "RiskScore", "ExperienceLevel_encoded", "IncomeLevel_encoded"
]


def prepare_features(df):
    """
    Assemble feature columns into a single vector.

    Args:
        df: Spark DataFrame

    Returns:
        Spark DataFrame with features vector column
    """
    assembler = VectorAssembler(
        inputCols=FEATURE_COLS,
        outputCol="features"
    )
    return assembler.transform(df)


def train_attrition_model(df):
    """
    Train Random Forest Classifier for attrition prediction.

    Args:
        df: Spark DataFrame with features vector

    Returns:
        tuple: (model, predictions, test_df)
    """
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    total = df.count()
    positive = df.filter(F.col("Attrition_encoded") == 1).count()
    negative = df.filter(F.col("Attrition_encoded") == 0).count()
    weight_positive = total / (2 * positive)
    weight_negative = total / (2 * negative)

    train_df = train_df.withColumn(
        "classWeight",
        F.when(F.col("Attrition_encoded") == 1, weight_positive)
         .otherwise(weight_negative)
    )

    rf_classifier = RandomForestClassifier(
        labelCol="Attrition_encoded",
        featuresCol="features",
        numTrees=100,
        maxDepth=10,
        seed=42,
        weightCol="classWeight"
    )

    model = rf_classifier.fit(train_df)
    predictions = model.transform(test_df)
    return model, predictions, test_df


def evaluate_attrition_model(predictions):
    """
    Evaluate the attrition classification model.

    Args:
        predictions: Spark DataFrame with predictions

    Returns:
        dict: Dictionary of evaluation metrics
    """
    accuracy_evaluator = MulticlassClassificationEvaluator(
        labelCol="Attrition_encoded",
        predictionCol="prediction",
        metricName="accuracy"
    )
    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="Attrition_encoded",
        predictionCol="prediction",
        metricName="f1"
    )
    precision_evaluator = MulticlassClassificationEvaluator(
        labelCol="Attrition_encoded",
        predictionCol="prediction",
        metricName="weightedPrecision"
    )
    recall_evaluator = MulticlassClassificationEvaluator(
        labelCol="Attrition_encoded",
        predictionCol="prediction",
        metricName="weightedRecall"
    )

    metrics = {
        "accuracy": accuracy_evaluator.evaluate(predictions),
        "f1": f1_evaluator.evaluate(predictions),
        "precision": precision_evaluator.evaluate(predictions),
        "recall": recall_evaluator.evaluate(predictions)
    }

    print("=" * 60)
    print("ATTRITION MODEL - CLASSIFICATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print("=" * 60)

    print("\nConfusion Matrix:")
    predictions.groupBy("Attrition_encoded", "prediction") \
        .count().orderBy("Attrition_encoded", "prediction").show()

    return metrics


def train_performance_model(df):
    """
    Train Random Forest Regressor for performance prediction.

    Args:
        df: Spark DataFrame with features vector

    Returns:
        tuple: (model, predictions)
    """
    feature_cols_reg = [col for col in FEATURE_COLS if col != "PerformanceRating"]

    assembler_reg = VectorAssembler(
        inputCols=feature_cols_reg,
        outputCol="features_reg"
    )
    df_reg = assembler_reg.transform(df)
    train_reg, test_reg = df_reg.randomSplit([0.8, 0.2], seed=42)

    rf_regressor = RandomForestRegressor(
        labelCol="PerformanceRating",
        featuresCol="features_reg",
        numTrees=100,
        maxDepth=10,
        seed=42
    )

    model = rf_regressor.fit(train_reg)
    predictions = model.transform(test_reg)
    return model, predictions


def evaluate_performance_model(predictions):
    """
    Evaluate the performance regression model.

    Args:
        predictions: Spark DataFrame with predictions

    Returns:
        dict: Dictionary of evaluation metrics
    """
    rmse_evaluator = RegressionEvaluator(
        labelCol="PerformanceRating",
        predictionCol="prediction",
        metricName="rmse"
    )
    mae_evaluator = RegressionEvaluator(
        labelCol="PerformanceRating",
        predictionCol="prediction",
        metricName="mae"
    )
    r2_evaluator = RegressionEvaluator(
        labelCol="PerformanceRating",
        predictionCol="prediction",
        metricName="r2"
    )

    metrics = {
        "rmse": rmse_evaluator.evaluate(predictions),
        "mae": mae_evaluator.evaluate(predictions),
        "r2": r2_evaluator.evaluate(predictions)
    }

    print("=" * 60)
    print("PERFORMANCE MODEL - REGRESSION RESULTS")
    print("=" * 60)
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"R2:   {metrics['r2']:.4f}")
    print("=" * 60)

    return metrics


def save_model(model, path):
    """
    Save a trained model to disk.

    Args:
        model: Trained Spark ML model
        path: Path to save the model
    """
    model.write().overwrite().save(path)
    print(f"Model saved: {path}")


def train_models(df):
    """
    Run the full model training pipeline.

    Args:
        df: Feature engineered Spark DataFrame

    Returns:
        tuple: (attrition_model, performance_model, classification_metrics, regression_metrics)
    """
    print("=" * 60)
    print("MODEL DEVELOPMENT")
    print("=" * 60)

    df_model = prepare_features(df)

    print("\nTraining Attrition Prediction Model...")
    attrition_model, attrition_predictions, _ = train_attrition_model(df_model)
    classification_metrics = evaluate_attrition_model(attrition_predictions)

    print("\nTraining Performance Prediction Model...")
    performance_model, performance_predictions = train_performance_model(df)
    regression_metrics = evaluate_performance_model(performance_predictions)

    save_model(attrition_model, "models/attrition_model")
    save_model(performance_model, "models/performance_model")

    print("=" * 60)
    print("MODEL DEVELOPMENT COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return attrition_model, performance_model, classification_metrics, regression_metrics