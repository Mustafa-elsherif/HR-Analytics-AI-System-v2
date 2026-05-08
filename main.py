# =============================================================
# HR Analytics AI System
# Main Entry Point
# =============================================================
# Team Members:
# - Mustafa Nabil       240103415  Model Development, Clustering, Dashboard
# - Mahmoud Hesham      240101375  Model Development, Clustering, Dashboard
# - Mariam Magdy        240102374  Exploratory Data Analysis
# - Omar Ahmed Wafik    240101244  Preprocessing, Feature Engineering
# - yousef hany        240100780  Data Ingestion, Insights
# =============================================================

import os
from hr_analytics.data_ingestion import (
    create_spark_session,
    load_data,
    explore_data,
    save_as_parquet
)
from hr_analytics.preprocessing import preprocess_data
from hr_analytics.eda import run_eda
from hr_analytics.feature_engineering import engineer_features
from hr_analytics.models import train_models
from hr_analytics.clustering import run_clustering
from hr_analytics.insights import generate_insights

# =============================================================
# CONFIGURATION
# =============================================================
DATA_PATH = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
PARQUET_RAW = "data/hr_data.parquet"
PARQUET_PREPROCESSED = "data/hr_data_preprocessed.parquet"
PARQUET_ENGINEERED = "data/hr_data_engineered.parquet"


def main():
    print("=" * 60)
    print("HR ANALYTICS AI SYSTEM")
    print("AI-Powered Employee Analytics and Decision Support")
    print("=" * 60)

    # Create outputs folder if not exists
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Step 1: Data Ingestion
    print("\nStep 1: Data Ingestion")
    spark = create_spark_session()
    df_raw = load_data(spark, DATA_PATH)
    explore_data(df_raw)
    save_as_parquet(df_raw, PARQUET_RAW)

    # Step 2: Preprocessing
    print("\nStep 2: Data Preprocessing")
    df_preprocessed = preprocess_data(df_raw)
    save_as_parquet(df_preprocessed, PARQUET_PREPROCESSED)

    # Step 3: EDA
    print("\nStep 3: Exploratory Data Analysis")
    run_eda(df_preprocessed, df_raw)

    # Step 4: Feature Engineering
    print("\nStep 4: Feature Engineering")
    df_engineered = engineer_features(df_preprocessed)
    save_as_parquet(df_engineered, PARQUET_ENGINEERED)

    # Step 5: Model Development
    print("\nStep 5: Model Development")
    attrition_model, performance_model, clf_metrics, reg_metrics = \
        train_models(df_engineered)

    # Step 6: Clustering
    print("\nStep 6: Employee Segmentation")
    clustering_model, df_clustered, silhouette = run_clustering(df_engineered)

    # Step 7: Insights
    print("\nStep 7: Insights and HR Recommendations")
    high_risk = generate_insights(df_engineered, df_clustered)

    # Final Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"Attrition Model Accuracy:  {clf_metrics['accuracy']*100:.2f}%")
    print(f"Attrition Model F1-Score:  {clf_metrics['f1']:.4f}")
    print(f"Performance Model RMSE:    {reg_metrics['rmse']:.4f}")
    print(f"Performance Model R2:      {reg_metrics['r2']:.4f}")
    print(f"Clustering Silhouette:     {silhouette:.4f}")
    print(f"High-Risk Employees:       {high_risk.count()}")
    print("=" * 60)
    print("HR Analytics AI System Completed Successfully.")
    print("=" * 60)

    spark.stop()


if __name__ == "__main__":
    main()