# =============================================================
# HR Analytics AI System
# Module 7: Insights and HR Recommendations
# Responsible: yousef hany - 240100780
# =============================================================

from pyspark.sql import functions as F


def identify_high_risk_employees(df_clustered):
    """
    Identify employees with RiskScore above 0.6.

    Args:
        df_clustered: Clustered Spark DataFrame

    Returns:
        Spark DataFrame of high-risk employees
    """
    high_risk = df_clustered.filter(F.col("RiskScore") >= 0.6) \
        .select(
            "Age", "MonthlyIncome", "JobLevel",
            "TotalWorkingYears", "YearsAtCompany",
            "OverTime_encoded", "EngagementScore",
            "SatisfactionScore", "RiskScore", "cluster"
        ).orderBy("RiskScore", ascending=False)

    total = df_clustered.count()
    high_risk_count = high_risk.count()
    pct = high_risk_count / total * 100

    print("=" * 60)
    print("HIGH-RISK EMPLOYEE IDENTIFICATION")
    print("=" * 60)
    print(f"Total Employees:      {total}")
    print(f"High-Risk Employees:  {high_risk_count}")
    print(f"High-Risk Percentage: {pct:.1f}%")
    print("-" * 60)
    print("High-Risk Employees by Cluster:")
    high_risk.groupBy("cluster").count().orderBy("cluster").show()
    print("Top 10 Highest Risk Employees:")
    high_risk.show(10)

    return high_risk


def analyze_risk_factors(df):
    """
    Analyze key attrition risk factors.

    Args:
        df: Spark DataFrame

    Returns:
        None
    """
    print("=" * 60)
    print("KEY ATTRITION RISK FACTORS")
    print("=" * 60)

    print("\n1. Overtime Impact:")
    df.groupBy("OverTime_encoded").agg(
        F.count("*").alias("Total"),
        F.round(F.avg("Attrition_encoded") * 100, 1).alias("Attrition Rate %")
    ).orderBy("OverTime_encoded").show()

    print("2. Age Group Impact:")
    df.withColumn("AgeGroup",
        F.when(F.col("Age") < 25, "Under 25")
         .when((F.col("Age") >= 25) & (F.col("Age") < 35), "25-34")
         .when((F.col("Age") >= 35) & (F.col("Age") < 45), "35-44")
         .otherwise("45+")
    ).groupBy("AgeGroup").agg(
        F.count("*").alias("Total"),
        F.round(F.avg("Attrition_encoded") * 100, 1).alias("Attrition Rate %")
    ).orderBy("AgeGroup").show()

    print("3. Income Group Impact:")
    df.withColumn("IncomeGroup",
        F.when(F.col("MonthlyIncome") <= 3000, "Low")
         .when((F.col("MonthlyIncome") > 3000) &
               (F.col("MonthlyIncome") <= 7000), "Medium")
         .otherwise("High")
    ).groupBy("IncomeGroup").agg(
        F.count("*").alias("Total"),
        F.round(F.avg("Attrition_encoded") * 100, 1).alias("Attrition Rate %")
    ).orderBy("IncomeGroup").show()


def print_hr_recommendations():
    """
    Print strategic HR recommendations.
    """
    print("=" * 60)
    print("STRATEGIC HR RECOMMENDATIONS")
    print("=" * 60)

    recommendations = {
        "1. Overtime Policy Reform": [
            "Cap overtime hours to a maximum of 10 hours per week",
            "Implement overtime compensation and time-off policies",
            "Monitor overtime patterns and flag employees exceeding limits",
            "Redistribute workload to reduce dependency on overtime"
        ],
        "2. Compensation Review": [
            "Conduct immediate salary benchmarking for low-income employees",
            "Implement performance-based salary increments",
            "Review compensation for employees earning below $3,000 per month",
            "Introduce stock options and benefits for entry-level employees"
        ],
        "3. Young Employee Retention": [
            "Create structured career development programs for under-35 employees",
            "Implement mentorship programs pairing juniors with senior employees",
            "Offer education assistance and professional certification support",
            "Establish clear promotion timelines and criteria"
        ],
        "4. Engagement Improvement": [
            "Launch quarterly employee satisfaction surveys",
            "Create cross-functional project opportunities",
            "Implement recognition and rewards programs",
            "Establish open-door communication policies with management"
        ],
        "5. High-Risk Employee Monitoring": [
            "Immediately flag and monitor all 194 identified high-risk employees",
            "Schedule monthly check-ins with high-risk employees",
            "Offer personalized retention packages for critical cases",
            "Track risk score changes monthly and update intervention plans"
        ]
    }

    for strategy, actions in recommendations.items():
        print(f"\n{strategy}:")
        for i, action in enumerate(actions, 1):
            print(f"  {i}. {action}")
        print("-" * 60)


def generate_insights(df, df_clustered):
    """
    Run the full insights and recommendations pipeline.

    Args:
        df: Feature engineered Spark DataFrame
        df_clustered: Clustered Spark DataFrame

    Returns:
        Spark DataFrame of high-risk employees
    """
    print("=" * 60)
    print("INSIGHTS AND HR RECOMMENDATIONS")
    print("=" * 60)

    high_risk = identify_high_risk_employees(df_clustered)
    analyze_risk_factors(df)
    print_hr_recommendations()

    print("=" * 60)
    print("INSIGHTS COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return high_risk