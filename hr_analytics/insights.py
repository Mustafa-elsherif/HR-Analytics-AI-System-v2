# =============================================================
# HR Analytics AI System
# Module 7: Insights and HR Recommendations
# Responsible: yousef hany - 240102846
# =============================================================

from pyspark.sql import functions as F
import matplotlib.pyplot as plt
import os


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


def plot_high_risk_analysis(df_clustered, high_risk):
    """
    Plot high-risk employee analysis visualizations.

    Args:
        df_clustered: Clustered Spark DataFrame
        high_risk: High-risk employees Spark DataFrame
    """
    cluster_pdf = df_clustered.select("RiskScore", "cluster").toPandas()
    high_risk_pdf = high_risk.toPandas()

    colors = {0: "steelblue", 1: "tomato", 2: "seagreen"}
    labels = {0: "Mid-Career Stable", 1: "Young High-Risk", 2: "Senior High-Income"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for cluster in [0, 1, 2]:
        subset = cluster_pdf[cluster_pdf["cluster"] == cluster]["RiskScore"]
        axes[0].hist(subset, bins=20, alpha=0.6,
                     color=colors[cluster], label=labels[cluster], edgecolor="black")
    axes[0].axvline(x=0.6, color="black", linestyle="--",
                    linewidth=2, label="High-Risk Threshold")
    axes[0].set_title("Risk Score Distribution by Cluster", fontweight="bold")
    axes[0].set_xlabel("Risk Score")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    hr_cluster = high_risk_pdf["cluster"].value_counts().sort_index()
    axes[1].bar([labels[i] for i in hr_cluster.index],
                hr_cluster.values,
                color=[colors[i] for i in hr_cluster.index],
                edgecolor="black")
    axes[1].set_title("High-Risk Employees by Cluster", fontweight="bold")
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("Number of High-Risk Employees")
    axes[1].tick_params(axis='x', rotation=15)
    for i, v in enumerate(hr_cluster.values):
        axes[1].text(i, v + 1, str(v), ha="center", fontweight="bold")

    normal_pdf = cluster_pdf[cluster_pdf["RiskScore"] < 0.6]
    axes[2].scatter(normal_pdf.index, normal_pdf["RiskScore"],
                    color="steelblue", alpha=0.4, label="Normal Risk", s=10)
    axes[2].scatter(high_risk_pdf.index, high_risk_pdf["RiskScore"],
                    color="tomato", alpha=0.8, label="High Risk", s=20)
    axes[2].axhline(y=0.6, color="black", linestyle="--",
                    linewidth=2, label="Threshold")
    axes[2].set_title("Risk Score - All Employees", fontweight="bold")
    axes[2].set_xlabel("Employee Index")
    axes[2].set_ylabel("Risk Score")
    axes[2].legend()

    plt.suptitle("High-Risk Employee Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "high_risk_analysis.png"), dpi=150)
    plt.show()


def analyze_risk_factors(df):
    """
    Analyze key attrition risk factors.

    Args:
        df: Spark DataFrame
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
    plot_high_risk_analysis(df_clustered, high_risk)
    analyze_risk_factors(df)
    print_hr_recommendations()

    print("=" * 60)
    print("INSIGHTS COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return high_risk