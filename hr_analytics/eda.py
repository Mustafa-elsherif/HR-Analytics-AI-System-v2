# =============================================================
# HR Analytics AI System
# Module 3: Exploratory Data Analysis
# Responsible: Mariam Magdy - 240102374
# =============================================================

from pyspark.sql import functions as F
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.style.use('seaborn-v0_8')
OUTPUTS_PATH = "outputs"


def plot_attrition_distribution(df_original):
    """
    Plot the attrition distribution as bar and pie charts.

    Args:
        df_original: Original Spark DataFrame with categorical columns
    """
    attrition_df = df_original.groupBy("Attrition") \
        .count().orderBy("count", ascending=False).toPandas()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(attrition_df["Attrition"], attrition_df["count"],
                color=["steelblue", "tomato"], edgecolor="black")
    axes[0].set_title("Attrition Count", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Attrition")
    axes[0].set_ylabel("Number of Employees")
    for i, v in enumerate(attrition_df["count"]):
        axes[0].text(i, v + 10, str(v), ha="center", fontweight="bold")

    axes[1].pie(attrition_df["count"], labels=attrition_df["Attrition"],
                autopct="%1.1f%%", colors=["steelblue", "tomato"],
                startangle=90, explode=[0, 0.05])
    axes[1].set_title("Attrition Percentage", fontsize=14, fontweight="bold")

    plt.suptitle("Employee Attrition Distribution", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_PATH, "attrition_distribution.png"), dpi=150)
    plt.show()
    print("Attrition Distribution:")
    print(attrition_df.to_string(index=False))


def plot_attrition_by_department(df_original):
    """
    Plot attrition count and rate by department.

    Args:
        df_original: Original Spark DataFrame
    """
    dept_attrition = df_original.groupBy("Department", "Attrition") \
        .count().orderBy("Department").toPandas()

    dept_pivot = dept_attrition.pivot(
        index="Department", columns="Attrition", values="count"
    ).fillna(0)
    dept_pivot["Total"] = dept_pivot["No"] + dept_pivot["Yes"]
    dept_pivot["Attrition Rate %"] = (
        dept_pivot["Yes"] / dept_pivot["Total"] * 100
    ).round(1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    dept_pivot[["No", "Yes"]].plot(kind="bar", stacked=True,
                                    color=["steelblue", "tomato"],
                                    edgecolor="black", ax=axes[0])
    axes[0].set_title("Attrition Count by Department", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Department")
    axes[0].set_ylabel("Number of Employees")
    axes[0].tick_params(axis='x', rotation=15)
    axes[0].legend(["Stayed", "Left"])

    axes[1].bar(dept_pivot.index, dept_pivot["Attrition Rate %"],
                color="tomato", edgecolor="black")
    axes[1].set_title("Attrition Rate by Department (%)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Department")
    axes[1].set_ylabel("Attrition Rate (%)")
    axes[1].tick_params(axis='x', rotation=15)
    for i, v in enumerate(dept_pivot["Attrition Rate %"]):
        axes[1].text(i, v + 0.3, f"{v}%", ha="center", fontweight="bold")

    plt.suptitle("Department Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_PATH, "attrition_by_department.png"), dpi=150)
    plt.show()
    print(dept_pivot[["No", "Yes", "Total", "Attrition Rate %"]].to_string())


def plot_salary_vs_performance(df_original):
    """
    Plot monthly income vs performance rating.

    Args:
        df_original: Original Spark DataFrame
    """
    income_perf = df_original.groupBy("PerformanceRating").agg(
        F.round(F.avg("MonthlyIncome"), 2).alias("Avg Income"),
        F.round(F.min("MonthlyIncome"), 2).alias("Min Income"),
        F.round(F.max("MonthlyIncome"), 2).alias("Max Income"),
        F.count("*").alias("Employee Count")
    ).orderBy("PerformanceRating").toPandas()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].bar(income_perf["PerformanceRating"].astype(str),
                income_perf["Avg Income"],
                color=["steelblue", "tomato"], edgecolor="black")
    axes[0].set_title("Average Monthly Income by Performance Rating",
                       fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Performance Rating")
    axes[0].set_ylabel("Average Monthly Income ($)")
    for i, v in enumerate(income_perf["Avg Income"]):
        axes[0].text(i, v + 50, f"${v:,.0f}", ha="center", fontweight="bold")

    income_pdf = df_original.select("PerformanceRating", "MonthlyIncome").toPandas()
    income_pdf.boxplot(column="MonthlyIncome", by="PerformanceRating", ax=axes[1],
                       boxprops=dict(color="steelblue"),
                       medianprops=dict(color="tomato", linewidth=2))
    axes[1].set_title("Income Distribution by Performance Rating",
                       fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Performance Rating")
    axes[1].set_ylabel("Monthly Income ($)")

    plt.suptitle("Monthly Income vs Performance Rating", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_PATH, "income_vs_performance.png"), dpi=150)
    plt.show()
    print(income_perf.to_string(index=False))


def plot_overtime_vs_attrition(df_original):
    """
    Plot overtime impact on attrition.

    Args:
        df_original: Original Spark DataFrame
    """
    overtime_attrition = df_original.groupBy("OverTime", "Attrition") \
        .count().orderBy("OverTime").toPandas()

    overtime_pivot = overtime_attrition.pivot(
        index="OverTime", columns="Attrition", values="count"
    ).fillna(0)
    overtime_pivot["Total"] = overtime_pivot["No"] + overtime_pivot["Yes"]
    overtime_pivot["Attrition Rate %"] = (
        overtime_pivot["Yes"] / overtime_pivot["Total"] * 100
    ).round(1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    overtime_pivot[["No", "Yes"]].plot(kind="bar", stacked=True,
                                        color=["steelblue", "tomato"],
                                        edgecolor="black", ax=axes[0])
    axes[0].set_title("Attrition Count by Overtime", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Overtime")
    axes[0].set_ylabel("Number of Employees")
    axes[0].tick_params(axis='x', rotation=0)
    axes[0].legend(["Stayed", "Left"])

    axes[1].bar(overtime_pivot.index, overtime_pivot["Attrition Rate %"],
                color=["steelblue", "tomato"], edgecolor="black")
    axes[1].set_title("Attrition Rate by Overtime (%)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Overtime")
    axes[1].set_ylabel("Attrition Rate (%)")
    axes[1].tick_params(axis='x', rotation=0)
    for i, v in enumerate(overtime_pivot["Attrition Rate %"]):
        axes[1].text(i, v + 0.3, f"{v}%", ha="center", fontweight="bold")

    plt.suptitle("Overtime vs Attrition Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_PATH, "overtime_vs_attrition.png"), dpi=150)
    plt.show()
    print(overtime_pivot[["No", "Yes", "Total", "Attrition Rate %"]].to_string())


def plot_correlation_heatmap(df_preprocessed):
    """
    Plot correlation heatmap for all numerical features.

    Args:
        df_preprocessed: Preprocessed Spark DataFrame
    """
    numeric_cols = [
        "Age", "DailyRate", "DistanceFromHome", "Education",
        "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement",
        "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate",
        "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
        "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
        "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
        "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
        "Attrition_encoded"
    ]

    corr_pdf = df_preprocessed.select(numeric_cols).toPandas()
    corr_matrix = corr_pdf.corr()

    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                linewidths=0.5,
                annot_kws={"size": 7})
    plt.title("Correlation Heatmap - All Features", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_PATH, "correlation_heatmap.png"), dpi=150)
    plt.show()

    print("Top Correlations with Attrition:")
    print("-" * 40)
    attrition_corr = corr_matrix["Attrition_encoded"].drop("Attrition_encoded") \
        .sort_values(ascending=False)
    print(attrition_corr.to_string())


def plot_attrition_by_age(df_original):
    """
    Plot attrition by age.
    """

    age_attrition = df_original.groupBy("Age", "Attrition") \
        .count().orderBy("Age").toPandas()

    age_pivot = age_attrition.pivot(
        index="Age",
        columns="Attrition",
        values="count"
    ).fillna(0)

    age_pivot[["No", "Yes"]].plot(
        kind="bar",
        stacked=True,
        figsize=(14, 6),
        color=["steelblue", "tomato"],
        edgecolor="black"
    )

    plt.title("Attrition by Age", fontsize=16, fontweight="bold")
    plt.xlabel("Age")
    plt.ylabel("Number of Employees")
    plt.legend(["Stayed", "Left"])

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUTS_PATH, "attrition_by_age.png"),
        dpi=150
    )

    plt.show()

    print(age_pivot.to_string())



def run_eda(df_preprocessed, df_original):
    """
    Run the full EDA pipeline.

    Args:
        df_preprocessed: Preprocessed Spark DataFrame
        df_original: Original Spark DataFrame with categorical columns
    """
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    plot_attrition_distribution(df_original)
    plot_attrition_by_age(df_original)
    plot_attrition_by_department(df_original)
    plot_salary_vs_performance(df_original)
    plot_overtime_vs_attrition(df_original)
    plot_correlation_heatmap(df_preprocessed)
    print("=" * 60)
    print("EDA COMPLETED SUCCESSFULLY")
    print("=" * 60)