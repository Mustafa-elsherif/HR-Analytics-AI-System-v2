# =============================================================
# HR Analytics AI System
# Module 6: Employee Segmentation using K-Means Clustering
# Responsible: Mustafa Nabil - 240103415
#              Mahmoud Hesham - 240101375
# =============================================================

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import functions as F
import matplotlib.pyplot as plt
import os

plt.style.use('seaborn-v0_8')
OUTPUTS_PATH = "outputs"

CLUSTER_FEATURES = [
    "Age", "MonthlyIncome", "JobLevel",
    "TotalWorkingYears", "YearsAtCompany", "RiskScore"
]

CLUSTER_LABELS = {
    0: "Mid-Career Stable",
    1: "Young High-Risk",
    2: "Senior High-Income"
}


def prepare_clustering_features(df):
    """
    Assemble and scale features for clustering.

    Args:
        df: Spark DataFrame

    Returns:
        Scaled Spark DataFrame
    """
    assembler = VectorAssembler(
        inputCols=CLUSTER_FEATURES,
        outputCol="features_raw"
    )
    df_assembled = assembler.transform(df)

    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features_scaled",
        withStd=True,
        withMean=True
    )
    df_scaled = scaler.fit(df_assembled).transform(df_assembled)
    return df_scaled


def find_optimal_clusters(df_scaled):
    """
    Use the Elbow Method to find optimal number of clusters.

    Args:
        df_scaled: Scaled Spark DataFrame

    Returns:
        None
    """
    wssse_list = []
    k_values = range(2, 11)

    for k in k_values:
        kmeans = KMeans(
            featuresCol="features_scaled",
            predictionCol="cluster",
            k=k,
            seed=42
        )
        model = kmeans.fit(df_scaled)
        wssse = model.summary.trainingCost
        wssse_list.append(wssse)
        print(f"K = {k} -> WSSSE = {wssse:.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, wssse_list, marker='o', color="steelblue", linewidth=2)
    plt.title("Elbow Method - Optimal Number of Clusters",
              fontsize=14, fontweight="bold")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WSSSE")
    plt.xticks(k_values)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_PATH, "elbow_method.png"), dpi=150)
    plt.show()


def train_kmeans(df_scaled, k=3):
    """
    Train K-Means clustering model.

    Args:
        df_scaled: Scaled Spark DataFrame
        k: Number of clusters

    Returns:
        tuple: (model, clustered DataFrame)
    """
    kmeans = KMeans(
        featuresCol="features_scaled",
        predictionCol="cluster",
        k=k,
        seed=42
    )
    model = kmeans.fit(df_scaled)
    df_clustered = model.transform(df_scaled)
    return model, df_clustered


def evaluate_clustering(df_clustered):
    """
    Evaluate clustering using Silhouette Score.

    Args:
        df_clustered: Clustered Spark DataFrame

    Returns:
        float: Silhouette Score
    """
    evaluator = ClusteringEvaluator(
        featuresCol="features_scaled",
        predictionCol="cluster",
        metricName="silhouette"
    )
    silhouette = evaluator.evaluate(df_clustered)

    print("=" * 60)
    print("CLUSTERING EVALUATION")
    print("=" * 60)
    print(f"Silhouette Score: {silhouette:.4f}")
    print("=" * 60)

    return silhouette


def analyze_clusters(df_clustered):
    """
    Analyze and display cluster characteristics.

    Args:
        df_clustered: Clustered Spark DataFrame
    """
    print("=" * 60)
    print("CLUSTER ANALYSIS SUMMARY")
    print("=" * 60)

    cluster_analysis = df_clustered.groupBy("cluster").agg(
        F.round(F.avg("Age"), 1).alias("Avg Age"),
        F.round(F.avg("MonthlyIncome"), 0).alias("Avg Income"),
        F.round(F.avg("JobLevel"), 1).alias("Avg JobLevel"),
        F.round(F.avg("TotalWorkingYears"), 1).alias("Avg Experience"),
        F.round(F.avg("YearsAtCompany"), 1).alias("Avg Years at Company"),
        F.round(F.avg("RiskScore"), 3).alias("Avg Risk Score"),
        F.count("*").alias("Total Employees")
    ).orderBy("cluster")

    cluster_analysis.show(truncate=False)


def plot_cluster_visualization(df_clustered):
    """
    Plot cluster visualization scatter plots.

    Args:
        df_clustered: Clustered Spark DataFrame
    """
    cluster_pdf = df_clustered.select(
        "cluster", "Age", "MonthlyIncome",
        "TotalWorkingYears", "RiskScore"
    ).toPandas()

    colors = {0: "steelblue", 1: "tomato", 2: "seagreen"}
    labels = {0: "Mid-Career Stable", 1: "Young High-Risk", 2: "Senior High-Income"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for cluster in [0, 1, 2]:
        subset = cluster_pdf[cluster_pdf["cluster"] == cluster]
        axes[0].scatter(subset["Age"], subset["MonthlyIncome"],
                        c=colors[cluster], label=labels[cluster], alpha=0.6)
    axes[0].set_title("Age vs Monthly Income", fontweight="bold")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Monthly Income ($)")
    axes[0].legend()

    for cluster in [0, 1, 2]:
        subset = cluster_pdf[cluster_pdf["cluster"] == cluster]
        axes[1].scatter(subset["TotalWorkingYears"], subset["MonthlyIncome"],
                        c=colors[cluster], label=labels[cluster], alpha=0.6)
    axes[1].set_title("Experience vs Monthly Income", fontweight="bold")
    axes[1].set_xlabel("Total Working Years")
    axes[1].set_ylabel("Monthly Income ($)")
    axes[1].legend()

    cluster_counts = cluster_pdf["cluster"].value_counts().sort_index()
    axes[2].bar([labels[i] for i in cluster_counts.index],
                cluster_counts.values,
                color=[colors[i] for i in cluster_counts.index],
                edgecolor="black")
    axes[2].set_title("Cluster Distribution", fontweight="bold")
    axes[2].set_xlabel("Cluster")
    axes[2].set_ylabel("Number of Employees")
    axes[2].tick_params(axis='x', rotation=15)
    for i, v in enumerate(cluster_counts.values):
        axes[2].text(i, v + 5, str(v), ha="center", fontweight="bold")

    plt.suptitle("Employee Segmentation - Cluster Visualization",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_PATH, "cluster_visualization.png"), dpi=150)
    plt.show()


def run_clustering(df):
    """
    Run the full clustering pipeline.

    Args:
        df: Feature engineered Spark DataFrame

    Returns:
        tuple: (model, clustered DataFrame, silhouette score)
    """
    print("=" * 60)
    print("EMPLOYEE SEGMENTATION - K-MEANS CLUSTERING")
    print("=" * 60)

    df_scaled = prepare_clustering_features(df)

    print("\nFinding Optimal Number of Clusters (Elbow Method):")
    find_optimal_clusters(df_scaled)

    print("\nTraining Final K-Means Model with K=3:")
    model, df_clustered = train_kmeans(df_scaled, k=3)

    silhouette = evaluate_clustering(df_clustered)

    print("\nCluster Distribution:")
    df_clustered.groupBy("cluster").count().orderBy("cluster").show()

    analyze_clusters(df_clustered)
    plot_cluster_visualization(df_clustered)

    print("=" * 60)
    print("CLUSTERING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return model, df_clustered, silhouette