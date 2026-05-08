import streamlit as st
import pandas as pd
import plotly.express as px
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="HR Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { color: #ffffff; font-size: 2rem; font-weight: 700;
         border-bottom: 3px solid #4a9eda; padding-bottom: 10px; }
    h2, h3 { color: #4a9eda; font-weight: 600; }
    [data-testid="stMetricValue"] { color: #4a9eda !important; font-size: 1.8rem !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] { color: #ffffff !important; font-size: 0.95rem !important; font-weight: 500 !important; }
    [data-testid="stMetric"] { background-color: #1e2530 !important; padding: 15px;
         border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.3);
         border-left: 5px solid #4a9eda; }
    .insight-box { background-color: #1e2530; padding: 15px 20px; border-radius: 8px;
         border-left: 4px solid #4a9eda; margin: 10px 0;
         color: #ffffff; font-size: 0.95rem; }
    .warning-box { background-color: #2a1a1a; padding: 15px 20px; border-radius: 8px;
         border-left: 4px solid #c0392b; margin: 10px 0;
         color: #ffffff; font-size: 0.95rem; }
    .section-divider { border: none; border-top: 2px solid #2d3748; margin: 20px 0; }
    </style>
""", unsafe_allow_html=True)

DARK = "#0e1117"
CARD = "#1e2530"
BLUE = "#4a9eda"
RED  = "#c0392b"
GREEN = "#16a085"
WHITE = "#ffffff"

@st.cache_resource
def init_spark():
    from pyspark.sql import SparkSession
    spark = SparkSession.builder \
        .appName("HR Analytics Dashboard") \
        .master("local[*]") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

@st.cache_data
def load_data():
    spark = init_spark()
    
    
    df_spark = spark.read.parquet("data/hr_data_engineered.parquet")
    df_original = spark.read.parquet("data/hr_data.parquet")
    
    cluster_features = ["Age", "MonthlyIncome", "JobLevel", 
                        "TotalWorkingYears", "YearsAtCompany", "RiskScore"]
    
   
    from pyspark.ml.feature import VectorAssembler, StandardScaler
    from pyspark.ml.clustering import KMeans

    assembler = VectorAssembler(inputCols=cluster_features, outputCol="features_raw")
    df_assembled = assembler.transform(df_spark)
    
    scaler = StandardScaler(inputCol="features_raw", outputCol="features_scaled", 
                            withStd=True, withMean=True)
    df_scaled = scaler.fit(df_assembled).transform(df_assembled)
    
    kmeans = KMeans(featuresCol="features_scaled", predictionCol="cluster", k=3, seed=42)
    df_clustered = kmeans.fit(df_scaled).transform(df_scaled)
    
    df = df_clustered.toPandas()
    df_orig = df_original.toPandas()
    
    df["cluster_name"] = df["cluster"].map({
        0: "Mid-Career Stable",
        1: "Young High-Risk",
        2: "Senior High-Income"
    })
    
    return df, df_orig

def chart_layout(fig, height=380):
    fig.update_layout(
        height=height,
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(color=WHITE, size=13),
        margin=dict(t=30, b=10, l=10, r=10)
    )
    fig.update_xaxes(gridcolor="#2d3748", zerolinecolor="#2d3748")
    fig.update_yaxes(gridcolor="#2d3748", zerolinecolor="#2d3748")
    return fig

with st.spinner("Loading HR Analytics data..."):
    df, df_orig = load_data()

# Sidebar
st.sidebar.title("HR Analytics System")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "Overview",
    "Attrition Analysis",
    "Employee Segmentation",
    "High-Risk Employees",
    "HR Recommendations"
])
st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Summary")
st.sidebar.markdown(f"- **Total Employees:** {len(df):,}")
st.sidebar.markdown(f"- **Attrition Rate:** {df['Attrition_encoded'].mean()*100:.1f}%")
st.sidebar.markdown(f"- **High-Risk:** {len(df[df['RiskScore'] >= 0.6]):,}")
st.sidebar.markdown(f"- **Avg Risk Score:** {df['RiskScore'].mean():.3f}")
st.sidebar.markdown("---")
st.sidebar.markdown("### Model Performance")
st.sidebar.markdown("- Accuracy: 88.98%")
st.sidebar.markdown("- F1-Score: 0.87")
st.sidebar.markdown("- R2 Score: 0.9964")
st.sidebar.markdown("- Silhouette: 0.4052")

# ============================================================
# PAGE 1: OVERVIEW
# ============================================================
if page == "Overview":
    st.title("HR Analytics - Executive Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Employees", f"{len(df):,}")
    with col2:
        st.metric("Employees Left", f"{int(df['Attrition_encoded'].sum()):,}")
    with col3:
        st.metric("Attrition Rate", f"{df['Attrition_encoded'].mean()*100:.1f}%")
    with col4:
        st.metric("High-Risk Employees", f"{len(df[df['RiskScore'] >= 0.6]):,}")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Attrition Distribution")
        attrition_counts = df_orig["Attrition"].value_counts().reset_index()
        attrition_counts.columns = ["Attrition", "Count"]
        fig = px.pie(attrition_counts, values="Count", names="Attrition",
                     color_discrete_sequence=[BLUE, RED], hole=0.4)
        st.plotly_chart(chart_layout(fig), use_container_width=True)
        st.markdown("""<div class='insight-box'>
        Out of 1,470 employees, <b>1,233 (83.9%)</b> stayed with the company
        while <b>237 (16.1%)</b> left. This class imbalance was addressed during
        model training using class weighting techniques.
        </div>""", unsafe_allow_html=True)

    with col2:
        st.subheader("Attrition Rate by Department")
        dept = df_orig.groupby("Department")["Attrition"].apply(
            lambda x: round((x == "Yes").mean() * 100, 1)
        ).reset_index()
        dept.columns = ["Department", "Attrition Rate %"]
        fig = px.bar(dept, x="Department", y="Attrition Rate %",
                     color="Attrition Rate %",
                     color_continuous_scale=[BLUE, RED],
                     text="Attrition Rate %")
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        st.plotly_chart(chart_layout(fig), use_container_width=True)
        st.markdown("""<div class='insight-box'>
        <b>Sales</b> has the highest attrition rate at <b>20.6%</b>, followed by
        <b>Human Resources</b> at <b>19.0%</b>. Research and Development has the
        lowest rate at <b>13.8%</b>.
        </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Performance Summary")
        metrics_df = pd.DataFrame({
            "Model": ["Attrition\nClassification", "Performance\nRegression", "K-Means\nClustering"],
            "Score": [0.87, 0.9964, 0.4052],
            "Metric": ["F1-Score", "R2 Score", "Silhouette"]
        })
        fig = px.bar(metrics_df, x="Model", y="Score",
                     color="Model",
                     color_discrete_sequence=[BLUE, GREEN, "#8e44ad"],
                     text="Score")
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig.update_layout(yaxis_range=[0, 1.15], showlegend=False)
        st.plotly_chart(chart_layout(fig), use_container_width=True)
        st.markdown("""<div class='insight-box'>
        Three ML models were developed. The attrition model achieved
        <b>88.98% accuracy</b> with F1-Score of <b>0.87</b>. The performance
        regression model achieved a near-perfect R2 of <b>0.9964</b>.
        </div>""", unsafe_allow_html=True)

    with col2:
        st.subheader("Risk Score Distribution")
        fig = px.histogram(df, x="RiskScore", nbins=30,
                           color_discrete_sequence=[BLUE])
        fig.add_vline(x=0.6, line_dash="dash", line_color=RED, line_width=2,
                      annotation_text="High-Risk Threshold (0.6)",
                      annotation_font_color=RED)
        fig.update_layout(xaxis_title="Risk Score", yaxis_title="Number of Employees")
        st.plotly_chart(chart_layout(fig), use_container_width=True)
        st.markdown("""<div class='warning-box'>
        <b>194 employees (13.2%)</b> have a Risk Score above 0.6 and are classified
        as high-risk. These employees require immediate HR intervention.
        The average risk score is <b>0.374</b>.
        </div>""", unsafe_allow_html=True)

# ============================================================
# PAGE 2: ATTRITION ANALYSIS
# ============================================================
elif page == "Attrition Analysis":
    st.title("Attrition Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Attrition Rate by Age Group")
        df_orig["AgeGroup"] = pd.cut(df_orig["Age"],
                                      bins=[0, 25, 35, 45, 100],
                                      labels=["Under 25", "25-34", "35-44", "45+"])
        age_attr = df_orig.groupby("AgeGroup", observed=True)["Attrition"].apply(
            lambda x: round((x == "Yes").mean() * 100, 1)
        ).reset_index()
        age_attr.columns = ["Age Group", "Attrition Rate %"]
        fig = px.bar(age_attr, x="Age Group", y="Attrition Rate %",
                     color="Attrition Rate %",
                     color_continuous_scale=[BLUE, RED],
                     text="Attrition Rate %")
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(yaxis_range=[0, 50])
        st.plotly_chart(chart_layout(fig), use_container_width=True)
        st.markdown("""<div class='warning-box'>
        Employees <b>under 25</b> have the highest attrition rate at <b>39.2%</b>,
        nearly double the 25-34 group at <b>20.2%</b>. Attrition drops significantly
        for employees aged 35 and above, reaching only <b>10.1%</b> for the 35-44 group.
        </div>""", unsafe_allow_html=True)

    with col2:
        st.subheader("Attrition Rate by Overtime")
        ot_attr = df_orig.groupby("OverTime")["Attrition"].apply(
            lambda x: round((x == "Yes").mean() * 100, 1)
        ).reset_index()
        ot_attr.columns = ["Overtime", "Attrition Rate %"]
        fig = px.bar(ot_attr, x="Overtime", y="Attrition Rate %",
                     color="Overtime",
                     color_discrete_sequence=[BLUE, RED],
                     text="Attrition Rate %")
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(yaxis_range=[0, 40], showlegend=False)
        st.plotly_chart(chart_layout(fig), use_container_width=True)
        st.markdown("""<div class='warning-box'>
        Employees working overtime have an attrition rate of <b>30.5%</b>,
        nearly <b>3 times higher</b> than non-overtime employees at <b>10.4%</b>.
        Overtime is the strongest single predictor of employee attrition.
        </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Monthly Income Distribution by Attrition")
        fig = px.box(df_orig, x="Attrition", y="MonthlyIncome",
                     color="Attrition",
                     color_discrete_sequence=[BLUE, RED])
        fig.update_layout(xaxis_title="Attrition Status",
                          yaxis_title="Monthly Income ($)", showlegend=False)
        st.plotly_chart(chart_layout(fig), use_container_width=True)
        st.markdown("""<div class='insight-box'>
        Employees who left had a significantly lower median income than those who stayed.
        Low-income employees (below $3,000) have an attrition rate of <b>28.6%</b>,
        compared to only <b>10.8%</b> for high-income employees.
        </div>""", unsafe_allow_html=True)

    with col2:
        st.subheader("Attrition Rate by Gender")
        gender_attr = df_orig.groupby("Gender")["Attrition"].apply(
            lambda x: round((x == "Yes").mean() * 100, 1)
        ).reset_index()
        gender_attr.columns = ["Gender", "Attrition Rate %"]
        fig = px.bar(gender_attr, x="Gender", y="Attrition Rate %",
                     color="Gender",
                     color_discrete_sequence=[BLUE, RED],
                     text="Attrition Rate %")
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(yaxis_range=[0, 25], showlegend=False)
        st.plotly_chart(chart_layout(fig), use_container_width=True)
        st.markdown("""<div class='insight-box'>
        Male employees have a slightly higher attrition rate at <b>17.0%</b>
        compared to female employees at <b>14.8%</b>. Gender-specific
        retention strategies may be beneficial.
        </div>""", unsafe_allow_html=True)

# ============================================================
# PAGE 3: EMPLOYEE SEGMENTATION
# ============================================================
elif page == "Employee Segmentation":
    st.title("Employee Segmentation - K-Means Clustering")

    col1, col2, col3 = st.columns(3)
    for i, (cluster, name, color) in enumerate([
        (0, "Mid-Career Stable", BLUE),
        (1, "Young High-Risk", RED),
        (2, "Senior High-Income", GREEN)
    ]):
        subset = df[df["cluster"] == cluster]
        with [col1, col2, col3][i]:
            st.metric(name,
                      f"{len(subset):,} employees",
                      f"Avg Risk: {subset['RiskScore'].mean():.3f}")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Age vs Monthly Income by Cluster")
        fig = px.scatter(df, x="Age", y="MonthlyIncome",
                         color="cluster_name",
                         color_discrete_sequence=[BLUE, RED, GREEN],
                         opacity=0.7)
        fig.update_layout(xaxis_title="Age", yaxis_title="Monthly Income ($)")
        st.plotly_chart(chart_layout(fig, height=400), use_container_width=True)
        st.markdown("""<div class='insight-box'>
        The scatter plot shows clear separation between clusters.
        <b>Senior High-Income</b> employees are concentrated at higher ages and incomes,
        while <b>Young High-Risk</b> employees cluster at lower income levels.
        </div>""", unsafe_allow_html=True)

    with col2:
        st.subheader("Experience vs Monthly Income by Cluster")
        fig = px.scatter(df, x="TotalWorkingYears", y="MonthlyIncome",
                         color="cluster_name",
                         color_discrete_sequence=[BLUE, RED, GREEN],
                         opacity=0.7)
        fig.update_layout(xaxis_title="Total Working Years",
                          yaxis_title="Monthly Income ($)")
        st.plotly_chart(chart_layout(fig, height=400), use_container_width=True)
        st.markdown("""<div class='insight-box'>
        A strong positive correlation exists between experience and income.
        Senior employees with 20+ years earn significantly more, while
        Young High-Risk employees with limited experience earn the least.
        </div>""", unsafe_allow_html=True)

    st.subheader("Cluster Summary Table")
    cluster_summary = df.groupby("cluster_name").agg(
        Total=("Age", "count"),
        Avg_Age=("Age", "mean"),
        Avg_Income=("MonthlyIncome", "mean"),
        Avg_RiskScore=("RiskScore", "mean"),
        Avg_Experience=("TotalWorkingYears", "mean"),
        Avg_Engagement=("EngagementScore", "mean"),
        Overtime_Rate=("OverTime_encoded", "mean")
    ).round(2).reset_index()
    cluster_summary.columns = ["Cluster", "Total", "Avg Age",
                                "Avg Income ($)", "Avg Risk Score",
                                "Avg Experience (Yrs)", "Avg Engagement",
                                "Overtime Rate"]
    st.dataframe(cluster_summary, use_container_width=True)
    st.markdown("""<div class='insight-box'>
    The clustering model achieved a <b>Silhouette Score of 0.4052</b>.
    The three clusters represent distinct employee profiles with meaningful
    differences in age, income, experience, and risk levels.
    </div>""", unsafe_allow_html=True)

# ============================================================
# PAGE 4: HIGH-RISK EMPLOYEES
# ============================================================
elif page == "High-Risk Employees":
    st.title("High-Risk Employee Identification")

    high_risk_df = df[df["RiskScore"] >= 0.6].sort_values("RiskScore", ascending=False)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("High-Risk Employees", f"{len(high_risk_df):,}")
    with col2:
        st.metric("% of Workforce", f"{len(high_risk_df)/len(df)*100:.1f}%")
    with col3:
        st.metric("Avg Risk Score", f"{high_risk_df['RiskScore'].mean():.3f}")
    with col4:
        st.metric("Avg Income", f"${high_risk_df['MonthlyIncome'].mean():,.0f}")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("High-Risk Employees by Cluster")
        hr_cluster = high_risk_df.groupby("cluster_name").size().reset_index()
        hr_cluster.columns = ["Cluster", "Count"]
        fig = px.bar(hr_cluster, x="Cluster", y="Count",
                     color="Cluster",
                     color_discrete_sequence=[BLUE, RED, GREEN],
                     text="Count")
        fig.update_traces(texttemplate="%{text}", textposition="outside")
        fig.update_layout(showlegend=False, yaxis_range=[0, 200])
        st.plotly_chart(chart_layout(fig), use_container_width=True)
        st.markdown("""<div class='warning-box'>
        <b>169 out of 194 high-risk employees (87.1%)</b> belong to the
        Young High-Risk cluster. This group requires the most urgent HR attention.
        </div>""", unsafe_allow_html=True)

    with col2:
        st.subheader("Age vs Risk Score")
        fig = px.scatter(df, x="Age", y="RiskScore",
                         color=df["RiskScore"].apply(
                             lambda x: "High Risk" if x >= 0.6 else "Normal Risk"),
                         color_discrete_map={"High Risk": RED, "Normal Risk": BLUE},
                         opacity=0.7)
        fig.add_hline(y=0.6, line_dash="dash", line_color=WHITE, line_width=2,
                      annotation_text="High-Risk Threshold (0.6)",
                      annotation_font_color=WHITE)
        fig.update_layout(xaxis_title="Age", yaxis_title="Risk Score")
        st.plotly_chart(chart_layout(fig), use_container_width=True)
        st.markdown("""<div class='warning-box'>
        High-risk employees (red) appear across all age groups but are more
        concentrated among younger employees. All employees above the threshold
        line require immediate HR attention.
        </div>""", unsafe_allow_html=True)

    st.subheader("Top 50 Highest-Risk Employees")
    display_cols = ["Age", "MonthlyIncome", "JobLevel", "TotalWorkingYears",
                    "YearsAtCompany", "EngagementScore", "SatisfactionScore",
                    "RiskScore", "cluster_name"]
    styled_df = high_risk_df[display_cols].head(50).round(3).reset_index(drop=True)
    styled_df.columns = ["Age", "Monthly Income ($)", "Job Level",
                         "Total Working Years", "Years at Company",
                         "Engagement Score", "Satisfaction Score",
                         "Risk Score", "Cluster"]
    st.dataframe(styled_df, use_container_width=True)
    st.markdown("""<div class='insight-box'>
    The table above displays the 50 highest-risk employees ranked by Risk Score.
    HR teams should prioritize direct engagement with these employees through
    personalized retention packages and immediate workload review.
    </div>""", unsafe_allow_html=True)

# ============================================================
# PAGE 5: HR RECOMMENDATIONS
# ============================================================
elif page == "HR Recommendations":
    st.title("HR Strategic Recommendations")

    st.subheader("Key Attrition Risk Factors")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Overtime Impact on Attrition**")
        ot_data = pd.DataFrame({
            "Group": ["No Overtime", "Overtime"],
            "Attrition Rate (%)": [10.4, 30.5]
        })
        fig = px.bar(ot_data, x="Group", y="Attrition Rate (%)",
                     color="Group",
                     color_discrete_sequence=[BLUE, RED],
                     text="Attrition Rate (%)")
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(showlegend=False, yaxis_range=[0, 40])
        st.plotly_chart(chart_layout(fig, height=300), use_container_width=True)
        st.markdown("""<div class='warning-box'>
        Overtime increases attrition risk by <b>3x</b>. Employees working overtime
        have a 30.5% rate vs 10.4% for those who do not.
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("**Income Level Impact on Attrition**")
        inc_data = pd.DataFrame({
            "Income Group": ["Low (below $3K)", "Medium ($3K-$7K)", "High (above $7K)"],
            "Attrition Rate (%)": [28.6, 12.0, 10.8]
        })
        fig = px.bar(inc_data, x="Income Group", y="Attrition Rate (%)",
                     color="Income Group",
                     color_discrete_sequence=[RED, "#e67e22", BLUE],
                     text="Attrition Rate (%)")
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(showlegend=False, yaxis_range=[0, 40])
        st.plotly_chart(chart_layout(fig, height=300), use_container_width=True)
        st.markdown("""<div class='warning-box'>
        Low-income employees have a <b>28.6%</b> attrition rate, nearly
        <b>3x higher</b> than high-income employees at 10.8%.
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown("**Age Group Impact on Attrition**")
        age_data = pd.DataFrame({
            "Age Group": ["Under 25", "25-34", "35-44", "45+"],
            "Attrition Rate (%)": [39.2, 20.2, 10.1, 11.5]
        })
        fig = px.bar(age_data, x="Age Group", y="Attrition Rate (%)",
                     color="Age Group",
                     color_discrete_sequence=[RED, "#e67e22", BLUE, GREEN],
                     text="Attrition Rate (%)")
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(showlegend=False, yaxis_range=[0, 50])
        st.plotly_chart(chart_layout(fig, height=300), use_container_width=True)
        st.markdown("""<div class='warning-box'>
        Employees under 25 have the highest attrition rate at <b>39.2%</b>,
        dropping to 10.1% for the 35-44 age group.
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.subheader("Strategic HR Action Plan")

    recommendations = {
        "1. Overtime Policy Reform": {
            "risk": "Attrition Rate: 30.5% for overtime employees",
            "actions": [
                "Cap overtime hours to a maximum of 10 hours per week",
                "Implement overtime compensation and time-off policies",
                "Monitor overtime patterns and flag employees exceeding limits",
                "Redistribute workload to reduce dependency on overtime"
            ]
        },
        "2. Compensation Review": {
            "risk": "Attrition Rate: 28.6% for low-income employees",
            "actions": [
                "Conduct immediate salary benchmarking for low-income employees",
                "Implement performance-based salary increments",
                "Review compensation for employees earning below $3,000 per month",
                "Introduce stock options and benefits for entry-level employees"
            ]
        },
        "3. Young Employee Retention": {
            "risk": "Attrition Rate: 39.2% for employees under 25",
            "actions": [
                "Create structured career development programs for under-35 employees",
                "Implement mentorship programs pairing juniors with senior employees",
                "Offer education assistance and professional certification support",
                "Establish clear promotion timelines and criteria"
            ]
        },
        "4. Engagement Improvement": {
            "risk": "Average Engagement Score: 2.72 out of 4.0",
            "actions": [
                "Launch quarterly employee satisfaction surveys",
                "Create cross-functional project opportunities",
                "Implement recognition and rewards programs",
                "Establish open-door communication policies with management"
            ]
        },
        "5. High-Risk Employee Monitoring": {
            "risk": "194 employees identified with Risk Score above 0.6",
            "actions": [
                "Immediately flag and monitor all 194 high-risk employees",
                "Schedule monthly check-ins with high-risk employees",
                "Offer personalized retention packages for critical cases",
                "Track risk score changes monthly and update intervention plans"
            ]
        }
    }

    for title, content in recommendations.items():
        with st.expander(f"{title}  —  {content['risk']}"):
            for i, action in enumerate(content["actions"], 1):
                st.markdown(f"**{i}.** {action}")