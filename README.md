# HR Analytics AI System

## An AI-Powered System for Employee Analytics and Decision Support

---

## Live Dashboard

Access the interactive dashboard here:  
https://hr-analytics-ai-system.streamlit.app/

---

## Project Overview

This project develops an end-to-end AI-powered HR analytics system using Apache Spark and Machine Learning to analyze employee data, predict attrition, evaluate performance, segment employees, and recommend targeted HR strategies.

The system processes the IBM HR Analytics Dataset containing 1,470 employees and 35 features, delivering actionable insights to support data-driven HR decision-making.

---

## Team Members

| Name | Student ID | Responsibilities |
|------|------------|------------------|
| Mustafa Nabil | 240103415 | Model Development, Clustering, Dashboard |
| Mahmoud Hesham | 240101375 | Model Development, Clustering, Dashboard |
| Mariam Magdy | 240102374 | Exploratory Data Analysis (EDA) |
| Omar Ahmed Wafik | 240101244 | Data Preprocessing, Feature Engineering |
| yousef hany | 240102846 | Data Ingestion, Insights and HR Recommendations |

---

## Project Structure

```bash
HR-Analytics-AI-System-v2/
│
├── hr_analytics/
│   ├── __init__.py
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── clustering.py
│   └── insights.py
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── src/
│   └── dashboard.py
│
├── models/
│   ├── attrition_model/
│   └── performance_model/
│
├── outputs/
│   ├── attrition_distribution.png
│   ├── attrition_by_department.png
│   ├── income_vs_performance.png
│   ├── overtime_vs_attrition.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── elbow_method.png
│   ├── cluster_visualization.png
│   └── high_risk_analysis.png
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Dataset

| Property | Details |
|----------|---------|
| Source | IBM HR Analytics Employee Attrition Dataset (Kaggle) |
| Records | 1,470 employees |
| Features | 35 columns |
| Target Variable | Attrition (Yes / No) |

---

## Technologies Used

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.13 | Main programming language |
| Apache Spark | 3.5.8 | Distributed data processing |
| PySpark | 3.5.8 | Spark Python API |
| Pandas | Latest | Data manipulation |
| NumPy | Latest | Numerical computing |
| Matplotlib | Latest | Static visualizations |
| Seaborn | Latest | Statistical visualizations |
| Plotly | Latest | Interactive visualizations |
| Streamlit | Latest | Interactive dashboard |

---

## Project Pipeline

### 1. Data Ingestion
- Loaded the IBM HR Analytics Dataset using Apache Spark
- Dataset contains 1,470 rows and 35 columns
- Saved data in Parquet format for optimized performance

### 2. Data Preprocessing
- No missing values found
- No duplicate records found
- Dropped 4 irrelevant columns:
  - EmployeeCount
  - StandardHours
  - Over18
  - EmployeeNumber
- Encoded 8 categorical columns using StringIndexer
- Final preprocessed dataset: 1,470 rows and 31 columns

### 3. Exploratory Data Analysis
- Attrition Distribution: 83.9% stayed, 16.1% left
- Department Analysis: Sales has highest attrition at 20.6%
- Age Analysis: Employees under 25 have highest attrition at 39.2%
- Overtime Analysis: Overtime employees have 30.5% attrition vs 10.4%
- Income Analysis: Low-income employees have 28.6% attrition rate
- Correlation Analysis: TotalWorkingYears and JobLevel are strongest negative predictors

### 4. Feature Engineering

Five new features were created:

- **EngagementScore:** Average of JobInvolvement, JobSatisfaction, RelationshipSatisfaction
- **SatisfactionScore:** Average of EnvironmentSatisfaction, JobSatisfaction, RelationshipSatisfaction, WorkLifeBalance
- **RiskScore:** Weighted combination of overtime, engagement, satisfaction, tenure, and income
- **ExperienceLevel:** Categorical grouping based on TotalWorkingYears
- **IncomeLevel:** Categorical grouping based on MonthlyIncome

### 5. Model Development

#### Attrition Prediction (Classification)
- Algorithm: Random Forest Classifier
- Training Set: 1,216 employees (80%)
- Testing Set: 254 employees (20%)
- Class imbalance handled using class weights

| Metric | Score |
|--------|-------|
| Accuracy | 88.98% |
| F1-Score | 0.87 |
| Precision | 0.88 |
| Recall | 0.89 |

#### Performance Prediction (Regression)
- Algorithm: Random Forest Regressor
- Target Variable: PerformanceRating

| Metric | Score |
|--------|-------|
| RMSE | 0.0208 |
| MAE | 0.0132 |
| R2 Score | 0.9964 |

### 6. Employee Segmentation (K-Means Clustering)
- Optimal clusters determined using Elbow Method
- Final model: K = 3 clusters
- Evaluation: Silhouette Score = 0.4052

| Cluster | Name | Employees | Avg Age | Avg Income | Avg Risk Score |
|---------|------|-----------|---------|------------|----------------|
| 0 | Mid-Career Stable | 628 | 38.8 | $6,436 | 0.313 |
| 1 | Young High-Risk | 624 | 31.2 | $3,405 | 0.466 |
| 2 | Senior High-Income | 218 | 47.9 | $15,562 | 0.289 |

### 7. Insights and HR Recommendations

#### High-Risk Employees
- 194 employees (13.2%) identified with RiskScore above 0.6
- 87.1% of high-risk employees belong to the Young High-Risk cluster

#### Strategic HR Recommendations
1. **Overtime Policy Reform:** Cap overtime to 10 hours per week
2. **Compensation Review:** Salary benchmarking for employees below $3,000
3. **Young Employee Retention:** Career development programs for under-35 employees
4. **Engagement Improvement:** Quarterly satisfaction surveys and recognition programs
5. **High-Risk Monitoring:** Monthly check-ins with 194 identified high-risk employees

---

## Model Performance Summary

| Model | Algorithm | Key Metric | Score |
|-------|-----------|------------|-------|
| Attrition Prediction | Random Forest Classifier | F1-Score | 0.87 |
| Performance Prediction | Random Forest Regressor | R2 Score | 0.9964 |
| Employee Segmentation | K-Means Clustering | Silhouette Score | 0.4052 |

---

## Cloud Platform

This project was developed and executed on **Databricks** using Apache Spark for distributed data processing and machine learning at scale.

The GitHub repository is connected to Databricks via Git Folder integration.

---

## How to Run

### Prerequisites
- Python 3.13
- Java JDK 11
- Apache Spark 3.5.8

### Installation

```bash
git clone https://github.com/Mustafa-elsherif/HR-Analytics-AI-System-v2.git
cd HR-Analytics-AI-System-v2
pip install -r requirements.txt
```

### Run Full Pipeline

```bash
python main.py
```

### Run Dashboard

```bash
streamlit run src/dashboard.py
```

---

## License

This project is developed for educational purposes as part of a university course requirement.