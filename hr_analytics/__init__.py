# HR Analytics AI System
# Modular Package

from hr_analytics.data_ingestion import load_data
from hr_analytics.preprocessing import preprocess_data
from hr_analytics.eda import run_eda
from hr_analytics.feature_engineering import engineer_features
from hr_analytics.models import train_models
from hr_analytics.clustering import run_clustering
from hr_analytics.insights import generate_insights

__version__ = "1.0.0"
__author__ = "Mustafa Nabil, Mahmoud Hesham, yousef hany , Omar Ahmed Wafik, Mohamed Nour"