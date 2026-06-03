# Healthcare Risk Modeling & Telehealth Strategy Analysis
### Predicting County-Level Health Outcomes Across the United States

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## Project Overview

This project analyzes county-level health data across **3,143 U.S. counties** to model health outcomes and identify geographic segments with the greatest need for telehealth intervention. It demonstrates the full data science lifecycle — from data acquisition and feature engineering through model training, evaluation, and strategic recommendations.

The methodology — risk segmentation using predictive modeling and unsupervised clustering on large, complex datasets — is directly applicable to fraud detection, identity risk scoring, and financial risk product development.

---

## Problem Statement

Millions of Americans in underserved counties lack access to adequate healthcare. The goal was to:

1. **Predict** which counties have the poorest health outcomes based on socioeconomic, demographic, and healthcare access variables
2. **Segment** counties into actionable risk clusters for targeted telehealth deployment
3. **Deliver** executive-level strategy recommendations backed by data

---

## Dataset

**Source:** County Health Rankings & Roadmaps 2024 (University of Wisconsin Population Health Institute)

| Attribute | Detail |
|---|---|
| Records | 3,143 U.S. counties |
| Features | 50+ variables (health behaviors, clinical care, social/economic factors, physical environment) |
| Target Variable | Composite health outcomes score (mortality, morbidity) |
| Missing Data | Handled via median imputation and feature-level drop thresholds |

---

## Methodology

### 1. Data Preprocessing
- Cleaned and standardized 50+ features across all 3,143 counties
- Applied median imputation for missing values; dropped features with >30% missingness
- Encoded categorical variables; scaled continuous features using StandardScaler
- Performed exploratory data analysis (EDA) with correlation heatmaps and distribution analysis

### 2. Supervised Learning — Random Forest Regressor
Built a Random Forest model to predict county-level health outcome scores:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
```

**Model Performance:**

| Metric | Score |
|---|---|
| R² (Test Set) | **0.81** |
| Key Predictors | Uninsured rate, preventable hospitalization rate, income inequality, primary care physician ratio |

### 3. Unsupervised Learning — K-Means Clustering
Segmented counties into distinct risk clusters to prioritize telehealth deployment:

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
county_data['cluster'] = kmeans.fit_predict(X_scaled)
```

Clusters were profiled by health outcome severity, demographic composition, and healthcare access gaps.

---

## Key Findings

- **477 priority counties** identified as highest-risk for poor health outcomes — representing the critical target segment for immediate telehealth expansion
- The strongest predictors of poor county health outcomes were **uninsured rate**, **income inequality (80th/20th percentile income ratio)**, and **preventable hospitalization rate**
- Priority counties are disproportionately concentrated in the **Southeast U.S.**, rural Appalachia, and the Mississippi Delta
- K-Means clustering revealed 4 distinct county profiles ranging from high-access/low-risk urban centers to low-access/high-risk rural counties with no primary care coverage

---

## Strategic Recommendations

Based on model outputs, three telehealth deployment strategies were proposed and evaluated:

| Strategy | Target Counties | Est. Reach | Priority |
|---|---|---|---|
| Rural High-Risk Intervention | 477 priority counties | 12M+ residents | High |
| Preventive Care Expansion | Cluster 2 mid-risk counties | 8M+ residents | Medium |
| Mental Health Access Initiative | Counties with >25% uninsured | 5M+ residents | Medium |

---

## Repository Structure

```
healthcare-risk-modeling/
│
├── data/
│   └── county_health_rankings_2024.csv    # Source data (public)
│
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb     # Exploratory data analysis
│   ├── 02_Random_Forest_Model.ipynb       # Supervised modeling
│   └── 03_KMeans_Clustering.ipynb         # Segmentation analysis
│
├── outputs/
│   ├── priority_counties_477.csv          # Model output: high-risk counties
│   └── cluster_profiles.csv              # K-Means cluster summary
│
├── requirements.txt
└── README.md
```

---

## Technologies Used

- **Python 3.10** — pandas, NumPy, scikit-learn, matplotlib, seaborn
- **Jupyter Notebook** — end-to-end analysis and documentation
- **Tableau** — executive dashboard for stakeholder communication
- **Data Source** — County Health Rankings & Roadmaps 2024 (public dataset)

---

## Skills Demonstrated

`Machine Learning` `Feature Engineering` `Risk Segmentation` `Random Forest` `K-Means Clustering` `Predictive Modeling` `Python` `scikit-learn` `Data Preprocessing` `EDA` `Tableau` `Executive Communication`

---

## About

Built as part of **MBA 551: Predictive Analytics for Big Data** at NC State University's Poole College of Management (GPA: 4.0/4.0).

**Author:** Divya Yadav
**Contact:** dyadav@ncsu.edu | [LinkedIn](https://linkedin.com/in/your-handle)

---

*Methodology and segmentation approach are transferable to financial risk scoring, identity verification, and fraud detection use cases — where large-scale, multi-variable risk modeling drives product decisions.*
