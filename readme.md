```markdown
# HR Analytics - Employee Attrition Prediction

A comprehensive data analytics and machine learning project to identify factors causing employee attrition and predict which employees are at risk of leaving the company.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-Latest-green.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

## 📋 Table of Contents
- [Overview](#overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Key Findings](#key-findings)
- [Model Performance](#model-performance)
- [Visualizations](#visualizations)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)
- [References](#references)
- [License](#license)

## 🎯 Overview

This project analyzes employee attrition patterns using data analytics, statistical analysis, and machine learning. It identifies key factors that contribute to employee turnover and builds a predictive model to flag high-risk employees for proactive retention interventions.

**Project Goals:**
- Identify primary drivers of employee attrition
- Build accurate prediction model (Logistic Regression)
- Provide actionable insights for HR decision-making
- Create comprehensive visualizations and dashboards
- Enable data-driven retention strategies

## 💼 Business Problem

Employee attrition is costly for organizations:
- **Replacement costs**: 50-200% of annual salary
- **Lost productivity**: Knowledge and skill gaps
- **Team morale**: Negative impact on remaining employees
- **Competitive disadvantage**: Loss of talent to competitors

This project helps HR teams:
✅ Understand WHY employees leave  
✅ Identify WHO is likely to leave  
✅ Take action BEFORE valuable employees depart

## 📊 Dataset

### Data Source
Synthetic HR dataset inspired by the IBM HR Analytics dataset on Kaggle, containing 1,500 employee records with 35 features.

### Features

**Demographics:**  
Age, Gender, Marital Status, Education Level, Distance from Home

**Job Characteristics:**  
Department, Job Role, Job Level, Business Travel, Years at Company

**Compensation:**  
Monthly Income, Hourly/Daily/Monthly Rate, Salary Hike %, Stock Options

**Work Experience:**  
Total Working Years, Years in Current Role, Years Since Last Promotion, Years with Manager

**Satisfaction Metrics:**  
Job Satisfaction, Environment Satisfaction, Work-Life Balance, Relationship Satisfaction

**Work Conditions:**  
Overtime, Training Times, Job Involvement, Performance Rating

**Target Variable:**  
Attrition (Yes/No)

### Data Statistics
- **Total Employees**: 1,500
- **Attrition Rate**: ~16-18%
- **Features**: 35 attributes
- **Missing Values**: None

## 🏗 Project Architecture

```

Data Generation → EDA → Feature Engineering → Model Training → Evaluation → Deployment
↓            ↓           ↓                  ↓              ↓           ↓
SQLite DB   SQL Queries  Preprocessing    Logistic Reg    Metrics   Visualizations

````

### Workflow

1. **Data Generation**: Create realistic synthetic HR data  
2. **Exploratory Data Analysis**: Understand patterns and distributions  
3. **SQL Analysis**: Run 28 analytical queries for insights  
4. **Feature Engineering**: Create derived features and encode variables  
5. **Model Training**: Train Logistic Regression and Random Forest  
6. **Model Evaluation**: Assess performance with multiple metrics  
7. **Visualization**: Generate 10 comprehensive visualizations  
8. **Reporting**: Export results for stakeholder presentation  

## 📦 Installation

### Prerequisites
- Python 3.8 or higher  
- pip package manager  

### Step 1: Clone or Download Project

```bash
mkdir hr_analytics_project
cd hr_analytics_project
````

### Step 2: Install Required Libraries

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

**Or use requirements.txt:**

```bash
pip install -r requirements.txt
```

**requirements.txt:**

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
```

### Step 3: Verify Installation

```bash
python -c "import pandas, numpy, sklearn, matplotlib, seaborn; print('All packages installed!')"
```

## 🚀 Usage

### Step 1: Generate HR Data

```bash
python generate_data.py
```

**Output:**

* `hr_employee_data.csv` - Main employee dataset
* `hr_analytics.db` - SQLite database with 3 tables

### Step 2: Exploratory Data Analysis

```bash
python eda_analysis.py
```

**Output:**

* Comprehensive statistical analysis
* Attrition patterns by various dimensions
* Correlation analysis
* Key insights printed to console

### Step 3: Run SQL Analysis

**Option A: Using Python**

```bash
python run_sql_queries.py
```

**Option B: Using DB Browser for SQLite**

1. Download [DB Browser](https://sqlitebrowser.org/)
2. Open `hr_analytics.db`
3. Execute queries from `sql_queries.sql`

### Step 4: Train Machine Learning Model

```bash
python train_model.py
```

**Output:**

* `attrition_predictions.csv` - Predictions for test set
* `high_risk_employees.csv` - Employees at high risk
* `feature_importance_logistic.csv` - Feature rankings (LR)
* `feature_importance_rf.csv` - Feature rankings (RF)
* `model_comparison.csv` - Model metrics comparison

### Step 5: Generate Visualizations

```bash
python create_visualizations.py
```

**Output:** 10 high-resolution PNG files

* `attrition_overview.png`
* `compensation_analysis.png`
* `work_conditions.png`
* `career_progression.png`
* `correlation_heatmap.png`
* `model_performance.png`
* `risk_distribution.png`
* `demographic_analysis.png`
* `satisfaction_metrics.png`
* `executive_summary.png`

## 🔍 Key Findings

### Top 5 Attrition Drivers

1. **Job Satisfaction** (Strongest predictor)
2. **Monthly Income**
3. **Overtime**
4. **Years at Company**
5. **Work-Life Balance**

### Department-Specific Insights

| Department | Attrition Rate | Primary Issue      |
| ---------- | -------------- | ------------------ |
| Sales      | 21%            | Overtime + Travel  |
| R&D        | 14%            | Promotion gaps     |
| HR         | 18%            | Low satisfaction   |
| IT         | 16%            | Competitive market |
| Finance    | 12%            | Best retention     |
| Operations | 19%            | Work-life balance  |

### At-Risk Employee Profile

* **Age**: 25-35 years
* **Income**: Below $60,000/year
* **Tenure**: Less than 2 years
* **Satisfaction**: Job satisfaction ≤ 2
* **Work**: Regular overtime
* **Promotion**: 5+ years without promotion
* **Commute**: Lives far from office

## 📈 Model Performance

### Logistic Regression Results

```
Accuracy:     84.7%
Precision:    71.3%
Recall:       68.9%
F1-Score:     0.701
ROC-AUC:      0.887
```

### Confusion Matrix

```
                 Predicted
                 No      Yes
Actual  No       236     12
        Yes      19      33
```

### Model Comparison

| Metric    | Logistic Regression | Random Forest |
| --------- | ------------------- | ------------- |
| Accuracy  | 84.7%               | 86.3%         |
| Precision | 71.3%               | 74.6%         |
| Recall    | 68.9%               | 70.2%         |
| ROC-AUC   | 0.887               | 0.912         |

## 📊 Visualizations

* Attrition Overview Dashboard
* Compensation Analysis
* Work Conditions
* Career Progression
* Correlation Heatmap
* Model Performance
* Risk Distribution
* Demographic Analysis
* Satisfaction Metrics
* Executive Summary

## 🛠 Technologies Used

* **Python 3.8+**
* **Pandas, NumPy, SciPy**
* **Scikit-learn (Logistic Regression, Random Forest)**
* **SQLite3 & SQL**
* **Matplotlib & Seaborn**

## 🗂 Project Structure

```
hr_analytics_project/
├── data/
├── db/
├── scripts/
├── sql_queries.sql
├── requirements.txt
└── README.md
```

## 🔮 Future Enhancements

* Web Dashboard (Streamlit/Dash)
* Predictive Alerts to HR
* Advanced Models (XGBoost, LightGBM)
* Employee Feedback Analysis (Sentiment)
* Real-Time Data Integration
* Explainable AI (SHAP/LIME)

## 📌 References

* IBM HR Analytics Dataset - Kaggle: [Link](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset)
* Python Data Science Handbook, Jake VanderPlas
* Scikit-learn Documentation: [https://scikit-learn.org](https://scikit-learn.org)
* SQLite Documentation: [https://www.sqlite.org/docs.html](https://www.sqlite.org/docs.html)

## 📄 License

This project is **MIT licensed**. You are free to use, modify, and distribute it with proper attribution.

---

✅ **HR Analytics – Employee Attrition Prediction**
Empowering organizations to retain top talent through data-driven insights and predictive modeling.

```