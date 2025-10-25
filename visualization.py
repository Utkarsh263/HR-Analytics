import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

print("=" * 70)
print("HR ANALYTICS - DATA VISUALIZATIONS")
print("=" * 70)

# Load data
df = pd.read_csv('hr_employee_data.csv')
predictions = pd.read_csv('attrition_predictions.csv')
feature_importance_lr = pd.read_csv('feature_importance_logistic.csv')

# Convert attrition to binary
df['AttritionBinary'] = (df['Attrition'] == 'Yes').astype(int)

print("\nGenerating visualizations...")

# ==========================================
# FIGURE 1: ATTRITION OVERVIEW
# ==========================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('HR Analytics - Attrition Overview Dashboard', fontsize=20, fontweight='bold')

# 1.1 Attrition Distribution
attrition_counts = df['Attrition'].value_counts()
colors = ['#2ecc71', '#e74c3c']
axes[0, 0].pie(attrition_counts, labels=attrition_counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
axes[0, 0].set_title('Overall Attrition Distribution', fontsize=14, fontweight='bold')

# 1.2 Attrition by Department
dept_attrition = df.groupby('Department')['AttritionBinary'].agg(['sum', 'count'])
dept_attrition['rate'] = (dept_attrition['sum'] / dept_attrition['count'] * 100)
dept_attrition = dept_attrition.sort_values('rate', ascending=False)

axes[0, 1].barh(dept_attrition.index, dept_attrition['rate'], color='#3498db')
axes[0, 1].set_xlabel('Attrition Rate (%)', fontsize=12)
axes[0, 1].set_title('Attrition Rate by Department', fontsize=14, fontweight='bold')
axes[0, 1].grid(axis='x', alpha=0.3)

# 1.3 Attrition by Age Group
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100], 
                        labels=['18-25', '26-35', '36-45', '46-55', '56+'])
age_attrition = df.groupby('AgeGroup')['AttritionBinary'].agg(['sum', 'count'])
age_attrition['rate'] = (age_attrition['sum'] / age_attrition['count'] * 100)

axes[1, 0].bar(age_attrition.index, age_attrition['rate'], color='#9b59b6')
axes[1, 0].set_xlabel('Age Group', fontsize=12)
axes[1, 0].set_ylabel('Attrition Rate (%)', fontsize=12)
axes[1, 0].set_title('Attrition Rate by Age Group', fontsize=14, fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)

# 1.4 Attrition by Job Satisfaction
satisfaction_attrition = df.groupby('JobSatisfaction')['AttritionBinary'].agg(['sum', 'count'])
satisfaction_attrition['rate'] = (satisfaction_attrition['sum'] / satisfaction_attrition['count'] * 100)

axes[1, 1].plot(satisfaction_attrition.index, satisfaction_attrition['rate'], 
                marker='o', linewidth=3, markersize=10, color='#e67e22')
axes[1, 1].set_xlabel('Job Satisfaction Level', fontsize=12)
axes[1, 1].set_ylabel('Attrition Rate (%)', fontsize=12)
axes[1, 1].set_title('Attrition Rate by Job Satisfaction', fontsize=14, fontweight='bold')
axes[1, 1].set_xticks([1, 2, 3, 4])
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('attrition_overview.png', dpi=300, bbox_inches='tight')
print("✓ Saved: attrition_overview.png")
plt.close()

# ==========================================
# FIGURE 2: COMPENSATION ANALYSIS
# ==========================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Compensation and Attrition Analysis', fontsize=20, fontweight='bold')

# 2.1 Income Distribution by Attrition
axes[0, 0].hist([df[df['Attrition'] == 'No']['MonthlyIncome'],
                 df[df['Attrition'] == 'Yes']['MonthlyIncome']],
                bins=30, label=['No Attrition', 'Attrition'], color=['#2ecc71', '#e74c3c'], alpha=0.7)
axes[0, 0].set_xlabel('Monthly Income ($)', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('Income Distribution by Attrition Status', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 2.2 Income vs Attrition Rate
df['IncomeQuartile'] = pd.qcut(df['MonthlyIncome'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
income_attrition = df.groupby('IncomeQuartile')['AttritionBinary'].agg(['sum', 'count'])
income_attrition['rate'] = (income_attrition['sum'] / income_attrition['count'] * 100)

axes[0, 1].bar(income_attrition.index, income_attrition['rate'], color='#16a085')
axes[0, 1].set_xlabel('Income Quartile', fontsize=12)
axes[0, 1].set_ylabel('Attrition Rate (%)', fontsize=12)
axes[0, 1].set_title('Attrition Rate by Income Quartile', fontsize=14, fontweight='bold')
axes[0, 1].grid(axis='y', alpha=0.3)

# 2.3 Stock Options Impact
stock_attrition = df.groupby('StockOptionLevel')['AttritionBinary'].agg(['sum', 'count'])
stock_attrition['rate'] = (stock_attrition['sum'] / stock_attrition['count'] * 100)

axes[1, 0].plot(stock_attrition.index, stock_attrition['rate'], 
                marker='s', linewidth=3, markersize=10, color='#c0392b')
axes[1, 0].set_xlabel('Stock Option Level', fontsize=12)
axes[1, 0].set_ylabel('Attrition Rate (%)', fontsize=12)
axes[1, 0].set_title('Stock Options vs Attrition', fontsize=14, fontweight='bold')
axes[1, 0].set_xticks([0, 1, 2, 3])
axes[1, 0].grid(alpha=0.3)

# 2.4 Salary Hike Analysis
df['SalaryHikeCategory'] = pd.cut(df['PercentSalaryHike'], 
                                  bins=[0, 13, 15, 18, 30],
                                  labels=['Low', 'Medium', 'High', 'Very High'])
hike_attrition = df.groupby('SalaryHikeCategory')['AttritionBinary'].agg(['sum', 'count'])
hike_attrition['rate'] = (hike_attrition['sum'] / hike_attrition['count'] * 100)

axes[1, 1].bar(hike_attrition.index, hike_attrition['rate'], color='#f39c12')
axes[1, 1].set_xlabel('Salary Hike Category', fontsize=12)
axes[1, 1].set_ylabel('Attrition Rate (%)', fontsize=12)
axes[1, 1].set_title('Attrition by Salary Hike', fontsize=14, fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('compensation_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: compensation_analysis.png")
plt.close()

# ==========================================
# FIGURE 3: WORK CONDITIONS
# ==========================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Work Conditions and Attrition', fontsize=20, fontweight='bold')

# 3.1 Overtime Impact
overtime_attrition = df.groupby('OverTime')['AttritionBinary'].agg(['sum', 'count'])
overtime_attrition['rate'] = (overtime_attrition['sum'] / overtime_attrition['count'] * 100)

axes[0, 0].bar(overtime_attrition.index, overtime_attrition['rate'], color=['#27ae60', '#e74c3c'])
axes[0, 0].set_xlabel('Overtime', fontsize=12)
axes[0, 0].set_ylabel('Attrition Rate (%)', fontsize=12)
axes[0, 0].set_title('Overtime vs Attrition Rate', fontsize=14, fontweight='bold')
axes[0, 0].grid(axis='y', alpha=0.3)

# 3.2 Work-Life Balance
wlb_attrition = df.groupby('WorkLifeBalance')['AttritionBinary'].agg(['sum', 'count'])
wlb_attrition['rate'] = (wlb_attrition['sum'] / wlb_attrition['count'] * 100)

axes[0, 1].plot(wlb_attrition.index, wlb_attrition['rate'], 
                marker='D', linewidth=3, markersize=10, color='#8e44ad')
axes[0, 1].set_xlabel('Work-Life Balance Level', fontsize=12)
axes[0, 1].set_ylabel('Attrition Rate (%)', fontsize=12)
axes[0, 1].set_title('Work-Life Balance vs Attrition', fontsize=14, fontweight='bold')
axes[0, 1].set_xticks([1, 2, 3, 4])
axes[0, 1].grid(alpha=0.3)

# 3.3 Business Travel
travel_attrition = df.groupby('BusinessTravel')['AttritionBinary'].agg(['sum', 'count'])
travel_attrition['rate'] = (travel_attrition['sum'] / travel_attrition['count'] * 100)

axes[1, 0].barh(travel_attrition.index, travel_attrition['rate'], color='#34495e')
axes[1, 0].set_xlabel('Attrition Rate (%)', fontsize=12)
axes[1, 0].set_title('Business Travel Impact', fontsize=14, fontweight='bold')
axes[1, 0].grid(axis='x', alpha=0.3)

# 3.4 Distance from Home
df['DistanceCategory'] = pd.cut(df['DistanceFromHome'], 
                                bins=[0, 5, 15, 30],
                                labels=['Near', 'Medium', 'Far'])
distance_attrition = df.groupby('DistanceCategory')['AttritionBinary'].agg(['sum', 'count'])
distance_attrition['rate'] = (distance_attrition['sum'] / distance_attrition['count'] * 100)

axes[1, 1].bar(distance_attrition.index, distance_attrition['rate'], color='#16a085')
axes[1, 1].set_xlabel('Distance from Home', fontsize=12)
axes[1, 1].set_ylabel('Attrition Rate (%)', fontsize=12)
axes[1, 1].set_title('Distance vs Attrition', fontsize=14, fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('work_conditions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: work_conditions.png")
plt.close()

# ==========================================
# FIGURE 4: CAREER PROGRESSION
# ==========================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Career Progression and Attrition', fontsize=20, fontweight='bold')

# 4.1 Tenure Analysis
df['TenureCategory'] = pd.cut(df['YearsAtCompany'], 
                              bins=[-1, 2, 5, 10, 50],
                              labels=['0-2', '3-5', '6-10', '10+'])
tenure_attrition = df.groupby('TenureCategory')['AttritionBinary'].agg(['sum', 'count'])
tenure_attrition['rate'] = (tenure_attrition['sum'] / tenure_attrition['count'] * 100)

axes[0, 0].bar(tenure_attrition.index, tenure_attrition['rate'], color='#d35400')
axes[0, 0].set_xlabel('Years at Company', fontsize=12)
axes[0, 0].set_ylabel('Attrition Rate (%)', fontsize=12)
axes[0, 0].set_title('Attrition by Tenure', fontsize=14, fontweight='bold')
axes[0, 0].grid(axis='y', alpha=0.3)

# 4.2 Years Since Last Promotion
df['PromotionCategory'] = pd.cut(df['YearsSinceLastPromotion'], 
                                 bins=[-1, 1, 3, 7, 20],
                                 labels=['<1', '1-3', '3-7', '7+'])
promo_attrition = df.groupby('PromotionCategory')['AttritionBinary'].agg(['sum', 'count'])
promo_attrition['rate'] = (promo_attrition['sum'] / promo_attrition['count'] * 100)

axes[0, 1].bar(promo_attrition.index, promo_attrition['rate'], color='#c0392b')
axes[0, 1].set_xlabel('Years Since Last Promotion', fontsize=12)
axes[0, 1].set_ylabel('Attrition Rate (%)', fontsize=12)
axes[0, 1].set_title('Promotion Gap vs Attrition', fontsize=14, fontweight='bold')
axes[0, 1].grid(axis='y', alpha=0.3)

# 4.3 Job Level
level_attrition = df.groupby('JobLevel')['AttritionBinary'].agg(['sum', 'count'])
level_attrition['rate'] = (level_attrition['sum'] / level_attrition['count'] * 100)

axes[1, 0].plot(level_attrition.index, level_attrition['rate'], 
                marker='o', linewidth=3, markersize=10, color='#2980b9')
axes[1, 0].set_xlabel('Job Level', fontsize=12)
axes[1, 0].set_ylabel('Attrition Rate (%)', fontsize=12)
axes[1, 0].set_title('Job Level vs Attrition', fontsize=14, fontweight='bold')
axes[1, 0].set_xticks([1, 2, 3, 4, 5])
axes[1, 0].grid(alpha=0.3)

# 4.4 Training Times
training_attrition = df.groupby('TrainingTimesLastYear')['AttritionBinary'].agg(['sum', 'count'])
training_attrition['rate'] = (training_attrition['sum'] / training_attrition['count'] * 100)

axes[1, 1].bar(training_attrition.index, training_attrition['rate'], color='#27ae60')
axes[1, 1].set_xlabel('Training Times Last Year', fontsize=12)
axes[1, 1].set_ylabel('Attrition Rate (%)', fontsize=12)
axes[1, 1].set_title('Training vs Attrition', fontsize=14, fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('career_progression.png', dpi=300, bbox_inches='tight')
print("✓ Saved: career_progression.png")
plt.close()

# ==========================================
# FIGURE 5: CORRELATION HEATMAP
# ==========================================

fig, ax = plt.subplots(figsize=(14, 10))

# Select numeric columns for correlation
numeric_cols = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'YearsInCurrentRole',
                'YearsSinceLastPromotion', 'TotalWorkingYears', 'JobSatisfaction',
                'EnvironmentSatisfaction', 'WorkLifeBalance', 'JobInvolvement',
                'PerformanceRating', 'DistanceFromHome', 'PercentSalaryHike',
                'StockOptionLevel', 'TrainingTimesLastYear', 'AttritionBinary']

corr_matrix = df[numeric_cols].corr()

# Create heatmap
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn_r', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Correlation Heatmap - Key HR Metrics', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: correlation_heatmap.png")
plt.close()

# ==========================================
# FIGURE 6: MODEL PERFORMANCE
# ==========================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Machine Learning Model Performance', fontsize=20, fontweight='bold')

# 6.1 ROC Curve
from sklearn.metrics import roc_curve, roc_auc_score

# Load predictions and calculate ROC
y_true = predictions['Actual_Attrition'].values
y_prob = predictions['Attrition_Probability'].values

fpr, tpr, thresholds = roc_curve(y_true, y_prob)
roc_auc = roc_auc_score(y_true, y_prob)

axes[0].plot(fpr, tpr, color='#e74c3c', linewidth=3, 
             label=f'ROC Curve (AUC = {roc_auc:.3f})')
axes[0].plot([0, 1], [0, 1], color='#95a5a6', linestyle='--', linewidth=2, label='Random Classifier')
axes[0].set_xlabel('False Positive Rate', fontsize=12)
axes[0].set_ylabel('True Positive Rate', fontsize=12)
axes[0].set_title('ROC Curve - Logistic Regression', fontsize=14, fontweight='bold')
axes[0].legend(loc='lower right', fontsize=11)
axes[0].grid(alpha=0.3)

# 6.2 Feature Importance (Top 15)
top_features = feature_importance_lr.head(15)

axes[1].barh(range(len(top_features)), top_features['Abs_Coefficient'], color='#3498db')
axes[1].set_yticks(range(len(top_features)))
axes[1].set_yticklabels(top_features['Feature'])
axes[1].set_xlabel('Absolute Coefficient Value', fontsize=12)
axes[1].set_title('Top 15 Most Important Features', fontsize=14, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: model_performance.png")
plt.close()

# ==========================================
# FIGURE 7: RISK DISTRIBUTION
# ==========================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Attrition Risk Distribution', fontsize=20, fontweight='bold')

# 7.1 Risk Level Distribution
risk_counts = predictions['Risk_Level'].value_counts()
colors_risk = ['#2ecc71', '#f39c12', '#e74c3c']

axes[0].pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%',
            colors=colors_risk, startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
axes[0].set_title('Employee Risk Level Distribution', fontsize=14, fontweight='bold')

# 7.2 Probability Distribution
axes[1].hist(predictions['Attrition_Probability'], bins=30, 
             color='#9b59b6', edgecolor='black', alpha=0.7)
axes[1].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
axes[1].set_xlabel('Attrition Probability', fontsize=12)
axes[1].set_ylabel('Number of Employees', fontsize=12)
axes[1].set_title('Distribution of Attrition Probability', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('risk_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: risk_distribution.png")
plt.close()

# ==========================================
# FIGURE 8: DEMOGRAPHIC ANALYSIS
# ==========================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Demographic Factors and Attrition', fontsize=20, fontweight='bold')

# 8.1 Gender
gender_attrition = df.groupby('Gender')['AttritionBinary'].agg(['sum', 'count'])
gender_attrition['rate'] = (gender_attrition['sum'] / gender_attrition['count'] * 100)

axes[0, 0].bar(gender_attrition.index, gender_attrition['rate'], color=['#3498db', '#e74c3c'])
axes[0, 0].set_xlabel('Gender', fontsize=12)
axes[0, 0].set_ylabel('Attrition Rate (%)', fontsize=12)
axes[0, 0].set_title('Attrition by Gender', fontsize=14, fontweight='bold')
axes[0, 0].grid(axis='y', alpha=0.3)

# 8.2 Marital Status
marital_attrition = df.groupby('MaritalStatus')['AttritionBinary'].agg(['sum', 'count'])
marital_attrition['rate'] = (marital_attrition['sum'] / marital_attrition['count'] * 100)

axes[0, 1].bar(marital_attrition.index, marital_attrition['rate'], color='#9b59b6')
axes[0, 1].set_xlabel('Marital Status', fontsize=12)
axes[0, 1].set_ylabel('Attrition Rate (%)', fontsize=12)
axes[0, 1].set_title('Attrition by Marital Status', fontsize=14, fontweight='bold')
axes[0, 1].grid(axis='y', alpha=0.3)

# 8.3 Education
education_attrition = df.groupby('EducationField')['AttritionBinary'].agg(['sum', 'count'])
education_attrition['rate'] = (education_attrition['sum'] / education_attrition['count'] * 100)
education_attrition = education_attrition.sort_values('rate', ascending=False)

axes[1, 0].barh(education_attrition.index, education_attrition['rate'], color='#16a085')
axes[1, 0].set_xlabel('Attrition Rate (%)', fontsize=12)
axes[1, 0].set_title('Attrition by Education Level', fontsize=14, fontweight='bold')
axes[1, 0].grid(axis='x', alpha=0.3)

# 8.4 Age Distribution by Attrition
axes[1, 1].hist([df[df['Attrition'] == 'No']['Age'],
                 df[df['Attrition'] == 'Yes']['Age']],
                bins=20, label=['No Attrition', 'Attrition'], 
                color=['#2ecc71', '#e74c3c'], alpha=0.7)
axes[1, 1].set_xlabel('Age', fontsize=12)
axes[1, 1].set_ylabel('Frequency', fontsize=12)
axes[1, 1].set_title('Age Distribution by Attrition Status', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('demographic_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: demographic_analysis.png")
plt.close()

# ==========================================
# FIGURE 9: SATISFACTION METRICS
# ==========================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Satisfaction Metrics and Attrition', fontsize=20, fontweight='bold')

satisfaction_metrics = [
    ('JobSatisfaction', 'Job Satisfaction'),
    ('EnvironmentSatisfaction', 'Environment Satisfaction'),
    ('WorkLifeBalance', 'Work-Life Balance'),
    ('RelationshipSatisfaction', 'Relationship Satisfaction')
]

colors_sat = ['#e74c3c', '#f39c12', '#27ae60', '#3498db']

for idx, (col, title) in enumerate(satisfaction_metrics):
    row = idx // 2
    col_idx = idx % 2
    
    sat_data = df.groupby(col)['AttritionBinary'].agg(['sum', 'count'])
    sat_data['rate'] = (sat_data['sum'] / sat_data['count'] * 100)
    
    axes[row, col_idx].plot(sat_data.index, sat_data['rate'], 
                            marker='o', linewidth=3, markersize=10, color=colors_sat[idx])
    axes[row, col_idx].set_xlabel('Satisfaction Level', fontsize=12)
    axes[row, col_idx].set_ylabel('Attrition Rate (%)', fontsize=12)
    axes[row, col_idx].set_title(f'{title} vs Attrition', fontsize=14, fontweight='bold')
    axes[row, col_idx].set_xticks([1, 2, 3, 4])
    axes[row, col_idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('satisfaction_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: satisfaction_metrics.png")
plt.close()

# ==========================================
# FIGURE 10: EXECUTIVE SUMMARY
# ==========================================

fig = plt.figure(figsize=(16, 10))
fig.suptitle('HR Analytics - Executive Summary Dashboard', fontsize=22, fontweight='bold')

# Create grid
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

# Key Metrics
ax1 = fig.add_subplot(gs[0, :])
ax1.axis('off')

total_emp = len(df)
attrition_count = (df['Attrition'] == 'Yes').sum()
attrition_rate = (attrition_count / total_emp * 100)
avg_age_left = df[df['Attrition'] == 'Yes']['Age'].mean()
avg_income_left = df[df['Attrition'] == 'Yes']['MonthlyIncome'].mean()
high_risk = (predictions['Risk_Level'] == 'High').sum()

metrics_text = f"""
KEY METRICS:
• Total Employees: {total_emp:,}
• Attrition Count: {attrition_count} employees
• Attrition Rate: {attrition_rate:.2f}%
• Average Age of Leavers: {avg_age_left:.1f} years
• Average Income of Leavers: ${avg_income_left:,.0f}
• High-Risk Employees: {high_risk} employees
"""

ax1.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=14, 
         bbox=dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='black', linewidth=2))

# Top Attrition Departments
ax2 = fig.add_subplot(gs[1, 0])
top_dept = dept_attrition.head(3)
ax2.barh(range(len(top_dept)), top_dept['rate'], color='#e74c3c')
ax2.set_yticks(range(len(top_dept)))
ax2.set_yticklabels(top_dept.index)
ax2.set_xlabel('Attrition Rate (%)', fontsize=10)
ax2.set_title('Top 3 Departments', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# Overtime Impact
ax3 = fig.add_subplot(gs[1, 1])
ax3.bar(overtime_attrition.index, overtime_attrition['rate'], color=['#27ae60', '#e74c3c'])
ax3.set_ylabel('Attrition Rate (%)', fontsize=10)
ax3.set_title('Overtime Impact', fontsize=12, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# Job Satisfaction
ax4 = fig.add_subplot(gs[1, 2])
ax4.plot(satisfaction_attrition.index, satisfaction_attrition['rate'], 
         marker='o', linewidth=2, markersize=8, color='#9b59b6')
ax4.set_xlabel('Satisfaction Level', fontsize=10)
ax4.set_ylabel('Attrition Rate (%)', fontsize=10)
ax4.set_title('Job Satisfaction', fontsize=12, fontweight='bold')
ax4.set_xticks([1, 2, 3, 4])
ax4.grid(alpha=0.3)

# Risk Distribution
ax5 = fig.add_subplot(gs[2, :2])
risk_dept = predictions.merge(df[['EmployeeID', 'Department']], on='EmployeeID')
risk_by_dept = risk_dept.groupby(['Department', 'Risk_Level']).size().unstack(fill_value=0)
risk_by_dept.plot(kind='bar', stacked=True, ax=ax5, color=['#2ecc71', '#f39c12', '#e74c3c'])
ax5.set_xlabel('Department', fontsize=10)
ax5.set_ylabel('Number of Employees', fontsize=10)
ax5.set_title('Risk Distribution by Department', fontsize=12, fontweight='bold')
ax5.legend(title='Risk Level', loc='upper right')
ax5.grid(axis='y', alpha=0.3)
ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha='right')

# Recommendations
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')

recommendations = """
RECOMMENDATIONS:

✓ Focus on early-tenure
  employees (0-2 years)

✓ Address low satisfaction
  (Levels 1-2)

✓ Review overtime policies

✓ Competitive compensation
  for Q1 earners

✓ Promote deserving staff

✓ Engage high-risk group
"""

ax6.text(0.1, 0.5, recommendations, ha='left', va='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='#fff9e6', edgecolor='black', linewidth=1.5))

plt.savefig('executive_summary.png', dpi=300, bbox_inches='tight')
print("✓ Saved: executive_summary.png")
plt.close()

# ==========================================
# SUMMARY
# ==========================================

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE!")
print("=" * 70)
print("\nGenerated Visualizations:")
print("  1. attrition_overview.png - Overall attrition metrics")
print("  2. compensation_analysis.png - Salary and benefits impact")
print("  3. work_conditions.png - Work environment factors")
print("  4. career_progression.png - Career development metrics")
print("  5. correlation_heatmap.png - Variable correlations")
print("  6. model_performance.png - ML model results")
print("  7. risk_distribution.png - Employee risk levels")
print("  8. demographic_analysis.png - Demographic factors")
print("  9. satisfaction_metrics.png - Satisfaction analysis")
print(" 10. executive_summary.png - Executive dashboard")
print("\nAll visualizations saved in high resolution (300 DPI)")
print("\nThese images can be used in:")
print("  • PowerPoint presentations")
print("  • Reports and documentation")
print("  • Power BI dashboards")
print("  • Stakeholder meetings")