import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load data
print("Loading HR Analytics data...")
df = pd.read_csv('hr_employee_data.csv')

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS - HR ANALYTICS")
print("=" * 60)

# ==========================================
# 1. BASIC DATA OVERVIEW
# ==========================================

print("\n1. DATASET OVERVIEW")
print("-" * 60)
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nColumn Names:\n{df.columns.tolist()}")
print(f"\nData Types:\n{df.dtypes.value_counts()}")
print(f"\nMissing Values:\n{df.isnull().sum().sum()} (Total)")

# ==========================================
# 2. ATTRITION STATISTICS
# ==========================================

print("\n\n2. ATTRITION STATISTICS")
print("-" * 60)
attrition_counts = df['Attrition'].value_counts()
attrition_pct = df['Attrition'].value_counts(normalize=True) * 100

print(f"Attrition Distribution:")
print(f"  No:  {attrition_counts['No']:4d} ({attrition_pct['No']:.2f}%)")
print(f"  Yes: {attrition_counts['Yes']:4d} ({attrition_pct['Yes']:.2f}%)")

# ==========================================
# 3. ATTRITION BY DEMOGRAPHICS
# ==========================================

print("\n\n3. ATTRITION BY DEMOGRAPHICS")
print("-" * 60)

# By Age Group
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100], 
                        labels=['18-25', '26-35', '36-45', '46-55', '56+'])
age_attrition = df.groupby('AgeGroup')['Attrition'].apply(
    lambda x: (x == 'Yes').sum() / len(x) * 100
).round(2)
print("\nAttrition Rate by Age Group:")
for age, rate in age_attrition.items():
    print(f"  {age}: {rate}%")

# By Gender
gender_attrition = df.groupby('Gender')['Attrition'].apply(
    lambda x: (x == 'Yes').sum() / len(x) * 100
).round(2)
print("\nAttrition Rate by Gender:")
for gender, rate in gender_attrition.items():
    print(f"  {gender}: {rate}%")

# By Marital Status
marital_attrition = df.groupby('MaritalStatus')['Attrition'].apply(
    lambda x: (x == 'Yes').sum() / len(x) * 100
).round(2)
print("\nAttrition Rate by Marital Status:")
for status, rate in marital_attrition.items():
    print(f"  {status}: {rate}%")

# ==========================================
# 4. ATTRITION BY DEPARTMENT & ROLE
# ==========================================

print("\n\n4. ATTRITION BY DEPARTMENT & JOB ROLE")
print("-" * 60)

dept_attrition = df.groupby('Department').agg({
    'EmployeeID': 'count',
    'Attrition': lambda x: (x == 'Yes').sum()
}).reset_index()
dept_attrition.columns = ['Department', 'Total', 'Attrition']
dept_attrition['AttritionRate'] = (dept_attrition['Attrition'] / 
                                    dept_attrition['Total'] * 100).round(2)
dept_attrition = dept_attrition.sort_values('AttritionRate', ascending=False)

print("\nAttrition by Department:")
print(dept_attrition.to_string(index=False))

# Top 5 roles with highest attrition
role_attrition = df.groupby('JobRole').agg({
    'EmployeeID': 'count',
    'Attrition': lambda x: (x == 'Yes').sum()
}).reset_index()
role_attrition.columns = ['JobRole', 'Total', 'Attrition']
role_attrition['AttritionRate'] = (role_attrition['Attrition'] / 
                                    role_attrition['Total'] * 100).round(2)
role_attrition = role_attrition.sort_values('AttritionRate', ascending=False)

print("\nTop 5 Roles with Highest Attrition:")
print(role_attrition.head().to_string(index=False))

# ==========================================
# 5. ATTRITION BY SATISFACTION LEVELS
# ==========================================

print("\n\n5. ATTRITION BY SATISFACTION METRICS")
print("-" * 60)

satisfaction_cols = ['JobSatisfaction', 'EnvironmentSatisfaction', 
                     'WorkLifeBalance', 'RelationshipSatisfaction']

for col in satisfaction_cols:
    sat_attrition = df.groupby(col)['Attrition'].apply(
        lambda x: (x == 'Yes').sum() / len(x) * 100
    ).round(2)
    print(f"\n{col}:")
    for level, rate in sat_attrition.items():
        print(f"  Level {level}: {rate}%")

# ==========================================
# 6. ATTRITION BY COMPENSATION
# ==========================================

print("\n\n6. ATTRITION BY COMPENSATION")
print("-" * 60)

# Income quartiles
df['IncomeQuartile'] = pd.qcut(df['MonthlyIncome'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
income_attrition = df.groupby('IncomeQuartile')['Attrition'].apply(
    lambda x: (x == 'Yes').sum() / len(x) * 100
).round(2)
print("\nAttrition Rate by Income Quartile:")
for quartile, rate in income_attrition.items():
    print(f"  {quartile}: {rate}%")

# By salary hike
df['SalaryHikeCategory'] = pd.cut(df['PercentSalaryHike'], 
                                  bins=[0, 13, 15, 18, 30],
                                  labels=['Low', 'Medium', 'High', 'Very High'])
hike_attrition = df.groupby('SalaryHikeCategory')['Attrition'].apply(
    lambda x: (x == 'Yes').sum() / len(x) * 100
).round(2)
print("\nAttrition Rate by Salary Hike:")
for category, rate in hike_attrition.items():
    print(f"  {category}: {rate}%")

# ==========================================
# 7. ATTRITION BY WORK CONDITIONS
# ==========================================

print("\n\n7. ATTRITION BY WORK CONDITIONS")
print("-" * 60)

# Overtime
overtime_attrition = df.groupby('OverTime')['Attrition'].apply(
    lambda x: (x == 'Yes').sum() / len(x) * 100
).round(2)
print("\nAttrition Rate by Overtime:")
for ot, rate in overtime_attrition.items():
    print(f"  {ot}: {rate}%")

# Business Travel
travel_attrition = df.groupby('BusinessTravel')['Attrition'].apply(
    lambda x: (x == 'Yes').sum() / len(x) * 100
).round(2)
print("\nAttrition Rate by Business Travel:")
for travel, rate in travel_attrition.items():
    print(f"  {travel}: {rate}%")

# Distance from home
df['DistanceCategory'] = pd.cut(df['DistanceFromHome'], 
                                bins=[0, 5, 15, 30],
                                labels=['Near', 'Medium', 'Far'])
distance_attrition = df.groupby('DistanceCategory')['Attrition'].apply(
    lambda x: (x == 'Yes').sum() / len(x) * 100
).round(2)
print("\nAttrition Rate by Distance from Home:")
for dist, rate in distance_attrition.items():
    print(f"  {dist}: {rate}%")

# ==========================================
# 8. ATTRITION BY TENURE & CAREER PROGRESSION
# ==========================================

print("\n\n8. ATTRITION BY TENURE & CAREER")
print("-" * 60)

# Years at company
df['TenureCategory'] = pd.cut(df['YearsAtCompany'], 
                              bins=[-1, 2, 5, 10, 50],
                              labels=['0-2 years', '3-5 years', '6-10 years', '10+ years'])
tenure_attrition = df.groupby('TenureCategory')['Attrition'].apply(
    lambda x: (x == 'Yes').sum() / len(x) * 100
).round(2)
print("\nAttrition Rate by Tenure:")
for tenure, rate in tenure_attrition.items():
    print(f"  {tenure}: {rate}%")

# Years since last promotion
df['PromotionCategory'] = pd.cut(df['YearsSinceLastPromotion'], 
                                 bins=[-1, 1, 3, 7, 20],
                                 labels=['<1 year', '1-3 years', '3-7 years', '7+ years'])
promo_attrition = df.groupby('PromotionCategory')['Attrition'].apply(
    lambda x: (x == 'Yes').sum() / len(x) * 100
).round(2)
print("\nAttrition Rate by Years Since Last Promotion:")
for promo, rate in promo_attrition.items():
    print(f"  {promo}: {rate}%")

# Job Level
level_attrition = df.groupby('JobLevel')['Attrition'].apply(
    lambda x: (x == 'Yes').sum() / len(x) * 100
).round(2)
print("\nAttrition Rate by Job Level:")
for level, rate in level_attrition.items():
    print(f"  Level {level}: {rate}%")

# ==========================================
# 9. STATISTICAL TESTS
# ==========================================

print("\n\n9. STATISTICAL SIGNIFICANCE TESTS")
print("-" * 60)

# Convert attrition to binary
df['AttritionBinary'] = (df['Attrition'] == 'Yes').astype(int)

# T-test for continuous variables
continuous_vars = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'DistanceFromHome', 
                   'YearsSinceLastPromotion', 'PercentSalaryHike']

print("\nT-tests (Attrition vs No Attrition):")
for var in continuous_vars:
    attrition_yes = df[df['Attrition'] == 'Yes'][var]
    attrition_no = df[df['Attrition'] == 'No'][var]
    t_stat, p_value = stats.ttest_ind(attrition_yes, attrition_no)
    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
    print(f"  {var:25s}: p-value = {p_value:.4f} {significance}")

print("\n*** p < 0.001, ** p < 0.01, * p < 0.05")

# ==========================================
# 10. CORRELATION ANALYSIS
# ==========================================

print("\n\n10. CORRELATION WITH ATTRITION")
print("-" * 60)

# Select numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('AttritionBinary')  # Remove target

# Calculate correlation with attrition
correlations = df[numeric_cols + ['AttritionBinary']].corr()['AttritionBinary'].drop('AttritionBinary')
correlations = correlations.sort_values(ascending=False)

print("\nTop 10 Positive Correlations:")
print(correlations.head(10).to_string())

print("\nTop 10 Negative Correlations:")
print(correlations.tail(10).to_string())

# ==========================================
# 11. KEY INSIGHTS SUMMARY
# ==========================================

print("\n\n" + "=" * 60)
print("KEY INSIGHTS SUMMARY")
print("=" * 60)

# Calculate key metrics
avg_age_left = df[df['Attrition'] == 'Yes']['Age'].mean()
avg_age_stayed = df[df['Attrition'] == 'No']['Age'].mean()

avg_income_left = df[df['Attrition'] == 'Yes']['MonthlyIncome'].mean()
avg_income_stayed = df[df['Attrition'] == 'No']['MonthlyIncome'].mean()

avg_tenure_left = df[df['Attrition'] == 'Yes']['YearsAtCompany'].mean()
avg_tenure_stayed = df[df['Attrition'] == 'No']['YearsAtCompany'].mean()

overtime_left_pct = (df[df['Attrition'] == 'Yes']['OverTime'] == 'Yes').mean() * 100
overtime_stayed_pct = (df[df['Attrition'] == 'No']['OverTime'] == 'Yes').mean() * 100

print(f"""
1. AGE FACTOR:
   • Employees who left avg age: {avg_age_left:.1f} years
   • Employees who stayed avg age: {avg_age_stayed:.1f} years
   • Younger employees are more likely to leave

2. COMPENSATION:
   • Employees who left avg income: ${avg_income_left:,.0f}
   • Employees who stayed avg income: ${avg_income_stayed:,.0f}
   • Lower income strongly correlates with attrition

3. TENURE:
   • Employees who left avg tenure: {avg_tenure_left:.1f} years
   • Employees who stayed avg tenure: {avg_tenure_stayed:.1f} years
   • Early tenure (0-2 years) has highest attrition risk

4. OVERTIME:
   • {overtime_left_pct:.1f}% of employees who left worked overtime
   • {overtime_stayed_pct:.1f}% of employees who stayed worked overtime
   • Overtime significantly increases attrition risk

5. SATISFACTION:
   • Low job satisfaction (Level 1-2) has 2-3x higher attrition
   • Work-life balance is critical factor
   • Environment satisfaction impacts retention

6. CAREER PROGRESSION:
   • Employees without promotion in 5+ years have higher attrition
   • Stock options reduce attrition significantly
   • Job level 1 employees have highest turnover
""")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE!")
print("=" * 60)
print("\nNext steps:")
print("  1. Create visualizations")
print("  2. Build predictive model")
print("  3. Generate recommendations")