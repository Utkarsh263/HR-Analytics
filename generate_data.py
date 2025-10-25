import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import sqlite3

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_hr_data(n_employees=1500):
    """Generate synthetic HR employee attrition data"""
    
    print("Generating HR Analytics dataset...")
    
    # Employee demographics
    data = {
        'EmployeeID': [f'EMP{str(i).zfill(4)}' for i in range(1, n_employees + 1)],
        'Age': np.random.normal(37, 10, n_employees).astype(int),
        'Gender': np.random.choice(['Male', 'Female'], n_employees, p=[0.6, 0.4]),
        'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], 
                                         n_employees, p=[0.32, 0.46, 0.22])
    }
    
    df = pd.DataFrame(data)
    
    # Clip age to realistic range
    df['Age'] = df['Age'].clip(18, 60)
    
    # Department and Job Role
    departments = ['Sales', 'Research & Development', 'Human Resources', 
                   'IT', 'Finance', 'Operations']
    df['Department'] = np.random.choice(departments, n_employees, 
                                       p=[0.25, 0.30, 0.08, 0.15, 0.12, 0.10])
    
    # Job roles based on department
    job_roles = {
        'Sales': ['Sales Executive', 'Sales Manager', 'Sales Representative'],
        'Research & Development': ['Research Scientist', 'Laboratory Technician', 
                                   'Research Director', 'Software Engineer'],
        'Human Resources': ['HR Manager', 'HR Specialist', 'Recruiter'],
        'IT': ['IT Manager', 'IT Support', 'Systems Administrator', 'Developer'],
        'Finance': ['Finance Manager', 'Financial Analyst', 'Accountant'],
        'Operations': ['Operations Manager', 'Operations Analyst', 'Logistics Coordinator']
    }
    
    df['JobRole'] = df['Department'].apply(lambda x: random.choice(job_roles[x]))
    
    # Education
    df['Education'] = np.random.choice([1, 2, 3, 4, 5], n_employees, 
                                      p=[0.10, 0.20, 0.30, 0.25, 0.15])
    education_map = {1: 'Below College', 2: 'College', 3: 'Bachelor', 
                     4: 'Master', 5: 'Doctorate'}
    df['EducationField'] = df['Education'].map(education_map)
    
    # Work experience
    df['YearsAtCompany'] = np.random.exponential(5, n_employees).clip(0, 40).astype(int)
    df['YearsInCurrentRole'] = (df['YearsAtCompany'] * np.random.uniform(0.3, 0.8, n_employees)).astype(int)
    df['YearsSinceLastPromotion'] = (df['YearsAtCompany'] * np.random.uniform(0.1, 0.5, n_employees)).astype(int)
    df['YearsWithCurrManager'] = (df['YearsInCurrentRole'] * np.random.uniform(0.4, 1.0, n_employees)).astype(int)
    df['TotalWorkingYears'] = df['YearsAtCompany'] + np.random.randint(0, 15, n_employees)
    df['NumCompaniesWorked'] = np.random.poisson(2.5, n_employees).clip(0, 9)
    
    # Job level and salary
    df['JobLevel'] = np.random.choice([1, 2, 3, 4, 5], n_employees, 
                                     p=[0.35, 0.30, 0.20, 0.10, 0.05])
    
    # Base salary on job level, education, and experience
    base_salary = {1: 45000, 2: 65000, 3: 85000, 4: 110000, 5: 150000}
    df['MonthlyIncome'] = df['JobLevel'].map(base_salary)
    df['MonthlyIncome'] = (df['MonthlyIncome'] * 
                           (1 + df['Education'] * 0.05) * 
                           (1 + df['TotalWorkingYears'] * 0.01) * 
                           np.random.uniform(0.85, 1.15, n_employees)).astype(int)
    
    df['HourlyRate'] = (df['MonthlyIncome'] / 160 * np.random.uniform(0.8, 1.2, n_employees)).astype(int)
    df['DailyRate'] = (df['HourlyRate'] * 8 * np.random.uniform(0.9, 1.1, n_employees)).astype(int)
    df['MonthlyRate'] = (df['MonthlyIncome'] * np.random.uniform(0.95, 1.05, n_employees)).astype(int)
    
    # Salary hike
    df['PercentSalaryHike'] = np.random.normal(15, 5, n_employees).clip(11, 25).astype(int)
    
    # Work-life balance and satisfaction
    df['DistanceFromHome'] = np.random.exponential(10, n_employees).clip(1, 29).astype(int)
    df['WorkLifeBalance'] = np.random.choice([1, 2, 3, 4], n_employees, p=[0.08, 0.27, 0.45, 0.20])
    df['JobSatisfaction'] = np.random.choice([1, 2, 3, 4], n_employees, p=[0.12, 0.28, 0.38, 0.22])
    df['EnvironmentSatisfaction'] = np.random.choice([1, 2, 3, 4], n_employees, p=[0.15, 0.25, 0.35, 0.25])
    df['RelationshipSatisfaction'] = np.random.choice([1, 2, 3, 4], n_employees, p=[0.10, 0.30, 0.35, 0.25])
    df['JobInvolvement'] = np.random.choice([1, 2, 3, 4], n_employees, p=[0.08, 0.22, 0.50, 0.20])
    
    # Performance and training
    df['PerformanceRating'] = np.random.choice([3, 4], n_employees, p=[0.85, 0.15])
    df['TrainingTimesLastYear'] = np.random.poisson(2.5, n_employees).clip(0, 6)
    
    # Work conditions
    df['OverTime'] = np.random.choice(['Yes', 'No'], n_employees, p=[0.30, 0.70])
    df['BusinessTravel'] = np.random.choice(['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'], 
                                           n_employees, p=[0.20, 0.60, 0.20])
    df['StockOptionLevel'] = np.random.choice([0, 1, 2, 3], n_employees, p=[0.40, 0.35, 0.15, 0.10])
    
    # ==========================================
    # GENERATE ATTRITION BASED ON RISK FACTORS
    # ==========================================
    
    # Calculate attrition risk score
    df['AttritionRisk'] = 0
    
    # Age factors (younger employees more likely to leave)
    df.loc[df['Age'] < 30, 'AttritionRisk'] += 25
    df.loc[df['Age'].between(30, 40), 'AttritionRisk'] += 15
    df.loc[df['Age'] > 50, 'AttritionRisk'] += 10
    
    # Salary factors (lower salary = higher risk)
    income_quartile = pd.qcut(df['MonthlyIncome'], 4, labels=[1, 2, 3, 4])
    df.loc[income_quartile == 1, 'AttritionRisk'] += 30
    df.loc[income_quartile == 2, 'AttritionRisk'] += 15
    
    # Job satisfaction (low satisfaction = higher risk)
    df.loc[df['JobSatisfaction'] == 1, 'AttritionRisk'] += 35
    df.loc[df['JobSatisfaction'] == 2, 'AttritionRisk'] += 20
    df.loc[df['JobSatisfaction'] == 3, 'AttritionRisk'] += 10
    
    # Work-life balance
    df.loc[df['WorkLifeBalance'] == 1, 'AttritionRisk'] += 30
    df.loc[df['WorkLifeBalance'] == 2, 'AttritionRisk'] += 15
    
    # Years at company (very short or very long tenure)
    df.loc[df['YearsAtCompany'] <= 2, 'AttritionRisk'] += 35
    df.loc[df['YearsAtCompany'] > 15, 'AttritionRisk'] += 10
    
    # Overtime
    df.loc[df['OverTime'] == 'Yes', 'AttritionRisk'] += 25
    
    # Distance from home
    df.loc[df['DistanceFromHome'] > 20, 'AttritionRisk'] += 15
    
    # Years since last promotion
    df.loc[df['YearsSinceLastPromotion'] > 5, 'AttritionRisk'] += 20
    
    # Stock options (lack of stock options = higher risk)
    df.loc[df['StockOptionLevel'] == 0, 'AttritionRisk'] += 20
    
    # Environment satisfaction
    df.loc[df['EnvironmentSatisfaction'] == 1, 'AttritionRisk'] += 20
    df.loc[df['EnvironmentSatisfaction'] == 2, 'AttritionRisk'] += 10
    
    # Job involvement
    df.loc[df['JobInvolvement'] == 1, 'AttritionRisk'] += 25
    df.loc[df['JobInvolvement'] == 2, 'AttritionRisk'] += 10
    
    # Business travel
    df.loc[df['BusinessTravel'] == 'Travel_Frequently', 'AttritionRisk'] += 15
    
    # Marital status (single more likely to leave)
    df.loc[df['MaritalStatus'] == 'Single', 'AttritionRisk'] += 15
    
    # Generate attrition based on risk score
    attrition_probability = df['AttritionRisk'] / df['AttritionRisk'].max()
    df['Attrition'] = (np.random.random(n_employees) < attrition_probability * 0.25).astype(int)
    df['Attrition'] = df['Attrition'].map({0: 'No', 1: 'Yes'})
    
    # Add some randomness to make it realistic
    random_attrition = np.random.choice(df[df['Attrition'] == 'No'].index, size=int(n_employees * 0.02))
    df.loc[random_attrition, 'Attrition'] = 'Yes'
    
    return df

# Generate data
print("=" * 60)
print("HR ANALYTICS - EMPLOYEE ATTRITION DATASET")
print("=" * 60)

df_hr = generate_hr_data(1500)

# Basic statistics
print(f"\nDataset Overview:")
print(f"Total Employees: {len(df_hr)}")
print(f"Attrition Cases: {(df_hr['Attrition'] == 'Yes').sum()}")
print(f"Attrition Rate: {(df_hr['Attrition'] == 'Yes').mean() * 100:.2f}%")
print(f"\nAttrition by Department:")
print(df_hr.groupby('Department')['Attrition'].apply(lambda x: (x == 'Yes').sum()))

# Save to CSV
df_hr.to_csv('hr_employee_data.csv', index=False)
print("\n✓ Data saved to: hr_employee_data.csv")

# Save to SQLite database
conn = sqlite3.connect('hr_analytics.db')
df_hr.to_sql('employees', conn, if_exists='replace', index=False)

# Create additional summary tables
# Attrition summary by department
attrition_dept = df_hr.groupby('Department').agg({
    'EmployeeID': 'count',
    'Attrition': lambda x: (x == 'Yes').sum()
}).reset_index()
attrition_dept.columns = ['Department', 'TotalEmployees', 'AttritionCount']
attrition_dept['AttritionRate'] = (attrition_dept['AttritionCount'] / 
                                    attrition_dept['TotalEmployees'] * 100).round(2)
attrition_dept.to_sql('attrition_by_department', conn, if_exists='replace', index=False)

# Attrition summary by job role
attrition_role = df_hr.groupby('JobRole').agg({
    'EmployeeID': 'count',
    'Attrition': lambda x: (x == 'Yes').sum()
}).reset_index()
attrition_role.columns = ['JobRole', 'TotalEmployees', 'AttritionCount']
attrition_role['AttritionRate'] = (attrition_role['AttritionCount'] / 
                                    attrition_role['TotalEmployees'] * 100).round(2)
attrition_role.to_sql('attrition_by_role', conn, if_exists='replace', index=False)

conn.close()
print("✓ Database created: hr_analytics.db")

# Display sample data
print("\n" + "=" * 60)
print("SAMPLE DATA (First 5 rows)")
print("=" * 60)
print(df_hr.head().to_string())

print("\n" + "=" * 60)
print("DATA GENERATION COMPLETE!")
print("=" * 60)
print("\nFiles created:")
print("  • hr_employee_data.csv")
print("  • hr_analytics.db")
print("\nYou can now proceed with:")
print("  1. Exploratory Data Analysis (EDA)")
print("  2. Machine Learning Model Training")
print("  3. SQL Analysis")
print("  4. Power BI Dashboard Creation")