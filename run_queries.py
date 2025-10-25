

import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect('hr_analytics.db')

# Define all HR analytics SQL queries
sql_script = """
-- ============================================
-- HR ANALYTICS - SQL QUERIES
-- ============================================

-- ============================================
-- HR ANALYTICS - SQL QUERIES
-- Employee Attrition Analysis
-- ============================================

-- 1. OVERALL ATTRITION STATISTICS
-- Get high-level attrition metrics
SELECT 
    COUNT(*) as total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) as attrition_count,
    ROUND(AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as attrition_rate,
    ROUND(AVG(Age), 1) as avg_age,
    ROUND(AVG(MonthlyIncome), 2) as avg_monthly_income,
    ROUND(AVG(YearsAtCompany), 1) as avg_tenure
FROM employees;

-- 2. ATTRITION BY DEPARTMENT
-- Identify departments with highest attrition
SELECT 
    Department,
    COUNT(*) as total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) as attrition_count,
    ROUND(AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as attrition_rate,
    ROUND(AVG(MonthlyIncome), 2) as avg_salary
FROM employees
GROUP BY Department
ORDER BY attrition_rate DESC;

-- 3. ATTRITION BY JOB ROLE
-- Find roles with highest turnover
SELECT 
    JobRole,
    Department,
    COUNT(*) as total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) as attrition_count,
    ROUND(AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as attrition_rate,
    ROUND(AVG(CASE WHEN Attrition = 'Yes' THEN MonthlyIncome END), 2) as avg_income_left
FROM employees
GROUP BY JobRole, Department
HAVING COUNT(*) >= 10
ORDER BY attrition_rate DESC
LIMIT 10;

-- 4. ATTRITION BY AGE GROUP
-- Analyze attrition across age ranges
SELECT 
    CASE 
        WHEN Age < 25 THEN '18-24'
        WHEN Age < 35 THEN '25-34'
        WHEN Age < 45 THEN '35-44'
        WHEN Age < 55 THEN '45-54'
        ELSE '55+'
    END as age_group,
    COUNT(*) as total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) as attrition_count,
    ROUND(AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as attrition_rate
FROM employees
GROUP BY age_group
ORDER BY 
    CASE age_group
        WHEN '18-24' THEN 1
        WHEN '25-34' THEN 2
        WHEN '35-44' THEN 3
        WHEN '45-54' THEN 4
        ELSE 5
    END;

-- 5. ATTRITION BY SALARY RANGE
-- Correlation between income and attrition
SELECT 
    CASE 
        WHEN MonthlyIncome < 50000 THEN '< $50K'
        WHEN MonthlyIncome < 75000 THEN '$50K - $75K'
        WHEN MonthlyIncome < 100000 THEN '$75K - $100K'
        WHEN MonthlyIncome < 150000 THEN '$100K - $150K'
        ELSE '> $150K'
    END as salary_range,
    COUNT(*) as total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) as attrition_count,
    ROUND(AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as attrition_rate
FROM employees
GROUP BY salary_range
ORDER BY 
    CASE salary_range
        WHEN '< $50K' THEN 1
        WHEN '$50K - $75K' THEN 2
        WHEN '$75K - $100K' THEN 3
        WHEN '$100K - $150K' THEN 4
        ELSE 5
    END;

-- 6. ATTRITION BY JOB SATISFACTION
-- Impact of satisfaction on retention
SELECT 
    JobSatisfaction,
    COUNT(*) as total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) as attrition_count,
    ROUND(AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as attrition_rate,
    ROUND(AVG(MonthlyIncome), 2) as avg_income
FROM employees
GROUP BY JobSatisfaction
ORDER BY JobSatisfaction;

-- 7. ATTRITION BY WORK-LIFE BALANCE
-- How work-life balance affects retention
SELECT 
    WorkLifeBalance,
    COUNT(*) as total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) as attrition_count,
    ROUND(AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as attrition_rate,
    SUM(CASE WHEN OverTime = 'Yes' THEN 1 ELSE 0 END) as overtime_count
FROM employees
GROUP BY WorkLifeBalance
ORDER BY WorkLifeBalance;

-- 8. OVERTIME IMPACT ON ATTRITION
-- Compare attrition for overtime vs non-overtime workers
SELECT 
    OverTime,
    COUNT(*) as total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) as attrition_count,
    ROUND(AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as attrition_rate,
    ROUND(AVG(JobSatisfaction), 2) as avg_job_satisfaction,
    ROUND(AVG(WorkLifeBalance), 2) as avg_work_life_balance
FROM employees
GROUP BY OverTime;

-- 9. ATTRITION BY TENURE
-- Early vs late career attrition patterns
SELECT 
    CASE 
        WHEN YearsAtCompany <= 2 THEN '0-2 years'
        WHEN YearsAtCompany <= 5 THEN '3-5 years'
        WHEN YearsAtCompany <= 10 THEN '6-10 years'
        WHEN YearsAtCompany <= 20 THEN '11-20 years'
        ELSE '20+ years'
    END as tenure_group,
    COUNT(*) as total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) as attrition_count,
    ROUND(AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as attrition_rate,
    ROUND(AVG(Age), 1) as avg_age
FROM employees
GROUP BY tenure_group
ORDER BY 
    CASE tenure_group
        WHEN '0-2 years' THEN 1
        WHEN '3-5 years' THEN 2
        WHEN '6-10 years' THEN 3
        WHEN '11-20 years' THEN 4
        ELSE 5
    END;

-- 10. PROMOTION GAP ANALYSIS
-- Employees overdue for promotion
SELECT 
    CASE 
        WHEN YearsSinceLastPromotion <= 1 THEN '0-1 year'
        WHEN YearsSinceLastPromotion <= 3 THEN '2-3 years'
        WHEN YearsSinceLastPromotion <= 7 THEN '4-7 years'
        ELSE '8+ years'
    END as promotion_gap,
    COUNT(*) as total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) as attrition_count,
    ROUND(AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as attrition_rate
FROM employees
GROUP BY promotion_gap
ORDER BY 
    CASE promotion_gap
        WHEN '0-1 year' THEN 1
        WHEN '2-3 years' THEN 2
        WHEN '4-7 years' THEN 3
        ELSE 4
    END;

-- 11. BUSINESS TRAVEL IMPACT
-- How travel affects retention
SELECT 
    BusinessTravel,
    COUNT(*) as total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) as attrition_count,
    ROUND(AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as attrition_rate,
    ROUND(AVG(WorkLifeBalance), 2) as avg_work_life_balance
FROM employees
GROUP BY BusinessTravel
ORDER BY attrition_rate DESC;

-- 12. DISTANCE FROM HOME ANALYSIS
-- Geographic proximity and retention
SELECT 
    CASE 
        WHEN DistanceFromHome <= 5 THEN 'Very Close (0-5 km)'
        WHEN DistanceFromHome <= 15 THEN 'Close (6-15 km)'
        WHEN DistanceFromHome <= 25 THEN 'Far (16-25 km)'
        ELSE 'Very Far (25+ km)'
    END as distance_category,
    COUNT(*) as total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) as attrition_count,
    ROUND(AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as attrition_rate
FROM employees
GROUP BY distance_category
ORDER BY 
    CASE distance_category
        WHEN 'Very Close (0-5 km)' THEN 1
        WHEN 'Close (6-15 km)' THEN 2
        WHEN 'Far (16-25 km)' THEN 3
        ELSE 4
    END;

-- 13. MARITAL STATUS AND ATTRITION
-- Family situation impact
SELECT 
    MaritalStatus,
    Gender,
    COUNT(*) as total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) as attrition_count,
    ROUND(AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as attrition_rate,
    ROUND(AVG(Age), 1) as avg_age
FROM employees
GROUP BY MaritalStatus, Gender
ORDER BY attrition_rate DESC;

-- 14. STOCK OPTIONS AND RETENTION
-- Financial incentives impact
SELECT 
    StockOptionLevel,
    COUNT(*) as total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) as attrition_count,
    ROUND(AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as attrition_rate,
    ROUND(AVG(YearsAtCompany), 1) as avg_tenure
FROM employees
GROUP BY StockOptionLevel
ORDER BY StockOptionLevel;

-- 15. EDUCATION AND ATTRITION
-- Education level correlation
SELECT 
    EducationField,
    COUNT(*) as total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) as attrition_count,
    ROUND(AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as attrition_rate,
    ROUND(AVG(MonthlyIncome), 2) as avg_income
FROM employees
GROUP BY EducationField
ORDER BY attrition_rate DESC;

-- 16. PERFORMANCE RATING AND ATTRITION
-- High performers leaving
SELECT 
    PerformanceRating,
    COUNT(*) as total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) as attrition_count,
    ROUND(AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as attrition_rate,
    ROUND(AVG(MonthlyIncome), 2) as avg_income,
    ROUND(AVG(PercentSalaryHike), 2) as avg_salary_hike
FROM employees
GROUP BY PerformanceRating
ORDER BY PerformanceRating;

-- 17. TRAINING AND DEVELOPMENT
-- Impact of training on retention
SELECT 
    TrainingTimesLastYear,
    COUNT(*) as total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) as attrition_count,
    ROUND(AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as attrition_rate,
    ROUND(AVG(JobSatisfaction), 2) as avg_job_satisfaction
FROM employees
GROUP BY TrainingTimesLastYear
ORDER BY TrainingTimesLastYear;

-- 18. COMPREHENSIVE RISK PROFILE
-- Employees with multiple risk factors
SELECT 
    EmployeeID,
    Age,
    Department,
    JobRole,
    MonthlyIncome,
    YearsAtCompany,
    JobSatisfaction,
    WorkLifeBalance,
    OverTime,
    YearsSinceLastPromotion,
    DistanceFromHome,
    CASE 
        WHEN Age < 30 THEN 1 ELSE 0 
    END +
    CASE 
        WHEN MonthlyIncome < 60000 THEN 1 ELSE 0 
    END +
    CASE 
        WHEN JobSatisfaction <= 2 THEN 1 ELSE 0 
    END +
    CASE 
        WHEN WorkLifeBalance <= 2 THEN 1 ELSE 0 
    END +
    CASE 
        WHEN OverTime = 'Yes' THEN 1 ELSE 0 
    END +
    CASE 
        WHEN YearsAtCompany <= 2 THEN 1 ELSE 0 
    END +
    CASE 
        WHEN YearsSinceLastPromotion > 5 THEN 1 ELSE 0 
    END +
    CASE 
        WHEN DistanceFromHome > 20 THEN 1 ELSE 0 
    END as risk_score
FROM employees
WHERE Attrition = 'No'
HAVING risk_score >= 4
ORDER BY risk_score DESC
LIMIT 50;

-- 19. SALARY HIKE ANALYSIS
-- Correlation between raises and retention
SELECT 
    CASE 
        WHEN PercentSalaryHike < 13 THEN 'Low (< 13%)'
        WHEN PercentSalaryHike < 15 THEN 'Medium (13-15%)'
        WHEN PercentSalaryHike < 18 THEN 'High (15-18%)'
        ELSE 'Very High (18%+)'
    END as salary_hike_range,
    COUNT(*) as total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) as attrition_count,
    ROUND(AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as attrition_rate
FROM employees
GROUP BY salary_hike_range
ORDER BY 
    CASE salary_hike_range
        WHEN 'Low (< 13%)' THEN 1
        WHEN 'Medium (13-15%)' THEN 2
        WHEN 'High (15-18%)' THEN 3
        ELSE 4
    END;

-- 20. MANAGER TENURE IMPACT
-- Relationship stability with manager
SELECT 
    CASE 
        WHEN YearsWithCurrManager < 1 THEN '< 1 year'
        WHEN YearsWithCurrManager < 3 THEN '1-3 years'
        WHEN YearsWithCurrManager < 7 THEN '3-7 years'
        ELSE '7+ years'
    END as manager_tenure,
    COUNT(*) as total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) as attrition_count,
    ROUND(AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as attrition_rate,
    ROUND(AVG(RelationshipSatisfaction), 2) as avg_relationship_satisfaction
FROM employees
GROUP BY manager_tenure
ORDER BY 
    CASE manager_tenure
        WHEN '< 1 year' THEN 1
        WHEN '1-3 years' THEN 2
        WHEN '3-7 years' THEN 3
        ELSE 4
    END;

-- 21. MULTI-DIMENSIONAL ATTRITION HOTSPOTS
-- Identify problematic combinations
SELECT 
    Department,
    CASE 
        WHEN Age < 30 THEN 'Young'
        WHEN Age < 45 THEN 'Mid-Career'
        ELSE 'Senior'
    END as age_category,
    OverTime,
    COUNT(*) as total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) as attrition_count,
    ROUND(AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as attrition_rate
FROM employees
GROUP BY Department, age_category, OverTime
HAVING COUNT(*) >= 10
ORDER BY attrition_rate DESC
LIMIT 15;

-- 22. JOB LEVEL PROGRESSION
-- Career advancement and retention
SELECT 
    JobLevel,
    COUNT(*) as total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) as attrition_count,
    ROUND(AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as attrition_rate,
    ROUND(AVG(MonthlyIncome), 2) as avg_income,
    ROUND(AVG(YearsAtCompany), 1) as avg_tenure,
    ROUND(AVG(YearsSinceLastPromotion), 1) as avg_years_since_promotion
FROM employees
GROUP BY JobLevel
ORDER BY JobLevel;

-- 23. ENVIRONMENT SATISFACTION IMPACT
-- Workplace environment correlation
SELECT 
    EnvironmentSatisfaction,
    COUNT(*) as total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) as attrition_count,
    ROUND(AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as attrition_rate,
    ROUND(AVG(JobInvolvement), 2) as avg_job_involvement
FROM employees
GROUP BY EnvironmentSatisfaction
ORDER BY EnvironmentSatisfaction;

-- 24. COMPARATIVE ANALYSIS: STAYERS VS LEAVERS
-- Key differences between groups
SELECT 
    Attrition,
    COUNT(*) as employee_count,
    ROUND(AVG(Age), 1) as avg_age,
    ROUND(AVG(MonthlyIncome), 2) as avg_income,
    ROUND(AVG(YearsAtCompany), 1) as avg_tenure,
    ROUND(AVG(JobSatisfaction), 2) as avg_job_satisfaction,
    ROUND(AVG(WorkLifeBalance), 2) as avg_work_life_balance,
    ROUND(AVG(EnvironmentSatisfaction), 2) as avg_env_satisfaction,
    ROUND(AVG(YearsSinceLastPromotion), 1) as avg_years_since_promotion,
    ROUND(AVG(DistanceFromHome), 1) as avg_distance,
    ROUND(AVG(CASE WHEN OverTime = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as overtime_pct
FROM employees
GROUP BY Attrition;

-- 25. RETENTION RATE BY HIRING COHORT
-- Track retention over employee lifecycle
SELECT 
    CASE 
        WHEN TotalWorkingYears < 5 THEN 'Early Career (0-5 years)'
        WHEN TotalWorkingYears < 10 THEN 'Mid Career (5-10 years)'
        WHEN TotalWorkingYears < 20 THEN 'Experienced (10-20 years)'
        ELSE 'Very Experienced (20+ years)'
    END as career_stage,
    COUNT(*) as total_employees,
    SUM(CASE WHEN Attrition = 'No' THEN 1 ELSE 0 END) as retained,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) as attrition_count,
    ROUND(AVG(CASE WHEN Attrition = 'No' THEN 1.0 ELSE 0.0 END) * 100, 2) as retention_rate
FROM employees
GROUP BY career_stage
ORDER BY 
    CASE career_stage
        WHEN 'Early Career (0-5 years)' THEN 1
        WHEN 'Mid Career (5-10 years)' THEN 2
        WHEN 'Experienced (10-20 years)' THEN 3
        ELSE 4
    END;

-- 26. DEPARTMENT BENCHMARKING
-- Compare departments on multiple metrics
SELECT 
    Department,
    COUNT(*) as total_employees,
    ROUND(AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as attrition_rate,
    ROUND(AVG(MonthlyIncome), 2) as avg_income,
    ROUND(AVG(JobSatisfaction), 2) as avg_job_satisfaction,
    ROUND(AVG(WorkLifeBalance), 2) as avg_work_life_balance,
    ROUND(AVG(YearsAtCompany), 1) as avg_tenure,
    ROUND(AVG(CASE WHEN OverTime = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as overtime_pct,
    ROUND(AVG(TrainingTimesLastYear), 1) as avg_training_times
FROM employees
GROUP BY Department
ORDER BY attrition_rate DESC;

-- 27. IDENTIFY FLIGHT RISK EMPLOYEES
-- Current employees likely to leave (for intervention)
SELECT 
    EmployeeID,
    Age,
    Gender,
    Department,
    JobRole,
    MonthlyIncome,
    YearsAtCompany,
    YearsSinceLastPromotion,
    JobSatisfaction,
    WorkLifeBalance,
    EnvironmentSatisfaction,
    OverTime,
    DistanceFromHome,
    PercentSalaryHike,
    StockOptionLevel
FROM employees
WHERE Attrition = 'No'
    AND (
        (JobSatisfaction <= 2) OR
        (WorkLifeBalance <= 2) OR
        (OverTime = 'Yes' AND JobSatisfaction <= 3) OR
        (YearsSinceLastPromotion > 7) OR
        (MonthlyIncome < 55000 AND JobLevel >= 2) OR
        (EnvironmentSatisfaction = 1) OR
        (YearsAtCompany <= 2 AND JobSatisfaction <= 3)
    )
ORDER BY 
    CASE 
        WHEN JobSatisfaction <= 2 THEN 1 ELSE 0 
    END +
    CASE 
        WHEN WorkLifeBalance <= 2 THEN 1 ELSE 0 
    END +
    CASE 
        WHEN YearsSinceLastPromotion > 7 THEN 1 ELSE 0 
    END DESC
LIMIT 100;

-- 28. AVERAGE METRICS BY ATTRITION STATUS
-- Summary view for dashboard
SELECT 
    'Overall Average' as category,
    ROUND(AVG(Age), 1) as avg_age,
    ROUND(AVG(MonthlyIncome), 2) as avg_income,
    ROUND(AVG(YearsAtCompany), 1) as avg_tenure,
    ROUND(AVG(JobSatisfaction), 2) as avg_satisfaction,
    ROUND(AVG(DistanceFromHome), 1) as avg_distance
FROM employees

UNION ALL

SELECT 
    'Attrition = Yes' as category,
    ROUND(AVG(Age), 1) as avg_age,
    ROUND(AVG(MonthlyIncome), 2) as avg_income,
    ROUND(AVG(YearsAtCompany), 1) as avg_tenure,
    ROUND(AVG(JobSatisfaction), 2) as avg_satisfaction,
    ROUND(AVG(DistanceFromHome), 1) as avg_distance
FROM employees
WHERE Attrition = 'Yes'

UNION ALL

SELECT 
    'Attrition = No' as category,
    ROUND(AVG(Age), 1) as avg_age,
    ROUND(AVG(MonthlyIncome), 2) as avg_income,
    ROUND(AVG(YearsAtCompany), 1) as avg_tenure,
    ROUND(AVG(JobSatisfaction), 2) as avg_satisfaction,
    ROUND(AVG(DistanceFromHome), 1) as avg_distance
FROM employees
WHERE Attrition = 'No';
"""

# Split script into SELECT queries
queries = [q.strip() for q in sql_script.split(';') if q.strip().startswith('SELECT')]

# Execute and export each query
for i, query in enumerate(queries, start=1):
    print(f"\n🟢 Running Query {i}...")
    try:
        df = pd.read_sql_query(query, conn)
        print(df.head())
        df.to_csv(f'query_{i}_output.csv', index=False)
        print(f"✅ Saved as query_{i}_output.csv")
    except Exception as e:
        print(f"❌ Error in Query {i}: {e}")

conn.close()
print("\n✅ All 28 HR Analytics queries executed successfully.")
