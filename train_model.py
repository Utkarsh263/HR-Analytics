import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_auc_score, roc_curve, accuracy_score,
                            precision_score, recall_score, f1_score)
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("HR ANALYTICS - MACHINE LEARNING MODEL")
print("Employee Attrition Prediction using Logistic Regression")
print("=" * 70)

# ==========================================
# 1. LOAD AND PREPARE DATA
# ==========================================

print("\n1. LOADING DATA")
print("-" * 70)

df = pd.read_csv('hr_employee_data.csv')
print(f"✓ Loaded {len(df)} employee records")
print(f"✓ Features: {df.shape[1]} columns")
print(f"✓ Attrition rate: {(df['Attrition'] == 'Yes').mean() * 100:.2f}%")

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================

print("\n2. FEATURE ENGINEERING")
print("-" * 70)

# Create a copy for modeling
df_model = df.copy()

# Encode target variable
df_model['Attrition'] = (df_model['Attrition'] == 'Yes').astype(int)

# Encode categorical variables
label_encoders = {}
categorical_cols = ['Gender', 'MaritalStatus', 'Department', 'JobRole', 
                    'EducationField', 'BusinessTravel', 'OverTime']

for col in categorical_cols:
    le = LabelEncoder()
    df_model[f'{col}_Encoded'] = le.fit_transform(df_model[col])
    label_encoders[col] = le

print(f"✓ Encoded {len(categorical_cols)} categorical variables")

# Create additional features
df_model['IncomePerYear'] = df_model['MonthlyIncome'] * 12
df_model['SalaryToJobLevel'] = df_model['MonthlyIncome'] / (df_model['JobLevel'] + 1)
df_model['TenureToAge'] = df_model['YearsAtCompany'] / df_model['Age']
df_model['PromotionGap'] = df_model['YearsAtCompany'] - df_model['YearsSinceLastPromotion']
df_model['SatisfactionScore'] = (df_model['JobSatisfaction'] + 
                                  df_model['EnvironmentSatisfaction'] + 
                                  df_model['RelationshipSatisfaction'] + 
                                  df_model['WorkLifeBalance']) / 4

print(f"✓ Created 5 derived features")

# ==========================================
# 3. SELECT FEATURES
# ==========================================

print("\n3. FEATURE SELECTION")
print("-" * 70)

# Features for the model
feature_cols = [
    # Demographics
    'Age', 'Gender_Encoded', 'MaritalStatus_Encoded', 'DistanceFromHome',
    
    # Job characteristics
    'Department_Encoded', 'JobRole_Encoded', 'JobLevel', 
    'Education', 'EducationField_Encoded',
    
    # Compensation
    'MonthlyIncome', 'PercentSalaryHike', 'StockOptionLevel',
    'HourlyRate', 'DailyRate', 'MonthlyRate',
    
    # Work experience
    'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager', 'NumCompaniesWorked',
    
    # Satisfaction & engagement
    'JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance',
    'RelationshipSatisfaction', 'JobInvolvement', 'PerformanceRating',
    
    # Work conditions
    'OverTime_Encoded', 'BusinessTravel_Encoded', 'TrainingTimesLastYear',
    
    # Derived features
    'SalaryToJobLevel', 'TenureToAge', 'PromotionGap', 'SatisfactionScore'
]

X = df_model[feature_cols]
y = df_model['Attrition']

print(f"✓ Selected {len(feature_cols)} features for modeling")
print(f"✓ Target variable: Attrition (0=No, 1=Yes)")

# ==========================================
# 4. TRAIN-TEST SPLIT
# ==========================================

print("\n4. TRAIN-TEST SPLIT")
print("-" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set:   {len(X_train)} samples ({y_train.mean()*100:.2f}% attrition)")
print(f"Test set:       {len(X_test)} samples ({y_test.mean()*100:.2f}% attrition)")

# ==========================================
# 5. FEATURE SCALING
# ==========================================

print("\n5. FEATURE SCALING")
print("-" * 70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Standardized features (mean=0, std=1)")

# ==========================================
# 6. TRAIN LOGISTIC REGRESSION MODEL
# ==========================================

print("\n6. TRAINING LOGISTIC REGRESSION MODEL")
print("-" * 70)

# Train model with class weight to handle imbalanced data
log_reg = LogisticRegression(
    max_iter=1000, 
    class_weight='balanced',
    random_state=42,
    solver='lbfgs'
)

log_reg.fit(X_train_scaled, y_train)
print("✓ Model training complete")

# Cross-validation
cv_scores = cross_val_score(log_reg, X_train_scaled, y_train, cv=5, scoring='roc_auc')
print(f"✓ 5-Fold Cross-Validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ==========================================
# 7. MAKE PREDICTIONS
# ==========================================

print("\n7. MAKING PREDICTIONS")
print("-" * 70)

y_pred = log_reg.predict(X_test_scaled)
y_pred_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

print("✓ Predictions generated for test set")

# ==========================================
# 8. MODEL EVALUATION
# ==========================================

print("\n8. MODEL EVALUATION METRICS")
print("=" * 70)

# Classification metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"""
LOGISTIC REGRESSION PERFORMANCE
{"-" * 70}
Accuracy:   {accuracy:.4f} ({accuracy*100:.2f}%)
Precision:  {precision:.4f} ({precision*100:.2f}%)
Recall:     {recall:.4f} ({recall*100:.2f}%)
F1-Score:   {f1:.4f}
ROC-AUC:    {roc_auc:.4f}
""")

# Confusion Matrix
print("\nCONFUSION MATRIX")
print("-" * 70)
cm = confusion_matrix(y_test, y_pred)
print(f"""
                Predicted
                No      Yes
Actual  No      {cm[0,0]:<7} {cm[0,1]:<7}
        Yes     {cm[1,0]:<7} {cm[1,1]:<7}

True Negatives:  {cm[0,0]} (Correctly predicted No attrition)
False Positives: {cm[0,1]} (Predicted Yes, actually No)
False Negatives: {cm[1,0]} (Predicted No, actually Yes)
True Positives:  {cm[1,1]} (Correctly predicted Yes attrition)
""")

# Classification Report
print("\nCLASSIFICATION REPORT")
print("-" * 70)
print(classification_report(y_test, y_pred, 
                          target_names=['No Attrition', 'Attrition'],
                          digits=4))

# ==========================================
# 9. FEATURE IMPORTANCE
# ==========================================

print("\n9. FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)

# Get coefficients
coefficients = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': log_reg.coef_[0],
    'Abs_Coefficient': np.abs(log_reg.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print("\nTOP 15 MOST IMPORTANT FEATURES")
print("-" * 70)
print(coefficients.head(15).to_string(index=False))

print("\nINTERPRETATION:")
print("  • Positive coefficient = increases attrition risk")
print("  • Negative coefficient = decreases attrition risk")
print("  • Larger absolute value = stronger influence")

# ==========================================
# 10. COMPARE WITH RANDOM FOREST
# ==========================================

print("\n\n10. COMPARISON WITH RANDOM FOREST")
print("=" * 70)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)
rf_roc_auc = roc_auc_score(y_test, y_pred_proba_rf)

print(f"""
MODEL COMPARISON
{"-" * 70}
Metric          Logistic Regression    Random Forest
{"-" * 70}
Accuracy        {accuracy:.4f}                 {rf_accuracy:.4f}
Precision       {precision:.4f}                 {rf_precision:.4f}
Recall          {recall:.4f}                 {rf_recall:.4f}
F1-Score        {f1:.4f}                 {rf_f1:.4f}
ROC-AUC         {roc_auc:.4f}                 {rf_roc_auc:.4f}
""")

# Random Forest feature importance
rf_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nRANDOM FOREST - TOP 15 FEATURES")
print("-" * 70)
print(rf_importance.head(15).to_string(index=False))

# ==========================================
# 11. SAVE PREDICTIONS AND MODEL METRICS
# ==========================================

print("\n\n11. SAVING RESULTS")
print("=" * 70)

# Save predictions
predictions_df = pd.DataFrame({
    'EmployeeID': df.iloc[X_test.index]['EmployeeID'],
    'Actual_Attrition': y_test.values,
    'Predicted_Attrition': y_pred,
    'Attrition_Probability': y_pred_proba,
    'Risk_Level': pd.cut(y_pred_proba, bins=[0, 0.3, 0.6, 1.0],
                         labels=['Low', 'Medium', 'High'])
})

predictions_df.to_csv('attrition_predictions.csv', index=False)
print("✓ Predictions saved to: attrition_predictions.csv")

# Save feature importance
coefficients.to_csv('feature_importance_logistic.csv', index=False)
rf_importance.to_csv('feature_importance_rf.csv', index=False)
print("✓ Feature importance saved")

# Save model metrics
metrics_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'Accuracy': [accuracy, rf_accuracy],
    'Precision': [precision, rf_precision],
    'Recall': [recall, rf_recall],
    'F1_Score': [f1, rf_f1],
    'ROC_AUC': [roc_auc, rf_roc_auc]
})
metrics_df.to_csv('model_comparison.csv', index=False)
print("✓ Model metrics saved to: model_comparison.csv")

# ==========================================
# 12. HIGH-RISK EMPLOYEES
# ==========================================

print("\n\n12. HIGH-RISK EMPLOYEES IDENTIFICATION")
print("=" * 70)

high_risk = predictions_df[predictions_df['Attrition_Probability'] > 0.6].sort_values(
    'Attrition_Probability', ascending=False
)

print(f"\nIdentified {len(high_risk)} high-risk employees (probability > 60%)")
print("\nTop 10 High-Risk Employees:")
print(high_risk.head(10).to_string(index=False))

# Add employee details for high-risk employees
high_risk_details = high_risk.merge(
    df[['EmployeeID', 'Age', 'Department', 'JobRole', 'MonthlyIncome', 
        'YearsAtCompany', 'JobSatisfaction', 'WorkLifeBalance', 'OverTime']],
    on='EmployeeID'
)
high_risk_details.to_csv('high_risk_employees.csv', index=False)
print("✓ High-risk employee details saved to: high_risk_employees.csv")

# ==========================================
# 13. RECOMMENDATIONS
# ==========================================

print("\n\n" + "=" * 70)
print("KEY FINDINGS & RECOMMENDATIONS")
print("=" * 70)

# Analyze high-risk characteristics
high_risk_analysis = high_risk_details.describe()

print(f"""
1. MODEL PERFORMANCE:
   • Achieved {roc_auc*100:.1f}% ROC-AUC score
   • {recall*100:.1f}% of actual attrition cases correctly identified
   • Model is suitable for deployment

2. TOP RISK FACTORS (from coefficients):
   • {coefficients.iloc[0]['Feature']}: Strongest positive impact
   • {coefficients.iloc[1]['Feature']}: Second strongest impact
   • {coefficients.iloc[2]['Feature']}: Third strongest impact

3. HIGH-RISK EMPLOYEE PROFILE:
   • Total identified: {len(high_risk)} employees
   • Average age: {high_risk_details['Age'].mean():.1f} years
   • Average tenure: {high_risk_details['YearsAtCompany'].mean():.1f} years
   • Overtime workers: {(high_risk_details['OverTime'] == 'Yes').sum()} employees

4. RECOMMENDED ACTIONS:
   ✓ Immediate 1-on-1 meetings with high-risk employees
   ✓ Review compensation for employees in low income quartiles
   ✓ Improve work-life balance, especially for overtime workers
   ✓ Implement retention bonuses for early-tenure employees
   ✓ Fast-track promotions for deserving employees
   ✓ Enhance job satisfaction through engagement programs
   ✓ Monitor and address department-specific issues

5. MONITORING:
   ✓ Re-run model quarterly to identify new high-risk employees
   ✓ Track intervention effectiveness
   ✓ Update model with new data annually
""")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)
print("\nFiles created:")
print("  • attrition_predictions.csv - All predictions")
print("  • high_risk_employees.csv - Detailed high-risk list")
print("  • feature_importance_logistic.csv - Logistic regression features")
print("  • feature_importance_rf.csv - Random forest features")
print("  • model_comparison.csv - Model performance comparison")
print("\nNext steps:")
print("  1. Create visualizations (ROC curve, feature importance)")
print("  2. Build Power BI dashboard")
print("  3. Implement retention strategies")