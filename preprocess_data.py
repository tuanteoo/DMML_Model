import pandas as pd

df = pd.read_csv('heart_statlog_cleveland_hungary_final.csv')


duplicates = df[df.duplicated(subset=['age','sex','chest pain type','resting bp s','cholesterol','fasting blood sugar','resting ecg','max heart rate','exercise angina','oldpeak','ST slope'])]
print(f"Number of duplicate rows: {duplicates}")
duplicates.to_csv('duplicate_data.csv')



