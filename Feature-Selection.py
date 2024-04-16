from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# df = pd.read_csv('dataset_train_encoded.csv')
# # # Tách cột 'Blood Pressure' thành hai cột mới 'Systolic_BP' và 'Diastolic_BP'
# df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(float)
# df.drop(['Patient ID','Income','Country','Continent','Hemisphere','Medication Use','Family History','Obesity','Diabetes', 'Diet_Average','Diet_Healthy','Diet_Unhealthy','Sex_Female','Sex_Male', 'Alcohol Consumption','Smoking','Blood Pressure'], axis=1).to_csv('dataset_train_processed.csv', index=False)
# # Loại bỏ cột 'Blood Pressure' cũ
X = pd.read_csv('dataset_train.csv')
y = pd.read_csv('label_train.csv')

# Khởi tạo mô hình Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X, y.values.ravel())

# Lấy điểm số quan trọng của đặc trưng
importances = model.feature_importances_

# In ra điểm số quan trọng của từng đặc trưng
feature_importances = pd.Series(importances, index=X.columns)
print(feature_importances.sort_values(ascending=False))


