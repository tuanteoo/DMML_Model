import pandas as pd

# Đọc dữ liệu từ file CSV
df_train = pd.read_csv('dataset_train.csv')
df_test = pd.read_csv('dataset_test.csv')

# # Áp dụng One-Hot Encoding cho các đặc trưng phân loại
# df_encoded = pd.get_dummies(df, columns=['Sex', 'Diet'])

# df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(float)

# # Loại bỏ cột 'Blood Pressure' cũ
df_train = df_train.drop(['Diet_Unhealthy'], axis=1)
df_test = df_test.drop(['Diet_Unhealthy'], axis=1)
# Lưu DataFrame đã mã hóa vào một file CSV mới
# df_encoded.to_csv('dataset_encoded.csv', index=False)

df_train.to_csv('dataset_train.csv', index=False)
df_test.to_csv('dataset_test.csv', index=False)


