import pandas as pd

# # Đọc dữ liệu từ file CSV
# df_train = pd.read_csv('dataset_train.csv')
# df_test = pd.read_csv('dataset_test.csv')

# # # Áp dụng One-Hot Encoding cho các đặc trưng phân loại
# # df_encoded = pd.get_dummies(df, columns=['Sex', 'Diet'])

# # df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(float)

# # # Loại bỏ cột 'Blood Pressure' cũ
# # df_train = df_train.drop(['Diet_Unhealthy'], axis=1)
# # df_test = df_test.drop(['Diet_Unhealthy'], axis=1)
# # Lưu DataFrame đã mã hóa vào một file CSV mới
# # df_encoded.to_csv('dataset_encoded.csv', index=False)

# # df_train.to_csv('dataset_train.csv', index=False)
# # df_test.to_csv('dataset_test.csv', index=False)


from sklearn.preprocessing import LabelEncoder

# Đọc dữ liệu từ file CSV
df = pd.read_csv('heart_attack_prediction_dataset.csv')

# Khởi tạo LabelEncoder
le = LabelEncoder()
df_encode = df.drop(['Patient ID','Income','Country','Continent','Hemisphere'], axis=1)
# Áp dụng Label Encoding cho mỗi cột phân loại
df_encode['Sex'] = le.fit_transform(df['Sex'])
df_encode['Diet'] = le.fit_transform(df['Diet'])

# Lưu DataFrame đã mã hóa vào một file CSV mới
df_encode.to_csv('dataset_encoded.csv', index=False)

