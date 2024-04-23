import pandas as pd

df = pd.read_csv('dataset_processed.csv')
# df[['Systolic_BP','Diastolic_BP']] = df['Blood Pressure'].str.split('/',expand=True).astype(int)

# df = df.drop(['Patient ID','Country','Continent','Hemisphere','Income','Blood Pressure'],axis=1)
# df.to_csv('dataset_processed.csv',index=False)



from sklearn.preprocessing import LabelEncoder

# Khởi tạo LabelEncoder
le = LabelEncoder()

df['Sex'] = le.fit_transform(df['Sex'])
df['Diet'] = le.fit_transform(df['Diet']) # 0-Average 1-Healthy 2-Unhealthy

# Lưu dataframe đã được xử lý vào file csv
df.to_csv('dataset_processed.csv', index=False)
