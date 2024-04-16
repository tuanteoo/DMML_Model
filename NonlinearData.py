import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

df = pd.read_csv('dataset_train_processed.csv')
dflabel = pd.read_csv('label_train.csv')

# Danh sách các features
features = ['Age', 'Cholesterol', 'Heart Rate', 
            'Exercise Hours Per Week', 'Previous Heart Problems', 'Stress Level',
            'Sedentary Hours Per Day', 'BMI', 'Triglycerides', 'Physical Activity Days Per Week',
            'Sleep Hours Per Day', 'Systolic_BP', 'Diastolic_BP']

# Biến target
target = 'Heart Attack Risk'

# Xây dựng mô hình tuyến tính cho từng feature
for feature in features:
    model = LinearRegression()
    model.fit(df[[feature]], dflabel[target])

    # Dự đoán giá trị và tính phần dư
    predictions = model.predict(df[[feature]])
    residuals = dflabel[target] - predictions

    # Biểu đồ hóa dữ liệu
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(df[feature], dflabel[target])
    plt.plot(df[feature], predictions, color='red')
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.title(f'Scatter plot of {feature} vs {target}')

    # Biểu đồ hóa phần dư
    plt.subplot(1, 2, 2)
    plt.scatter(df[feature], residuals)
    plt.xlabel(feature)
    plt.ylabel('Residuals')
    plt.title(f'Scatter plot of {feature} vs residuals')
    plt.show()

    # Kiểm tra phần dư
    if np.mean(residuals) == 0:
        print(f"Dữ liệu {feature} có thể có tính chất tuyến tính.")
        
    else:
        print(f"Dữ liệu {feature} có thể có tính chất phi tuyến.")
        
        

