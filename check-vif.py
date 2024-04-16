import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Đọc dữ liệu từ file CSV
X = pd.read_csv('dataset_train_processed.csv')

# Thêm một cột hằng số vào X
X['Intercept'] = 1

# Tính VIF cho mỗi đặc trưng
vif = pd.DataFrame()
vif['Feature'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif)
