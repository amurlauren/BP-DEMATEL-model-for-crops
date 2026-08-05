# BP-DEMATEL-model-for-crops
Identify driving factors and characteristic factors influencing crop water‑footprint based on the BP‑neural‑network and decision‑making trial and evaluation laboratory (BP‑DEMATEL) model.
基于BP-DEMATEL模型的黑龙江省大豆水足迹影响因素
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

# 读取数据，从Excel文件中读取
data = pd.read_excel('crops.xlsx')

# 步骤1: 自动处理缺失值
def fill_missing_values(df):
    for column in df.columns:
        if df[column].dtype in [np.float64, np.int64]:  # 数值列
            df[column] = df[column].fillna(df[column].mean())
        elif df[column].dtype == 'object':  # 分类列
            df[column] = df[column].fillna(df[column].mode()[0])  # 使用众数填补
    return df

data = fill_missing_values(data)

# 分离特征列和目标变量
X_features = data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']]
y = data['WF']

# 使用60%训练集，20%验证集，20%测试集划分数据
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# 确保数据集划分后的大小满足模型训练需求
print(f"训练集大小: {X_train.shape[0]}")
print(f"验证集大小: {X_val.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# 步骤2: 正向化处理仅对特征列进行
b_min = X_features.min().min()
b_max = X_features.max().max()
X_train_scaled = (X_train - b_min) / (b_max - b_min)
X_val_scaled = (X_val - b_min) / (b_max - b_min)
X_test_scaled = (X_test - b_min) / (b_max - b_min)

# 打印正向化后的特征矩阵
print("\n正向化后的训练集特征矩阵:")
print(X_train_scaled.values)

print("\n正向化后的验证集特征矩阵:")
print(X_val_scaled.values)

print("\n正向化后的测试集特征矩阵:")
print(X_test_scaled.values)

# 步骤3: 构建BP神经网络
bp_network = MLPRegressor(hidden_layer_sizes=(8,), activation='logistic',
                          solver='adam', learning_rate_init=0.01,
                          max_iter=1000, tol=1e-5, random_state=42)

# 步骤4: 训练模型
bp_network.fit(X_train_scaled, y_train)
# 验证和测试模型（输出得分只是为了确认训练有效性）
train_score = bp_network.score(X_train_scaled, y_train)
val_score = bp_network.score(X_val_scaled, y_val)
test_score = bp_network.score(X_test_scaled, y_test)
# 打印结果
print(f"训练集得分: {train_score}")
print(f"验证集得分: {val_score}")
print(f"测试集得分: {test_score}")

# 步骤5: 获取训练后的权值矩阵
W1 = bp_network.coefs_[0]  # 输入层到隐含层的权值矩阵
W2 = bp_network.coefs_[1]  # 隐含层到输出层的权值矩阵

# 打印权值矩阵
print("\n输入层到隐含层的权值矩阵W1:")
print(W1)
print("\n隐含层到输出层的权值矩阵W2:")
print(W2)

# 步骤6: 计算综合权值矩阵
W = (np.abs(W1) + np.abs(W2.T)) / 2
print("\n综合权值矩阵W:")
print(W)

# 步骤7: 计算直接影响矩阵
impact_matrix = np.zeros_like(W)
n_features = X_train_scaled.shape[1]
for i in range(n_features):
    for j in range(n_features):
        if i == j:
            impact_matrix[i, j] = 0
        else:
            if W[j, :].sum() != 0:
                impact_matrix[i, j] = W[i, :].sum() / W[j, :].sum()
            else:
                impact_matrix[i, j] = 0
# 打印直接影响矩阵
print("\n直接影响矩阵impact_matrix:")
print(impact_matrix)

# 步骤8: 计算综合影响矩阵
I = np.eye(n_features)  # 创建一个形状为 n_features x n_features 的单位矩阵
# 计算 Y 时，确保其形状与 I 一致
Y = np.zeros_like(impact_matrix)
for i in range(n_features):
    if np.max(impact_matrix.sum(axis=1)) != 0:
        Y[i, :] = impact_matrix[i, :] / np.max(impact_matrix.sum(axis=1))
T = np.linalg.inv(I - Y[:, :n_features]) - I
print("\n综合影响矩阵T:")
print(T)

# 步骤9: 计算影响度、被影响度、中心度和原因度
D = T.sum(axis=1)  # 影响度
R = T.sum(axis=0)  # 被影响度
C = D + R         # 中心度
Q = D - R         # 原因度

# 打印结果
print("\n影响度D:")
print(D)
print("\n被影响度R:")
print(R)
print("\n中心度C:")
print(C)
print("\n原因度Q:")
print(Q)
