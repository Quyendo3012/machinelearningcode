import numpy as np

X_train = np.array([[147],[150],[153],[155],[158],[160],[163],[165],[168],[170],[173],[175],[178],[180],[183]]).T
y_train = np.array([49,50,51,52,54,56,58,59,60,72,63,64,66,67,68]).T
x1_test = np.array([155]).T
x2_test = np.array([160]).T

w = np.linalg.pinv(X_train@X_train.T)@X_train@y_train
y1_test_hat = x1_test.T@w
y2_test_hat = x2_test.T@w
print('Giá trị dự đoán cân nặng mới 155: ', y1_test_hat)
print('Giá trị dự đoán tập huấn luyện: ', X_train.T@w)
print('Giá trị dự đoán cân nặng mới 160: ', y2_test_hat)


