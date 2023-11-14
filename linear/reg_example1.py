import numpy as np

X_train = np.array([[60, 2, 10], [40, 2, 5], [100, 3, 7]]).T
y_train = np.array([10, 12, 20]).T
x_test = np.array([[50, 2, 8]]).T

w = np.linalg.pinv(X_train@X_train.T)@X_train@y_train
y_test_hat = x_test.T@w
print('Giá trị dự đoán mẫu mới: ', y_test_hat)
print('Giá trị dự đoán tập huấn luyện: ', X_train.T@w)


