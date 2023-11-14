import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np

#Y là nhãn
# chia dl làm 2 phần với công thức f(x) = W*XT để tìm ra W
#Tập 1 là train dùng để timd ra quy luật tìm W
# Tập 2 là test độc lập với train dùng để đánh giá mô hình 

data = pd.read_csv('winequality-red.csv')
#train luôn nhiều hơn test, shuffle dùng để xáo trộn dl
dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle = False)
#đọc dl và tách ra thành data_train, test mỗi 1 dòng là 1 dl, mỗi cột là thuộc tính mô tả dl

X_train = dt_Train.iloc[:, :11] #iloc: đọc dl từ 9 cột đầu => lấy từ cột 0 đến cột 8
y_train = dt_Train.iloc[:, 11]#cột cuối là y train
X_test = dt_Test.iloc[:,  :11]
y_test = dt_Test.iloc[:, 11]

# x_data=data.iloc[:, :9]
#y_data = data.iloc[:, 9]
def NSE(y_test, y_pred):
    return (1 - (np.sum((y_pred - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)))

def MAE(y_test, y_pred):
    return mean_absolute_error(y_test, y_pred)

def RMSE(y_test, y_pred):
    return mean_squared_error(y_test, y_pred, squared=False)
def MAX_ERROR(y_test, y_pred):
    max = 0
    for i in range(0, len(y_test)):
        if (max < abs(y_test[i] - y_pred[i])):
            max = abs(y_test[i] - y_pred[i])
    return max

reg = LinearRegression().fit(X_train, y_train) #fit dùng để huấn luyện mô hình HQTT dựa trên x train y train, sau khi huấn luyện tìm ra dc w
print('w=', reg.coef_) # coef_ (w1, w2, w3) hệ số tương ứng của kiểu dl
print('w0=', reg.intercept_) #intercept_W0 hệ số tự do
y_pred = reg.predict(X_test) # predict : dự đoán mẫu mới trả giá trị dữ đoán trên tập 
# thay x test vào bộ w để tính dự đoán y_pred

y=np.array(y_test) # ytest là gtr thuc te cua X_test
print("Thuc te   Du doan    Chenh lech")
for i in range (0, len(y)):
    print("%.2f" %y[i], "  ", y_pred[i], "  ", abs(y[i]-y_pred[i]))
print('NSE:', NSE(y, y_pred)) #tìm sự dai khác giữa dự đoán và thực tế càng gần 1 mô hình càng tốt
print ('Coef. :%.2f' %  r2_score(y_test,y_pred)) #tìm sự dai khác giữa dự đoán và thực tế càng gần 1 mô hình càng tốt
print('MAE:', MAE(y, y_pred)) # càng nhỏ càng tốt
print('RMSE:', RMSE(y, y_pred))# càng nhỏ càng tốt
print('MAX_ERROR:', MAX_ERROR(y, y_pred))

# print('x: \n', x_data)
# print('y: \n',y_data)
# print('Tập dữ liệu tập train: \n', dt_Train)
# print('Tập dữ liệu test: \n', dt_Test)