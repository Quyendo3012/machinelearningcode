import pandas as pd 
from sklearn.metrics import r2_score #đánh giá chất lượng mô hình hồi quy
from sklearn.linear_model import LinearRegression # thực hiện hồi quy tuyến tính 
from sklearn.model_selection import train_test_split #tạp ra file train và file test từ dữ liệu có sẵn
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np

#Y là nhãn
# chia dl làm 2 phần với công thức f(x) = W*XT để tìm ra W
#Tập 1 là train dùng để timd ra quy luật tìm W
# Tập 2 là test độc lập với train dùng để đánh giá mô hình 


#y_pred ~ y_test => mô hình tốt
#X_test là mẫu, y_test là nhãn

data = pd.read_csv('winequality-red.csv')

dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle = False)
#đọc dl và tách ra thành data_train, test mỗi 1 dòng là 1 dl, mỗi cột là thuộc tính mô tả dl
# chia du lieu dt_train va dt_test voi ti le 70 , 30 cua du lieu goc data , shuffle = false ko de xa tron du lieu trc khi chạy

X_train = dt_Train.iloc[:, :11] # lấy từ cột 0 đến cột 10
y_train = dt_Train.iloc[:, 11] #lấy cột cuối
X_test = dt_Test.iloc[:,  :11]
y_test = dt_Test.iloc[:, 11]

# x_data=data.iloc[:, :11]
# y_data = data.iloc[:, 11]
# print(x_data)
# print(y_data)

# print(X_train)
# print(X_test)

def NSE(y_test, y_pred): 
    return (1 - (np.sum((y_pred - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)))

def MAE(y_test, y_pred):
    return mean_absolute_error(y_test, y_pred)

def RMSE(y_test, y_pred):
    return mean_squared_error(y_test, y_pred, squared=False)


reg = LinearRegression().fit(X_train, y_train) #fit dùng để huấn luyện mô hình HQTT dựa trên x train y train, sau khi huấn luyện tìm ra dc w
print('w=', reg.coef_) # coef_ (w1, w2, w3) hệ số tương ứng của kiểu dl
print('w0=', reg.intercept_) #intercept_W0 hệ số tự do
y_pred = reg.predict(X_test) # # du doa gia tri y dua tren x-test bang mo hinh hoi quy reg 
# thay x test vào bộ w để tính dự đoán y_pred

reg = LinearRegression().fit(X_train, y_train) # huan luyen giua x_train , y_train de tim ra mqh tuyen tinh 


y_pred = reg.predict(X_test) # du doa gian tri y dua tren x-test bang mo hinh hoi quy reg
y = np.array(y_test)

print("Predicted:", y_pred)

y=np.array(y_test) # ytest là gtr thuc te cua X_test
print("Thuc te   Du doan    Chenh lech")
for i in range (0, len(y)):
    print("%.2f" %y[i], "  ", y_pred[i], "  ", abs(y[i]-y_pred[i]))
print('NSE:', NSE(y, y_pred)) #tìm sự sai khác giữa dự đoán và thực tế càng gần 1 mô hình càng tốt
print ('R2:%.2f' %  r2_score(y_test,y_pred)) #tìm sự sai khác giữa dự đoán và thực tế càng gần 1 mô hình càng tốt
print('MAE:', MAE(y, y_pred)) # càng nhỏ càng tốt
print('RMSE:', RMSE(y, y_pred))# càng nhỏ càng tốt