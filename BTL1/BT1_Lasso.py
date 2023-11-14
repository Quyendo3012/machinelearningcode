import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np
#sklearn.metrics tính toán độ đo 
data = pd.read_csv('winequality-red.csv')# Đọc tập dữ liệu từ file csv và lưu vào biến data.

dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle = False)


X_train = dt_Train.iloc[:, :11] # lấy dữ liệu từ dt_train cho các cột từ 0 đến 10 (11 cột đầu tiên) và gán cho biến x_train.
y_train = dt_Train.iloc[:, 11] #lấy dữ liệu từ dt_train cho cột 11 và gán cho biến y_train.
X_test = dt_Test.iloc[:,  :11] #lấy dữ liệu từ dt_test cho các cột từ 0 đến 10 (11 cột đầu tiên) và gán cho biến x_test.
y_test = dt_Test.iloc[:, 11] # lấy dữ liệu từ dt_test cho cột 11 và gán cho biến y_test.
def NSE(y_test, y_pred):# đánh giá su sai khac giữa giá trị du doan va gia tri thuc te
    return (1 - (np.sum((y_pred - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)))

def MAE(y_test, y_pred): #tính toan do loi trung binh tuyet doi
    return mean_absolute_error(y_test, y_pred)

def RMSE(y_test, y_pred):  #tính toán độ lỗi trung bình bình phương.
    return mean_squared_error(y_test, y_pred, squared=False)

model = Lasso().fit(X_train, y_train) # khoi tao mo hinh lasso va huan luyen mo hinh tren du lieu x_train va y_train
y_pred = model.predict(X_test) # dự đoán kết quả dau ra dua tren trên x_test  bang mo hinh da duoc huan luyen
print("Hoi quy lasso")
print("Ket qua Lasso",y_pred[:3]) #n ra các thông tin về kết quả dự đoán và giá trị thực tế cho 3 mẫu dữ liệu đầu tiên.
print("Ket qua thuc te",y_test[:3])
y=np.array(y_test) # chuyen doi y_test thanh mang numpy va gan vao bien y
print("Thuc te   Du doan   Chenh lech")
for i in range (0, len(y)): # in các giá trị thực tế, dự đoán và chênh lệch tương ứng cho tất cả các mẫu dữ liệu.
    print("%.2f" %y[i], "  ", y_pred[i], "  ", abs(y[i]-y_pred[i]))
print('NSE:', NSE(y, y_pred)) #tìm sự sai khác giữa dự đoán và thực tế càng gần 1 mô hình càng tốt
print ('R2 :%.2f' %  r2_score(y_test,y_pred)) #tìm sự sai khác giữa dự đoán và thực tế càng gần 1 mô hình càng tốt
print('MAE:', MAE(y, y_pred)) # càng nhỏ càng tốt
print('RMSE:', RMSE(y, y_pred))# càng nhỏ càng tốt