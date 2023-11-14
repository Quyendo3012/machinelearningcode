import pandas as pd
#các thư viện tính độ đo để đánh giá
from sklearn.metrics import r2_score 
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split  #chia dữ liệu thành 2 phần
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np

#y_pred ~ y_test => mô hình tốt
#X_test là mẫu, y_test là nhãn
data = pd.read_csv('winequality-red.csv')
#tập train nhiều dl hơn tập test
dt_Train, dt_Test = train_test_split(data, test_size = 0.3 , shuffle = False) #30% dl làm tập test
k = 3 #mot tap la validate, 2 la train,chi tap data  thanh 3 
kf = KFold(n_splits=k, random_state=None) #chia tập dl thành 3p, (random_state = none điều khiển sự ngẫu nhiên trong quá trình tách data)
# tinh error, y thuc te, y_pred: dl du doan

#Với mỗi cặp tập con (train,validation) ta xây dựng một mô hình Linear Regression để dự đoán giá trị của nhãn. Sau đó tính tổng lỗi trên 
# tập train và tập validation và tìm mô hình có tổng lỗi nhỏ nhất để sử dụng cho dự đoán trên tập test.
def error(y,y_pred): #tính trung bình lỗi 
    sum=0
    for i in range(0,len(y)):
        sum = sum + abs(y[i] - y_pred[i])
    return sum/len(y) 
def NSE(y_test, y_pred): #đánh giá sự sai khác giữa các gtri dự doán và gtri thuwcjc tế 
    return (1 - (np.sum((y_pred - y_test) * 2) / np.sum((y_test - np.mean(y_test)) * 2)))
def MAE(y_test, y_pred):# tính toán độ lỗi trung bình tuyệt đối 
    return mean_absolute_error(y_test, y_pred)

def RMSE(y_test, y_pred): #tính toán độ lỗi tb bình phương 
    return mean_squared_error(y_test, y_pred, squared=False)

min=999999
for train_index, validation_index in kf.split(dt_Train): #tach ra 3 cap {A,B,C}
    #1 phần là val thì 2 phần là tập train
    #{A} là val thì train sẽ là {C,B}
    #{B} là val thì train sẽ là {A,C}
    #print ('train ', train_index)
    #print('val', validation_index)
    X_train = dt_Train.iloc[train_index,:11]
    X_validation = dt_Train.iloc[validation_index, :11]
    y_train, y_validation = dt_Train.iloc[train_index, 11], dt_Train.iloc[validation_index, 11]

    lr = LinearRegression() #Khởi tạo mô hình LinearRegression
    lr.fit(X_train, y_train) # huấn luyện
    y_train_pred = lr.predict(X_train) #dự đoán nhãn trên tập train
    y_validation_pred = lr.predict(X_validation) #dự đoán nhãn trên tập validation
    y_train = np.array(y_train) #ép kiểu từ frame về mảng
    y_validation = np.array(y_validation) # ép kiểu từ frame về mảng
    
    sum_error = error(y_train,y_train_pred)+error(y_validation, y_validation_pred) #Tính tổng lỗi => càng nhỏ càng tốt
    if(sum_error < min): #tìm min để đưa ra mô hình tốt nhất
        min = sum_error
        regr=lr #tìm ra mô hình tốt nhất
y_test_pred=regr.predict(dt_Test.iloc[:,:11])#su du doan cua mo hinh tot nhat de dua ra tap du doan tu tap test 
y_test=np.array(dt_Test.iloc[:,11])
# print("Thuc te     Du doan         Chenh lech")
# for i in range(0,len(y_test)):
#     print(y_test[i]," ",y_test_pred[i], " " , abs(y_test[i]-y_test_pred[i]))
    
print("NSE: ", NSE(y_test,y_test_pred))  #tìm sự sai khác giữa dự đoán và thực tế càng gần 1 mô hình càng tốt
print ('R2: %.2f' %  r2_score(y_test,y_test_pred)) #tìm sự sai khác giữa dự đoán và thực tế càng gần 1 mô hình càng tốt
print('MAE:', MAE(y_test,y_test_pred)) # càng nhỏ càng tốt
print('RMSE:', RMSE(y_test,y_test_pred)) # càng nhỏ càng tốt