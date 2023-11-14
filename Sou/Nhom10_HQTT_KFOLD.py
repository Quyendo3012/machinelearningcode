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
data = pd.read_csv('D:/Code/MachineLearn/data.csv')
#tập train nhiều dl hơn tập test
dt_Train, dt_Test = train_test_split(data, test_size=0.3 , shuffle = False) #30% dl làm tập test, cần xáo lộn dl dùng false
# Implementing cross validation
k = 5#mottaplavalidate,4 latrain,chiatapdatatrsai thanh5
kf = KFold(n_splits=k, random_state=None) #chia tập dl thành 5p
# tinh error, y thuc te, y_pred: dl du doan
def error(y,y_pred):
    sum=0
    for i in range(0,len(y)):
        sum = sum + abs(y[i] - y_pred[i])
    return sum/len(y) 
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
min=999999
for train_index, validation_index in kf.split(dt_Train): #tach ra 5 cap (traint index{B,C,D,E},validation index{A}...)
    X_train = dt_Train.iloc[train_index,:9]
    X_validation = dt_Train.iloc[validation_index, :9]
    y_train, y_validation = dt_Train.iloc[train_index, 9], dt_Train.iloc[validation_index, 9]

    lr = LinearRegression()
    lr.fit(X_train, y_train) # huấn luyện
    y_train_pred = lr.predict(X_train) #dự đoán nhãn trên tập train
    y_validation_pred = lr.predict(X_validation) #dự đoán nhãn trên tập validation
    y_train = np.array(y_train)
    y_validation = np.array(y_validation)
    
    sum_error = error(y_train,y_train_pred)+error(y_validation, y_validation_pred) #so sánh giá trị dự đoán và gtri thưc tế
    if(sum_error < min):
        min = sum_error
        regr=lr
y_test_pred=regr.predict(dt_Test.iloc[:,:9])#sududoancuamohinhtotnhatlaytrenvonglapfor
y_test=np.array(dt_Test.iloc[:,9])
print("Thuc te     Du doan         Chenh lech")
for i in range(0,len(y_test)):
    print(y_test[i]," ",y_test_pred[i], " " , abs(y_test[i]-y_test_pred[i]))
    
print("NSE: ", NSE(y_test,y_test_pred))
print ('Coef. :%.2f' %  r2_score(y_test,y_test_pred)) 
print('MAE:', MAE(y_test,y_test_pred))
print('RMSE:', RMSE(y_test,y_test_pred))
print('MAX_ERROR:', MAX_ERROR(y_test,y_test_pred))