from __future__ import print_function 
import numpy as np 
import pandas as pd
from sklearn.metrics import precision_score,mean_absolute_error, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 

data= pd.read_csv('Iris.csv')
le=preprocessing.LabelEncoder() #Chuyển đổi các nhãn thành dữ liệu số nguyên
data=data.apply(le.fit_transform) 

dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle = True) #cắt data thành 30% test và 70% train, ko ngẫu nhiên

X_train = dt_Train.drop(['Id','Species' ], axis = 1)  #bỏ cột id với species lấy dữ liệu ở các cột còn lại
y_train = dt_Train['Species'] # lấy cột species
X_test= dt_Test.drop(['Id','Species' ], axis = 1)#bỏ cột id với species lấy dữ liệu ở các cột còn lại
y_test= dt_Test['Species'] # lấy cột species
y_test = np.array(y_test)
#max_depth độ sâu tối đa khi cây phát triển đến độ sâu này thì kh sâu hơn nữa, min sample leaf là số mẫu tối thiểu để phân chia tiếp 
#gini là độ đo độ tinh khiết
tree = DecisionTreeClassifier(criterion = 'gini',max_depth = 4 , min_samples_leaf = 5) # ham danh gia do do tot nhat

tree.fit(X_train, y_train) # truyen vao tap train de huan luyen mo hinh 

y_pred = tree.predict(X_test) # du doan nhan cua tap test 

count = 0
for i in range(0,len(y_test)):
    if(y_test[i] == y_pred[i]):
        count = count + 1

print('Accuracy:', np.around(count/len(y_pred)*100,10), "%")
print("Độ đo Precision:",precision_score(y_test, y_pred, average ='macro')) # Precision (độ chuẩn xác) tỉ lệ số điểm Positive (dự đoán đúng) / tổng số điểm mô hình dự đoán là Positive
#càng cao càng tốt tức là tất cả số điểm đều đúng 
print("Độ đo Recall:",recall_score(y_test, y_pred, average ='micro')) #Recall tỉ lệ số điểm Positive mô hình dự đoán đúng trên tổng số điểm được gán nhãn là Positive ban đầu
#Recall càng cao, tức là số điểm là positive bị bỏ sót càng ít
#micro tbc của precision và recall theo các lớp.
print("Độ đo F1:", f1_score(y_test, y_pred, average='macro'))
mae = mean_absolute_error(y_test, y_pred) 
print('Tỉ lệ dự đoán sai: %f' % mae)
