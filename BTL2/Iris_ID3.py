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
dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle = True) #cắt data thành 30% test và 70% train,

X_train = dt_Train.drop(['Id','Species' ], axis = 1)  #bỏ cột id với species lấy dữ liệu ở các cột còn lại
y_train = dt_Train['Species'] # lấy cột species
X_test= dt_Test.drop(['Id','Species' ], axis = 1)#bỏ cột id với species lấy dữ liệu ở các cột còn lại
y_test= dt_Test['Species'] # lấy cột species
y_test = np.array(y_test)

#Xây dựng cây ( dùng độ đo Entropy: độ không chắc chắn , max_depth : độ sâu tối đa ,min_samples_leaf số lượng mẫu tối thiểu để phân nhánh
tree = DecisionTreeClassifier(criterion = 'entropy', min_samples_split=2, min_samples_leaf=7,max_depth = 4).fit(X_train, y_train)
#min sample split số lượng mẫu tối thiểu để tách 1 nút trong cây qđ 

y_pred = tree.predict(X_test) #đưa ra giá trị dự đoán X_Test

# print("Thực tế \t Dự đoán")
# for i in range (0, len(y_test)):
#     print("%.5s" % y_test[i], "\t\t", y_pred[i])
count = 0
for i in range(0,len(y_test)): #tìm số lần dự đoán chính xác trong các dự đoán
    if(y_test[i] == y_pred[i]):# nếu i test  bằng y dự đoán 
        count = count + 1 #nếu dự đoán chính xác thì cộng 1

print('Tỷ lệ dự đoán đúng:', np.around(count/len(y_pred)*100,10), "%")
print("Độ đo Precision:",precision_score(y_test, y_pred, average ='macro')) # Precision (độ chuẩn xác) tỉ lệ số điểm Positive (dự đoán đúng) / tổng số điểm mô hình dự đoán là Positive
#càng cao càng tốt tức là tất cả số điểm đều đúng 
print("Độ đo Recall:",recall_score(y_test, y_pred, average ='micro')) #Recall tỉ lệ số điểm Positive mô hình dự đoán đúng trên tổng số điểm được gán nhãn là Positive ban đầu
#Recall càng cao, tức là số điểm là positive bị bỏ sót càng ít
#micro tbc của precision và recall theo các lớp.
print("Độ đo F1:", f1_score(y_test, y_pred, average='macro'))
mae = mean_absolute_error(y_test, y_pred) 
print('Tỉ lệ dự đoán sai: %f' % mae)