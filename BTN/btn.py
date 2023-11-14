import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score # danh gia chat luong cua mo hinh hoi quy tuyen tinh 
from sklearn.model_selection import train_test_split 
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
data = pd.read_csv('heart_disease.csv')

dt_Train, dt_Test = train_test_split(data,test_size = 0.3, shuffle = False) # chia du lieu dt_train va dt_test voi ti le 70 , 30 cua du lieu goc data , shuffle ko de xa tron du lieu trc khi chai
# X_Test = np.array([[4, 5 ]])
X_Train = dt_Train.iloc[:, :12] #: lay tat ca cac hang , :12 lay tat ca cac cot n-1 (0-13)
Y_Train = dt_Train.iloc[:, 12]
X_Test = dt_Test.iloc[:, :12]
Y_Test = dt_Test.iloc[:, 12]

reg = LinearRegression().fit(X_Train, Y_Train) # huan luyen giua x_train , y_train de tim ra mqh tuyen tinh 


y_pred = reg.predict(X_Test) # du doa gian tri y dua tren x-test bang mo hinh hoi quy reg
y = np.array(Y_Test)

print("Predicted:", y_pred)

print("Coef of determination: %.2f" % r2_score(Y_Test,y_pred))   #đánh giá độ chính xác của tt
# print("Coefficient of determination: %.2f" %r2_score(Y_Test, y_pred))
# print("Thuc te         Du doan            Chenh lech")
# print("--------------------------------------------------")
# for i in range(0,len(y)):
    
#   print("%.2f" %y[i]," ",y_pred[i]," ",abs(y[i]-y_pred[i]))


model2 = Ridge().fit(X_Train, Y_Train) #tạo mô hình Ridge và huấn luyện nó trên dữ liệu huấn luyện
y_pred2= model2.predict(X_Test)
print("Ket qua Ridge",y_pred2[:3]) 
# lay ket qua 3 hang+ dau tien

model3 = Lasso().fit(X_Train, Y_Train)
y_pred3= model3.predict(X_Test)
print("Ket qua Lasso",y_pred3[:3])
# print("Ket qua thuc te",Y_Test[:3])