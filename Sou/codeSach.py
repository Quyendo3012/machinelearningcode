import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
dataframe = pd.read_csv('bitcoin.csv') #Đường dẫn file csv
dt_train,dt_test = train_test_split(dataframe,test_size=0.3,shuffle=True)
x_train = dataframe.drop('gia',axis='columns')
y_train = dataframe.gia
x_test = dt_test.iloc[:,:2]
y_test = dt_test.iloc[:,2]
reg = LinearRegression().fit(x_train,y_train)
y_predict = reg.predict(x_test)
y = np.array(y_test)

print("Coef of determination: %.2f" % r2_score(y_test,y_predict))   #đánh giá độ chính xác của tt
# for i in range(len(y_test)):
#      print("Chenh lech gia nha so %.2f" % y[i] ," ",abs(y_predict[i] - y[i])) #do chenh lech giua du doan va thuc te
#Ridge 
clf = Ridge(alpha=1.0,max_iter=1000,tol=0.01).fit(x_train,y_train)
y_pred = clf.predict(x_test)
print(r2_score(y_test,y_pred))
#Lasso
lasso = Lasso(alpha=1.0,max_iter=1000,tol=0.01).fit(x_train,y_train)
y_pred1 = lasso.predict(x_test)
print(r2_score(y_pred1,y_test))