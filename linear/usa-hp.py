import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('USA_Housing.csv')
df.head()

print(df.columns) 

X = df [['TB_ThuNhapKhuVuc ', 'TB_tuoinha  ', 'TB_dientich ', 'TB_sophong ',
       'Dansokhuvuc  ']]
y = df ['Gia']

X_data = np.array(X)
y_data = np.array(y)

reg = LinearRegression().fit(X_data, y_data)

#lay ve gia tri cac tham so 
w = reg.coef_
w0 = reg.intercept_
print('w = ', w) #Hệ số m
print('w0 = ',w0) # b

# print("Coefficients (w):", reg.coef_)  
# print("Intercept (w0):", reg.intercept_) 

X_test = np.array([[78394.33928,6.989779748,6.620477995,2.42,36516.35897]])
predicted_price = reg.predict(X_test)

print("Predicted Price:", predicted_price)

