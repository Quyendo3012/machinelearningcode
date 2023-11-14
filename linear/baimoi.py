import numpy as np 
from sklearn import linear_model
import  matplotlib.pyplot as plt
height = np.array([[147,150,153,155,158,160,163,165,168,170,173,175,178,180,183]]).T
weight = np.array([49,50,51,52,54,56,58,59,60,72,63,64,66,67,68]).T


# #trực quan hóa 
# plt.xlabel("Chiều cao (cm)")
# plt.ylabel("Cân nặng (kg)")
# plt.scatter(height, weight, color = 'red')

regr = linear_model.LinearRegression()

#huấn luyện mô hình qua hàm fit 
regr.fit(height, weight)

plt.xlabel("Chiều cao (cm)")
plt.ylabel("Cân nặng (kg)")
plt.scatter(height, weight, color = 'red')
plt.plot(height, regr.predict(height), color = 'blue')
plt.show()

#lay ve gia tri cac tham so 
w1 = regr.coef_
w0 = regr.intercept_
print('w1 = ', w1) #Hệ số m
print('w0 = ',w0) # b
 
test = np.array([[162]])
f = regr.predict(test)
print('f(162) = ', f)

print("\nTest: ")
y1 = w1*155 + w0
print('f(155) = ', y1)

y2 = w1*160 + w0
print('f(160) = ', y2)



