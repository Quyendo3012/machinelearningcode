import numpy as np
from sklearn.linear_model import LinearRegression


# Y = m.X + b
X_train = np.array([[147,150,153,155,158,160,163,165,168,170,173,175,178,180,183]]).T
y_train = np.array([49,50,51,52,54,56,58,59,60,72,63,64,66,67,68]).T
reg = LinearRegression().fit(X_train, y_train)
#lay ve gia tri tham so 
w1 = reg.coef_
w0 = reg.intercept_
print('w1 = ', w1) #Hệ số m
print('w0 = ',w0) # b

X_test = np.array([[162]])
f = reg.predict(X_test)
print('f(162) = ', f)

print("\nTest: ")
y1 = w1*155 + w0
print('f(155) = ', y1)

y2 = w1*160 + w0
print('f(160) = ', y2)