#sử dụng numpy cho ĐSTT và matplottlip cho vẽ hình
from __future__ import division,print_function,unicode_literals
import numpy as np 
import  matplotlib.pyplot as plt
#chiều cao (cm)
X = np.array([[147,150,153,158,163,165,168,170,173,175,178,180,183]]).T
#cân nặng (kg)
y = np.array([[49,50,51,54,58,59,60,72,63,64,66,67,68]]).T
#visualize data
plt.plot(X,y,'ro')
plt.axis([140,190,45,75])
plt.xlabel('Chiều cao ( cm)')
plt.ylabel('Cân nặng (kg)')
plt.show()
# hàm numpy.ones() trả về một mảng có hình dạng và kiểu dữ liệu nhất định trong đó giá trị của phần tử được đặt là 1
one = np.ones((X.shape[0],1))
#np.concatenate  ghép nối mảng theo một trục cụ thể
Xbar = np.concatenate((one,X),axis = 1)

#tính toán trọng lượng đường vừa lắp

A = np.dot(Xbar.T,Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A),b)
print("w= ",w)
w_0 = w[0][0]
w_1 = w[1][0]
#trả về các số cách đều nhau trong một khoảng nhất định
x0 = np.linspace(145,185,2)
y0 = w_0 +w_1*x0

# vẽ đường phù hợp
plt.plot(X.T,y.T,'ro') #dữ liệu
plt.plot(x0,y0) # dòng phù hợp
plt.axis([140,190,45,75])
plt.xlabel('Chiều cao (cm )')
plt.ylabel('Cân nặng (kg)')
plt.show()

y1 = w_1*155 + w_0
y2 = w_1*160 +w_0
print(u'Dự đoán cân nặng của người cao 155cm : %.2f (kg), cân nặng thực là: 52 (kg)' %(y1))
print(u'Dự đoán cân nặng của người cao 160cm : %.2f (kg), cân nặng thực là: 56 (kg)' %(y2))