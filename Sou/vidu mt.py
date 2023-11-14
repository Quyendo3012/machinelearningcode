import numpy as np
# Dinh nghia mot ma tran
A = np.array([[ 1, 3, 4],
              [-2, 6, 0], 
              [-5, 7, 2]])

B = np.array([[ 2, 3, 4], 
              [-1,-2,-3], 
              [0, 4, -4]])
#Cong tru ma tran
print("A + B = \n", A + B)
print("A - B = \n", A - B)
#Nhan chia ma tran voi mot so
print("A * 3 = \n", A * 3)
print("A  / 4 = \n", A / 4)
#Nhan 2 ma tran
print("A * B = \n", A @ B)
print("B * A = \n", B @ A)
print("B * A = \n", B.dot(A))

print("A = \n", A )
print("Ma tran chuyen vi cua A \n", A.T ) #np.transpose(A)
print("Ma tran nghich dao cua A \n", np.linalg.inv(A))
print("Ma tran gia nghich dao cua A \n", np.linalg.pinv(A))

arr = np.eye(3)
print("Ma tran don vi \n",arr)

arr1 = np.diag([1, 2, 3])
print("Ma tran duong cheo \n",arr1)
print("Lay duong cheo ma tran A \n",np.diag(A))

Afile = np.loadtxt("./A.txt")
print("Afile = \n" , Afile)
# tìm hiểu trích cột
