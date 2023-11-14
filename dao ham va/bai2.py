import numpy as np 

# khai bao matr
A = np.array ([
    [1,4,-1],
    [2,0,1]
])

B = np.array ([
    [-1,0],
    [1,3],
    [-1,1]
])


#cau a 
print("Ma tran chuyen vi cua B \n", B.T ) #np.transpose(B)
print("A+B^T = \n", A + B.T)
print("A-B^T= \n", A - B.T)

#cau b
print("A * 2 = \n", A * 2) #nhan matran vs 1 so 
print("A * B = \n", A @ B) #nhan 2 matran 
#print("A * B = \n", A.dot(B))

#cau c 
print("Ma tran gia nghich dao cua A \n", np.linalg.pinv(A))
print("A * A^-1 = \n", A @ np.linalg.pinv(A))
