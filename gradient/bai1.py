import numpy as np
#fx = x*x - 2

def grad(x): #tinh dao ham 
    return 2*x

def cost(x): #ham goc 
    return x**2 - 2


def myGD1(x0 , eta): #thuc hien thuat toan Gradient 
    x = [x0] # khoi tao list rong 
    for i in range(100): #
        x = x[-1] - eta*grad(x[-1]) # x0 gtri ban dau de tim min, eta toc do hoc (buoc)
        if abs(grad(x)) < 1e-3: #neu < 1e -3 thi dung lai 
            break
        # x.append(x_new)
    return (x , i) # sau i vong lap thi tra ve x 

    
(x1 , i1) = myGD1(-5, .1) #ket qua tra ve cua ham GD
(x2 , i2) = myGD1(5, .1)
print('Solution x1 = %f , cost = %f , after = %d iterations'%(x1[-1] , cost(x1[-1] ), i1 ))
print('Solution x1 = %f , cost = %f , after = %d iterations'%(x2[-1] , cost(x2[-1] ), i2 ))