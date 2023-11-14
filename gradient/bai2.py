# Chi tim duoc x cuc tieu , doi so co gia tri duong
#fx = (1/3)*(x**3) - x
def grad(x):
    return x*x - 1

def cost(x):
    return (1/3)*(x**3) - x


def myGD1(x0 , eta):
    x = [x0]
    for i in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return (x , i)

    
(x1 , i1) = myGD1(-0.01, 0.05)
(x2 , i2) = myGD1(0.01, 0.05)
print('Solution x1 = %f , cost = %f , after = %d iterations'%(x1[-1] , cost(x1[-1] ), i1 ))
print('Solution x2 = %f , cost = %f , after = %d iterations'%(x2[-1] , cost(x2[-1] ), i2 ))