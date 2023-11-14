import sympy as sp 
from scipy.optimize import fsolve
import numpy as np 

#khai bao bien 
x = sp.symbols('x')

#define ham 
f = x**2 + 5*(sp.sin(x)) + 2 

#Dao ham 
df = sp.diff(f,x)

#Chuyen doi bieuthuc sympy thanh ham so hoc sd lambdify
df_numeric = sp.lambdify(x, df)

#tim gia tri x thoa man pt df = 0 
solutions = fsolve(df_numeric, x0=0)

print("Đạo hàm của hàm f(x) là: ", df)
print("Các giá trị x thỏa mãn: ", solutions)

