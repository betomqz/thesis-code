import numpy as np
from scipy.optimize import rosen, rosen_der
from optimus import int_point_qp, ls_sqp

# Test for int_point_qp() -----------------------------------------------------
G = np.array([[2,0],[0,2]])
c = np.array([-2,-5])
A = np.array([
    [ 1, -2],
    [-1, -1],
    [-1,  2],
    [ 1,  0],
    [ 0,  1]
])
b = np.array([-2,-6,-2,0,0])

x0 = np.array([2.,0.])
print("\n#########################################")
print("##### Running test for int_point_qp #####")
print("#########################################\n")
x, y, lam = int_point_qp(G=G, c=c, A=A, b=b, x_0=x0.copy(), sigma=0.1,
                         tol=10e-10)

# Test for ls_sqp() -----------------------------------------------------------
def fun(x):
    return rosen(x), rosen_der(x)

def restr(x):
    # We'll use the same restrictions for this method. Why not.
    return np.dot(A, x) - b, A

x0 = np.array([2.,0.])

print("\n###################################")
print("##### Running test for ls_sqp #####")
print("###################################\n")
ls_sqp(fun, restr, x_0=x0, lam_0=np.ones(5), B_0=np.eye(x0.size), eta=0.4,
       tau=0.7, maxiters=1000, tol=10e-10)