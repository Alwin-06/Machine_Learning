import numpy as np

def gradient(x, y, l=0.00001, it=1000):
    t0 = 0
    t1 = 0

    for i in range(it):
        pred_y = t0 + t1 * x

        g0 = -np.sum(y - pred_y)
        g1 = -np.sum((y - pred_y)*x)

        t0 -= l * g0
        t1 -= l * g1
    
    sse1=np.sum((y-(t0+(t1*x)))**2)
    sst1=np.sum((y-(np.sum(y)/np.size(x)))**2)
    r21=1-(sse1/sst1)

    return t0, t1,sse1,r21

def coefficient(x,y):
    alpha0=((np.sum(x**2)*np.sum(y))-(np.sum(x)*np.sum(x*y)))/((np.size(x)*np.sum(x**2))-(np.sum(x)*np.sum(x)))
    alpha1=(-(np.sum(x)*np.sum(y))+(np.size(x)*np.sum(x*y)))/((np.size(x)*np.sum(x**2))-(np.sum(x)*np.sum(x)))

    sse=np.sum((y-(alpha0+(alpha1*x)))**2)
    sst=np.sum((y-(np.sum(y)/np.size(x)))**2)
    r2=1-(sse/sst)

    return alpha0,alpha1,sse,r2

x = np.array([12, 23, 34, 45, 56, 67, 78, 89, 123, 134])
y = np.array([240, 1135, 2568, 4521, 7865, 9236, 11932, 14589, 19856, 23145])

theta0, theta1,sse1,r21 = gradient(x, y)
print("Gradient Descent Algorithm")
print("The coefficients are: ")
print("theta0 = ", theta0)
print("theta1 = ", theta1)
print("SSE = ",sse1)
print("R2 score = ",r21)

print("")

alpha0,alpha1,sse,r2 = coefficient(x,y)
print("Simple Linear Regression")
print("The coefficients are: ")
print("alpha0 = ", alpha0)
print("alpha1 = ", alpha1)
print("SSE = ",sse)
print("R2 score = ",r2)


