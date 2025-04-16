import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def coefficient(x,y,x1,y1):
    alpha0=((np.sum(x**2)*np.sum(y))-(np.sum(x)*np.sum(x*y)))/((np.size(x)*np.sum(x**2))-(np.sum(x)*np.sum(x)))
    alpha1=(-(np.sum(x)*np.sum(y))+(np.size(x)*np.sum(x*y)))/((np.size(x)*np.sum(x**2))-(np.sum(x)*np.sum(x)))

    sse=np.sum((y1-(alpha0+(alpha1*x1)))**2)
    sst=np.sum((y1-(np.sum(y1)/np.size(x1)))**2)
    r2=1-(sse/sst)

    return [alpha0,alpha1,sse,r2]

x = np.array([12, 23, 34, 45, 56, 67, 78, 89, 123, 134])
y = np.array([240, 1135, 2568, 4521, 7865, 9236, 11932, 14589, 19856, 23145])

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)

a=coefficient(xtrain,ytrain,xtest,ytest)

print("Regression Coffecients: ",a[0],a[1])
print("SSE: ",a[2])
print("R2: ",a[3])

plt.scatter(x, y, label="Data")
plt.plot(x,(a[0]+(a[1]*x)) ,color='red', label="Linear Regression")
plt.legend()
plt.show()






'''
import numpy as np
import matplotlib.pyplot as plt

def coefficient(x,y):
    alpha0=((np.sum(x**2)*np.sum(y))-(np.sum(x)*np.sum(x*y)))/((np.size(x)*np.sum(x**2))-(np.sum(x)*np.sum(x)))
    alpha1=(-(np.sum(x)*np.sum(y))+(np.size(x)*np.sum(x*y)))/((np.size(x)*np.sum(x**2))-(np.sum(x)*np.sum(x)))

    sse=np.sum((y-(alpha0+(alpha1*x)))**2)
    sst=np.sum((y-(np.sum(y)/np.size(x)))**2)
    r2=1-(sse/sst)

    return [alpha0,alpha1,sse,r2]

x = np.array([12, 23, 34, 45, 56, 67, 78, 89, 123, 134])
y = np.array([240, 1135, 2568, 4521, 7865, 9236, 11932, 14589, 19856, 23145])

a=coefficient(x,y)

print("Regression Coffecients: ",a[0],a[1])
print("SSE: ",a[2])
print("R2: ",a[3])
'''




'''
sum_x=np.sum(x)
sum_y=np.sum(y)
sq_x=np.sum(x**2)
sq_y=np.sum(y**2)
xy=np.sum(x.flatten()*y)
l=np.size(x)

print(sum_x,sum_y,sq_x,sq_y,xy,l)

a0=((sq_x*sum_y)-(sum_x*xy))/((l*sq_x)-(sum_x*sum_x))
print(a0)
'''
'''
print(y)
y_pred=alpha0+(alpha1*x)
print(y_pred)
y1=y-y_pred.flatten()
print(y1)
ss=np.sum(y1**2)
print(ss)

mean=sum_y/l
y2=y-mean
st=np.sum(y2**2)
r=1-(ss/st)
print(r)
'''