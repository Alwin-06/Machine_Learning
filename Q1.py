import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split

x = np.array([12, 23, 34, 45, 56, 67, 78, 89, 123, 134]).reshape(-1, 1)
y = np.array([240, 1135, 2568, 4521, 7865, 9236, 11932, 14589, 19856, 23145])

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)

lm=LinearRegression()
lm.fit(xtrain,ytrain)
predicted_y=lm.predict(xtest)
l_mse=mean_squared_error(ytest,predicted_y)
l_sse=l_mse*len(ytest)
l_r2=r2_score(ytest,predicted_y)

poly2=PolynomialFeatures(degree=2)
x_poly2=poly2.fit_transform(xtrain)
x_test=poly2.transform(xtest)
poly2_model=LinearRegression()
poly2_model.fit(x_poly2,ytrain)
y_pred_poly2=poly2_model.predict(x_test)
mse_poly2=mean_squared_error(ytest,y_pred_poly2)
sse_poly2=mse_poly2*len(ytest)
poly2_r2=r2_score(ytest,y_pred_poly2)

poly3=PolynomialFeatures(degree=3)
x_poly3=poly3.fit_transform(xtrain)
x_test1=poly3.transform(xtest)
poly3_model=LinearRegression()
poly3_model.fit(x_poly3,ytrain)
y_pred_poly3=poly3_model.predict(x_test1)
mse_poly3=mean_squared_error(ytest,y_pred_poly3)
sse_poly3=mse_poly3*len(ytest)
poly3_r2=r2_score(ytest,y_pred_poly3)


print("Linear Regression SSE: ", l_sse)
print("Linear Regression R2: ", l_r2)
plt.scatter(x, y, label="Data")
plt.plot(x, lm.predict(x),color='red', label="Linear Regression")
plt.legend()
plt.show()

print("Polynomial Regression (Degree 2) SSE: ", sse_poly2)
print("Polynomial Regression (Degree 2) R2: ", poly2_r2)
plt.scatter(x, y, label="Data")
plt.plot(x, poly2_model.predict(poly2.fit_transform(x)),color='green', label="Polynomial Regression (Degree 2)")
plt.legend()  
plt.show()

print("Polynomial Regression (Degree 3) SSE: ", sse_poly3)
print("Polynomial Regression (Degree 3) R2: ", poly3_r2) 
plt.scatter(x, y, label="Data")
plt.plot(x,poly3_model.predict(poly3.fit_transform(x)),color='blue', label="Polynomial Regression (Degree 3)")
plt.legend()
plt.show()


















'''
lm=LinearRegression()
lm.fit(x,y)
predicted_y=lm.predict(x)
l_mse=mean_squared_error(y,predicted_y)
l_sse=l_mse*len(y)
l_r2=r2_score(y,predicted_y)

poly2=PolynomialFeatures(degree=2)
x_poly2=poly2.fit_transform(x)
poly2_model=LinearRegression()
poly2_model.fit(x_poly2,y)
y_pred_poly2=poly2_model.predict(x_poly2)
mse_poly2=mean_squared_error(y,y_pred_poly2)
sse_poly2=mse_poly2*len(y)
poly2_r2=r2_score(y,y_pred_poly2)

poly3=PolynomialFeatures(degree=3)
x_poly3=poly3.fit_transform(x)
poly3_model=LinearRegression()
poly3_model.fit(x_poly3,y)
y_pred_poly3=poly3_model.predict(x_poly3)
mse_poly3=mean_squared_error(y,y_pred_poly3)
sse_poly3=mse_poly3*len(y)
poly3_r2=r2_score(y,y_pred_poly3)


print("Linear Regression SSE: ", l_sse)
print("Linear Regression R2: ", l_r2)

print("Polynomial Regression (Degree 2) SSE: ", sse_poly2)
print("Polynomial Regression (Degree 2) R2: ", poly2_r2)

print("Polynomial Regression (Degree 3) SSE: ", sse_poly3)
print("Polynomial Regression (Degree 3) R2: ", poly3_r2)


plt.scatter(x, y, label="Data")
plt.plot(x, predicted_y,color='red', label="Linear Regression")
plt.plot(x, y_pred_poly2,color='green', label="Polynomial Regression (Degree 2)")
plt.plot(x, y_pred_poly3,color='blue', label="Polynomial Regression (Degree 3)")
plt.legend()
plt.show()
'''