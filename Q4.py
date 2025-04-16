import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,r2_score
from mlxtend.evaluate import bias_variance_decomp

def linear(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

    lm=LinearRegression()
    lm.fit(x_train,y_train)
    predicted_y=lm.predict(x_test)
    l_sse=np.sum((y_test-predicted_y)**2)

    loss,bias1, variance1 = bias_variance_decomp(lm, x_train, y_train,x_test, y_test,loss='mse', random_seed=42)
    
    print("Linear Regression")
    print("SSE(test): ", l_sse)
    print("Bias: ",bias1)
    print("Variance: ",variance1)
    print(" ")
    plt.scatter(x, y, label="Data")
    plt.plot(x,lm.predict(x),color='red', label="Linear Regression")
    plt.legend()
    plt.show()

def polynomial(x,y,z):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

    poly=PolynomialFeatures(degree=z)
    x_poly=poly.fit_transform(x_train)
    xtest=poly.transform(x_test)

    poly_model=LinearRegression()
    poly_model.fit(x_poly,y_train)

    y_pred_poly=poly_model.predict(xtest)
    sse_poly=np.sum((y_test - y_pred_poly)**2)

    loss1,bias2, variance2 = bias_variance_decomp(poly_model, x_poly, y_train, xtest, y_test, loss='mse', random_seed=42)
    
    x1=poly.transform(x)
    y1=poly_model.predict(x1)

    print("Polynomial Regression (Degree{z}")
    print("SSE(train): ", sse_poly)
    print("Bias: ",bias2)
    print("Variance: ",variance2) 
    print(" ")
    plt.scatter(x, y, label="Data")
    plt.plot(x ,y1,color='red', label=f"Polynomial Regression")
    plt.legend()
    plt.show()



data = pd.read_csv("Position_Salaries.csv")

x = data['Level'].values
y = data['Salary'].values
x=x.reshape(-1,1)

linear(x,y)
polynomial(x,y,2)
polynomial(x,y,5)
polynomial(x,y,11)

