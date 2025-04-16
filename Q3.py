import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

data = pd.read_csv("diabetes.csv")

x = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = data['Outcome']
print(x)

def logistic(x_train,x_test,y_train,y_test):
    model=LogisticRegression(max_iter=200)
    model.fit(x_train,y_train)
    predicted_y=model.predict(x_test)
    a=accuracy_score(y_test,predicted_y)
    cm=confusion_matrix(y_test,predicted_y)

    print("Accuracy:",a)
    print("Confusion Matrix:\n", cm)
    print(" ")

x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, test_size=0.2, random_state=42)
print("Training with 80% and Testing with 20%:")
logistic(x_train1, x_test1, y_train1, y_test1)

x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, test_size=0.3, random_state=42)
print("Training with 70% and Testing with 30%:")
logistic(x_train2, x_test2, y_train2, y_test2)

x_train3, x_test3, y_train3, y_test3 = train_test_split(x, y, test_size=0.4, random_state=42)
print("Training with 60% and Testing with 40%:")
logistic(x_train3, x_test3, y_train3, y_test3)














'''
cm=confusion_matrix(y_test,predicted_y)
r=classification_report(y_test,predicted_y)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n",r)
'''