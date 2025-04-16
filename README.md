# Machine_Learning

## Regression & Classification Tasks

###  Question 1: Regression Models using Python

Given the dataset:

| x   | y     |
|-----|-------|
| 12  | 240   |
| 23  | 1135  |
| 34  | 2568  |
| 45  | 4521  |
| 56  | 7865  |
| 67  | 9236  |
| 78  | 11932 |
| 89  | 14589 |
| 123 | 19856 |
| 134 | 23145 |

**Tasks:**
- [ ] (a) Fit the following models using Python's built-in functions:
  - Linear Regression
  - Polynomial Regression (degree 2)
  - Polynomial Regression (degree 3)
- [ ] (b) Compare the models using:
  - Sum of Squared Errors (SSE)
  - Coefficient of Determination (R²)

---

###  Question 2: Manual Regression Calculations

**Tasks:**
- [ ] (a) Calculate regression coefficients using the formula:  
  \[
  α = (X^T X)^{-1} X^T Y
  \]
- [ ] (b) Compute:
  - SSE:
    \[
    SSE = \sum_{i=1}^{m} (y_i - \hat{y}_i)^2
    \]
  - \( R^2 = 1 - \frac{SSE}{SST} \)

---

###  Question 3: Diabetes Classification (Logistic Regression)

Download the **Pima Indians Diabetes Dataset** from Kaggle.

**Tasks:**
- [ ] (a) Build a logistic regression model to classify patients as:
  - WITH DIABETES
  - WITHOUT DIABETES
- [ ] (b) Evaluate the model with different data splits:
  - 0% training, 20% testing
  - 70% training, 30% testing
  - 60% training, 40% testing

---

###  Question 4: Position-Salaries Dataset – Bias-Variance

Download the Position-Salaries dataset.

**Tasks:**
- [ ] Fit models:
  - Linear Regression
  - Polynomial Regression (degrees 2, 5, and 11)
- [ ] Use `mixtend` to calculate:
  - Bias
  - Variance
- [ ] Analyze:
  - Underfitting
  - Overfitting

---

###  Question 5: Linear Regression with Gradient Descent

**Tasks:**
- [ ] Implement linear regression using **Gradient Descent**:
  - Write a function that returns regression coefficients.
  - You may hardcode initial parameter values.
- [ ] Compare the parameters with those from **Question 1**.

---
