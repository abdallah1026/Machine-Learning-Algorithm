import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from model import LinearRegression
from RidgeRegression import RidgeRegression
from LassoRegression import LassoRegression

X, y = datasets.make_regression(n_samples=1000, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

reg = LinearRegression(lr=0.01)
rig = RidgeRegression(lr=0.001, alpha=20)
lag = LassoRegression(lr=0.001, alpha=20)

reg.fit(X_train, y_train)
rig.fit(X_train, y_train)
lag.fit(X_train, y_train)

y_pred = reg.predict(X_test)
ridge_prediction = rig.predict(X_test)
lasso_prediction = lag.predict(X_test)


def mes(y_test, y_pred):
    return np.mean(pow(y_test - y_pred, 2))


mes_L = mes(y_test, y_pred)
mes_R = mes(y_test, ridge_prediction)
mes_i = mes(y_test, lasso_prediction)

print(f"MSE_L ===> {mes_L}")
print(f"MES_R ===> {mes_R}")
print(f"MES_I ===> {mes_i}")


y_pred_line = reg.predict(X)

cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(15, 6))
plt.scatter(X_train, y_train, cmap=cmap(0.9), s=10, label="Training data")
plt.scatter(X_test, y_test, cmap=cmap(0.5), s=10, label="Test data")
plt.plot(X, y_pred_line, label="Fitted line", linewidth=3, color="black")
plt.legend()
plt.show()
