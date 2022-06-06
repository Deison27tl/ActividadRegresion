import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

datos = pd.read_excel("AirQualityUCI.xlsx")
x = datos["RH"]
y = datos["T"]

x,y = np.array(x).reshape(-1,1), np.array(y)

trenX = x[:8000]
trenY = y[:8000]

pruebaX = x[8000:]
pruebaY = y[8000:]

model = LinearRegression().fit(trenX,trenY)
r_sq_train = model.score(trenX,trenY)
r_sq_test = model.score(pruebaX,pruebaY)
y_predict = model.predict(pruebaX)
plt.scatter(pruebaX,pruebaY)
plt.plot(pruebaX, y_predict)
plt.show()
print("score for train", r_sq_train)
print("Score for test", r_sq_test)
