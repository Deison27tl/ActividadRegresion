import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
## Diplomado Python
#- Nombre: Deison Tuiran Londoño
#- Email:  deison.tuiran@upb.edu.co
#- Codigo: 014810
datos = pd.read_excel("AirQualityUCI.xlsx")
x = datos["RH"]
y = datos["T"]

trenX = x[:8000]
trenY = y[:8000]

pruebaX = x[8000:]
pruebaY = y[8000:]

mymodel = np.poly1d(np.polyfit(trenX, trenY, 3))
myline = np.linspace(100, 1000, 8000)
plt.scatter(trenX,trenY)
plt.plot(trenX, mymodel(myline))
plt.show()
r2 = r2_score(trenY, mymodel(trenX))

print(r2)