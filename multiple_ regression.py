import numpy as np
import pandas as pd
from sklearn import linear_model


## Diplomado Python
#- Nombre: Deison Tuiran Londoño
#- Email:  deison.tuiran@upb.edu.co
#- Codigo: 014810

datos = pd.read_csv("cars.csv")
condicionList = [
    (datos["Car"] == "Mitsubishi"),
    (datos["Car"] == "Ford"),
    (datos["Car"] == "Audi"),
    (datos["Car"] == "Honda"),
    (datos["Car"] == "Hundai"),
    (datos["Car"] == "Opel"),
    (datos["Car"] == "BMW"),
    (datos["Car"] == "Skoda"),
    (datos["Car"] == "Fiat"),
    (datos["Car"] == "Hyundai"),
    (datos["Car"] == "Suzuki"),
    (datos["Car"] == "Volvo"),
    (datos["Car"] == "Mazda"),
    (datos["Car"] == "Toyoty"),
    (datos["Car"] == "Mini"),
    (datos["Car"] == "VW"),
    (datos["Car"] == "Mercedes"),
   
]
eleccionLista = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
datos['marcaCarroNormalizado'] = np.select(condicionList, eleccionLista, default="not_specified")
new_df = pd.DataFrame()
new_df["marca"] = datos["Car"].drop_duplicates()
new_df["marcaCarroNormalizado"] = eleccionLista
x = datos[['Volume', 'Weight', 'CO2']]
y = datos["marcaCarroNormalizado"]
x,y = np.array(x), np.array(y)
regr = linear_model.LinearRegression().fit(x,y)
print(regr.coef_)
y_predicted = regr.predict([[1000, 790, 99]])
marcaCarro=int(np.round(y_predicted,decimals = 0))
nombre = new_df[new_df["marcaCarroNormalizado"].isin([marcaCarro])]
print(datos)
print("Posiblemente la marca es: ",nombre["marca"].values[0])