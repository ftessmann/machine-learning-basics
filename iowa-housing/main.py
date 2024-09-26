import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

filePath = "./iowa-housing/input/train.csv"

homeData = pd.read_csv(filePath)

print(homeData.describe())

y = homeData.SalePrice

print(y)

homeFeatures = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]

X = homeData[homeFeatures]

print(X.describe())

print(X.head())

homeModel = DecisionTreeRegressor(random_state=1)

print(homeModel.fit(X, y))

print("Predição para as seguintes casas: ")
print(X)
print("Predição de valores: ")
print(homeModel.predict(X))

print("Valor das amostras no modelo treinado: ", homeModel.predict(X.head()))
print("Valor real: ", y.head().tolist())

predictedHomePrices = homeModel.predict(X)
print("Discrepância média de valores: ", mean_absolute_error(y, predictedHomePrices))

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

homeModel = DecisionTreeRegressor()
homeModel.fit(train_X, train_y)

valuePredictions = homeModel.predict(val_X)

print("Valor das amostras de treinamento: ", homeModel.predict(val_X.head()))
print("Discrepância média de valores: ", mean_absolute_error(val_y, valuePredictions))