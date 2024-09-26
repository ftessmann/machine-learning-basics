import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

def getMae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    predVal = model.predict(val_X)
    mae = mean_absolute_error(val_y, predVal)
    return(mae)


melbFilePath = "melb-housing/input/melb_data.csv"

melbourneData = pd.read_csv(melbFilePath)

print(melbourneData.describe())

print(melbourneData.columns)

melbourneData = melbourneData.dropna(axis=0)

print(melbourneData.describe())

y = melbourneData.Price

melFeatures = ["Rooms", "Bathroom", "Landsize", "Lattitude", "Longtitude", "BuildingArea", "Car"]

X = melbourneData[melFeatures]

print(X.describe())

print(X.head())

melbModel = DecisionTreeRegressor(random_state=1)

print(melbModel.fit(X, y))

print("Predicoções para as proximas casas: ")
print(X)
print("As predições de preço são: ")
print(melbModel.predict(X))

predictedHomePrices = melbModel.predict(X)
print(mean_absolute_error(y, predictedHomePrices))

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

melbModel = DecisionTreeRegressor()
melbModel.fit(train_X, train_y)

valuePredictions = melbModel.predict(val_X)
print(mean_absolute_error(val_y, valuePredictions))

for max_leaf_nodes in [5, 50, 500, 5000]:
    mae = getMae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d \t Mean Absolute Error: %d" %(max_leaf_nodes, mae))
    
finalModel = DecisionTreeRegressor(max_leaf_nodes=500, random_state=1)

finalModel.fit(X, y)

print(melbModel.predict(X))
print(finalModel.predict(X))