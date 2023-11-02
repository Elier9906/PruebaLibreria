from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

def mtrrf():
    x, y = make_regression(n_targets=3)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42) 
    clf = MultiOutputRegressor(RandomForestRegressor(max_depth=2, random_state=0)) 
    clf.fit(x_train, y_train)
    return clf.predict(x_test)

def mtrrf (data):

    data = pd.read_csv(data)
    df=pd.DataFrame(data)

    x = data.drop(columns=['pm', 'stator_yoke', 'stator_tooth', 'stator_winding'])
    y = pd.DataFrame (df[['pm', 'stator_yoke', 'stator_tooth', 'stator_winding']])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
    clf = MultiOutputRegressor(RandomForestRegressor(max_depth=2, random_state=0))
    clf.fit(x_train, y_train)
    return clf.predict(x_test)