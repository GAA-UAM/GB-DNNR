# %%
import os
import numpy as np
import pandas as pd
from scipy.io import arff
from model.gbdnnr import DeepRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


name = "oes10.arff"
d = 298
random_state = 1

np.random.seed(random_state)


path = r"D:\Academic\Ph.D\Programming\Datasets\Regression\mtr_datasets"


def dt(path):
    df = arff.loadarff(path)
    df = pd.DataFrame(df[0])
    return df


def df(name, d):
    dt_name = name
    dt_path = os.path.join(path, dt_name)
    df = dt(dt_path)
    X = (df.iloc[:, :d]).values
    y = (df.iloc[:, d:]).values

    scl = StandardScaler()
    X = scl.fit_transform(X)

    return X, y


X, y = df(name, d)
# y = y[:, 3]


x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state
)

model = DeepRegressor(
    iter=200,
    eta=0.75,
    learning_rate=0.1,
    total_nn=300,
    num_nn_step=100,
    batch_size=128,
    early_stopping=None,
    random_state=random_state,
    l2=0.001,
    dropout=0.1,
    record=True,
    freezing=False,
)

model.fit(x_train, y_train, x_test, y_test)
mean_squared_error(
    y_test, model.predict(x_test), multioutput="raw_values", squared=False
)

# %%
