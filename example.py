# %%
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.io import arff
from model.gbdnnr import DeepRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

random_state = 111
np.random.seed(random_state)
path = r"D:\Ph.D\Programming\Datasets\Regression\mtr_datasets"


def dataset(name):
    def dt(path):
        df = arff.loadarff(path)
        df = pd.DataFrame(df[0])
        return df

    dt_name = name
    dt_path = os.path.join(path, dt_name)
    df = dt(dt_path)
    info = pd.read_csv(os.path.join(path, "info.txt"))
    d = int(info.d.loc[info["Dataset"] == name])

    if name == "rf1.arff" or "rf2.arff":
        for i in df.columns:
            df[i] = df[i].fillna(0)
    X = (df.iloc[:, :d]).values
    y = (df.iloc[:, d:]).values
    if name == "scpf.arff":
        imp = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        X = imp.fit_transform(X)

    scl = StandardScaler()
    X = scl.fit_transform(X)

    return X, y


info = pd.read_csv(os.path.join(path, "info.txt"))
# print(info.Dataset)

X, y = dataset("atp1d.arff")

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state
)

model = DeepRegressor(
    iter=200,
    eta=1,
    learning_rate=0.01,
    total_nn=600,
    num_nn_step=100,
    batch_size=128,
    early_stopping=5,
    random_state=random_state,
    l2=0.1,
    dropout=0.1,
    record=True,
    freezing=True,
)

t0 = time.process_time()
model.fit(x_train, y_train, x_test, y_test)
print("time", time.process_time() - t0)
print("score", model.score(x_test, y_test))

trained_model = model._models[-1]

trainable_params = sum(
    [tf.keras.backend.count_params(w) for w in trained_model.trainable_weights]
)
non_trainable_params = sum(
    [tf.keras.backend.count_params(w) for w in trained_model.non_trainable_weights]
)
print("Trainable parameters:", trainable_params)
print("Non-trainable parameters:", non_trainable_params)
