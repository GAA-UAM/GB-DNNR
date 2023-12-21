from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.io import arff
import pandas as pd
import numpy as np
import os

random_state = 111
np.random.seed(random_state)
path = "https://github.com/lefman/mulan-extended/tree/master/datasets"


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
