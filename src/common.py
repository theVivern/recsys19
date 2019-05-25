# %%
import numpy as np
import pandas as pd
from pathlib import Path


# %% Important project paths
root_dir = Path().resolve()
data_dir = root_dir / 'data'
cache_dir = root_dir / 'cache'


# %% Create special directories if the don't exist
cache_dir.mkdir(parents=True, exist_ok=True)


# %% Dataset loading

def load_sessions(path: Path, nrows: int = None):
    df = pd.read_csv( str(path),
        header=0,
        nrows=nrows,
        dtype={
        })

    # raise NotImplementedError

    return df


def load_metadata(path: Path, nrows: int = None):
    df = pd.read_csv( str(path),
        header=0,
        nrows = nrows,
        dtype={
        })

    return df


def load_dataset(nrows: int = None):
    df_train = load_sessions(data_dir / 'train.csv', nrows)
    df_test = load_sessions(data_dir / 'test.csv', nrows)
    df_meta = load_metadata(data_dir / 'metadata.csv')

    return df_train, df_test, df_meta

# df_train = load_sessions(data_dir / 'train.csv', 1000)
df_meta = load_metadata(data_dir / 'item_metadata.csv', 1000)


# %%
from sklearn.preprocessing import MultiLabelBinarizer, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, TransformerMixin
from sklearn.base import BaseEstimator


class MetadataEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = MultiLabelBinarizer()

    def _split(self, x):
        return x.str.split('|')

    def fit(self, x, *unused):
        x = self._split(x.properties)
        self.encoder.fit(x)
        return self

    def transform(self, x, *unused):
        # Classes
        out = self._split(x.properties)
        out = self.encoder.transform(out)

        return pd.DataFrame(
            data=np.concatenate(
                (x.item_id[:, None], out),
                axis=1),
            columns=['item_id'] + list(self.encoder.classes_)
        )


meta_pipe = MetadataEncoder()
meta_pipe.fit(df_meta)
tmp = meta_pipe.transform(df_meta)




#%%
