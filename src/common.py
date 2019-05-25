# %%
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer, FunctionTransformer, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, TransformerMixin
from sklearn.base import BaseEstimator
from sklearn_pandas import DataFrameMapper
import gc


# %% Important project paths
root_dir = Path().resolve()
data_dir = root_dir / 'data'
cache_dir = root_dir / 'cache'


# %% Create special directories if the don't exist
cache_dir.mkdir(parents=True, exist_ok=True)


# %% Load the metadata
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


def load_metadata(path: Path = None, nrows: int = None):
    if path is None:
        path = data_dir / 'item_metadata.csv'

    # Read
    df = pd.read_csv( str(path),
        header=0,
        nrows = nrows,
        dtype={
            'item_id': np.int64
        })

    # Preprocess
    pipe = MetadataEncoder()
    df = pipe.fit_transform(df)

    return df

data_meta = load_metadata()
print(f'Loaded {len(data_meta)} rows of metadata')


# %% Load the sessions

# Free up memory
try:
    del data_sessions
except NameError:
    pass
else:
    gc.collect()


def load_session_data(path: Path, nrows: int = None):
    df = pd.read_csv( str(path),
        header=0,
        nrows=nrows,
        dtype={
            'timestamp': np.int64
        })

    return df


# LabelEncoders for label columns
label_columns = [
    'user_id', 'session_id',
    'action_type', 'platform', 'city', 'device'
]

label_columns = {
    k: LabelEncoder() for k in label_columns
}


def load_sessions(nrows: int = None):
    # Load the data
    data_train = load_session_data(data_dir / 'train.csv', nrows=nrows)
    data_test = load_session_data(data_dir / 'test.csv', nrows=nrows)

    # Merge the dataframes
    data_train['is_train'] = True
    data_test['is_train'] = False

    data = pd.concat(
        (data_test, data_train),
        axis=0
    )

    del data_train, data_test
    gc.collect()

    # Preprocess the columns
    # ... timestamp
    data.timestamp = pd.to_datetime(
        data.timestamp,
        unit='s',
        utc=True
    )

    # ... convert categorical strings to integers
    for colname, encoder in label_columns.items():
        col = encoder.fit_transform(data[colname])

        # Save memory
        assert len(encoder.classes_) < 10e6, (colname, len(encoder.classes_))
        col = col.astype(np.int32)

        data[colname] = col

    return data


data_sessions = load_sessions()
print(f'Loaded {len(data_sessions)} rows of session data')
n_bytes = data_sessions.memory_usage().sum()
print(f'Session data is using {n_bytes/1e6:.2f}MB')


