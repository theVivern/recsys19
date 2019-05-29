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


# Important project paths
root_dir = Path().resolve()
data_dir = root_dir / 'data'
cache_dir = root_dir / 'cache'

data_meta_path = cache_dir / 'data_meta.h5'
data_sessions_full_path = cache_dir / 'data_sessions_full.h5'
data_sessions_small_path = cache_dir / 'data_sessions_small.h5'


import sys
sys.path.append(str(root_dir / 'src'))
import deppy


# %% Create special directories if they don't exist
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


@deppy.generator(creates=data_meta_path)
def _preprocess_metadata(path: Path = None, nrows: int = None):
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

    df.to_hdf(
        data_meta_path,
        'data_meta'
    )


@deppy.generator(needs=data_meta_path)
def get_metadata():
    result = pd.read_hdf(data_meta_path)
    print(f'Loaded {len(result)} rows of metadata')
    return result


# %% Load the sessions

# This is very memory intensive. Free previous results to avoid using twice as
# much.
try:
    del data_sessions
except NameError:
    pass
else:
    gc.collect()


def _read_sessions_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        header=0,
        dtype={
            'timestamp': np.int64
        })

    return df

def _preprocess_sessions_shared(use_subset: bool, out_path: Path):
    # Load the data
    data_train = _read_sessions_csv(data_dir / 'train.csv')
    data_test = _read_sessions_csv(data_dir / 'test.csv')

    # TODO: Honor the `use_subset` parameter properly
    if use_subset:
        data_train = data_train.head(50000)
        data_test = data_test.head(50000)

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

    # Dump the preprocessed data as CSV
    data.to_hdf(
        out_path,
        'data_sessions'
    )


@deppy.generator(creates=data_sessions_full_path)
def _preprocess_sessions_full():
    _preprocess_sessions_shared(
        False,
        data_sessions_full_path
    )


@deppy.generator(creates=data_sessions_small_path)
def _preprocess_sessions_small():
    _preprocess_sessions_shared(
        True,
        data_sessions_small_path
    )


def get_sessions_data(use_subset: bool):
    # Determine the path to the cached file
    if use_subset:
        path = data_sessions_small_path
    else:
        path = data_sessions_full_path

    # Ensure it exists
    deppy.generate(path)

    # Load it
    result = pd.read_hdf(path)

    # Print statistics
    n_bytes = result.memory_usage().sum()
    print(f'Loaded {len(result)} rows of session data ({n_bytes/1e6:.2f}MB)')

    return result


# LabelEncoders for label columns
label_columns = [
    'user_id', 'session_id',
    'action_type', 'platform', 'city', 'device'
]


label_columns = {
    k: LabelEncoder() for k in label_columns
}



#%%
