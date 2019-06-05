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


# %% Loads the metadata
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


@deppy.cache(cache_dir=cache_dir)
def get_metadata(path: Path = None, nrows: int = None):
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


# %% Loads the sessions

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

def filter_sessions_with_no_clicks(data_train):
    sessions_with_clickouts = data_train.loc[data_train.action_type=='clickout item','session_id']
    sessions_with_clickouts
    data_train=data_train.loc[data_train.session_id.isin(sessions_with_clickouts)].copy()
    
    return data_train

def get_unique_session_id(data_train):
    session_ids_sorted=data_train.loc[data_train.step==1].sort_values('timestamp').session_id.unique()
    session_ids_sorted
    
    return session_ids_sorted


@deppy.cache(cache_dir=cache_dir)
def get_sessions(use_subset: bool, frac_sessions: float, create_validation: bool, frac_for_fake: float):
    # Load the data
    data_train = _read_sessions_csv(data_dir / 'train.csv')
    data_test = _read_sessions_csv(data_dir / 'test.csv')

    # This filters sessions with clickouts and then orders those sessions
    # by their timestamp for first step. The returns the full session of
    # fracsessions for the earliest timestamps
    if use_subset:
        data_train=filter_sessions_with_no_clicks(data_train)
        session_ids_sorted = get_unique_session_id(data_train)
        data_train = data_train.loc[data_train.session_id.isin(session_ids_sorted[0:round(len(session_ids_sorted)*frac_sessions)])].copy()

        data_test=filter_sessions_with_no_clicks(data_test)
        test_session_ids_sorted = get_unique_session_id(data_test)
        data_test = data_test.loc[data_test.session_id.isin(test_session_ids_sorted[0:round(len(test_session_ids_sorted)*frac_sessions)])].copy()

        del session_ids_sorted, test_session_ids_sorted

    if create_validation:
        data_train=filter_sessions_with_no_clicks(data_train)
        
        session_ids_sorted = get_unique_session_id(data_train)

        index_for_split=round(len(session_ids_sorted)*(1-frac_for_fake))
        index_for_split

        data_train_x = data_train.loc[data_train.session_id.isin(session_ids_sorted[0:index_for_split])].copy()
        print(data_train_x.shape)
        data_train_x
        data_train_y = data_train.loc[data_train.session_id.isin(session_ids_sorted[(index_for_split):len(session_ids_sorted)])].copy()
        print(data_train_y.shape)
        data_train_y

        # add dummy switch for
        data_train_x['is_validation'] = False
        data_train_y['is_validation'] = True
        data_train = pd.concat(
            (data_train_x, data_train_y),
            axis=0,
        )

        data_test['is_validation'] = False

        del data_train_x, data_train_y


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


    return data


# takes the already split is_train==True & is_validation==True and creates
# a ground truth (gt) and dev_test on frac_nan % of sampled clickout references.
def process_validation(dev, frac_nan: float, seed: int):
    dev['key'] = (dev['session_id'] + '_' + dev['step'].astype(str))

    last_clickout_in_session=dev.loc[dev.action_type=='clickout item'].groupby('session_id',as_index=False)['step'].max()
    last_clickout_in_session['key']=(last_clickout_in_session['session_id'] + '_' + last_clickout_in_session['step'].astype(str))
    
    n_references=round(frac_nan*len(last_clickout_in_session))
    np.random.seed(seed)
    index_sampled_clickouts=np.random.choice(last_clickout_in_session.key,n_references,replace=False)
    
    dev['target']=dev.loc[dev.key.isin(index_sampled_clickouts),'reference']
    dev.loc[dev.key.isin(index_sampled_clickouts),'reference']=np.NaN

    dev.drop('key',axis=1,inplace=True)

    return dev


# %% Create the CSV files
def create_common_csvs():
    dataframes = {
        'df_metadata':
            lambda: get_metadata(),

        'df_sessions_full':
            lambda: get_sessions(False, 1, True, 0.25),

        'df_sessions_small':
            lambda: get_sessions(True, .05, True, 0.25),
    }

    print('Generating common CSV datafiles.')
    print(f'They are stored at {cache_dir.resolve()}')

    for name, df_generator in dataframes.items():
        print(f'Creating {name}')

        file_name = f'{name}.csv'
        file_path = cache_dir / file_name

        # Skip if the file exists
        if file_path.exists():
            print('  File exists, skipping')
            continue

        # Generate the data
        df = df_generator()

        # Print statistics
        n_bytes = df.memory_usage().sum()
        print(f'    {len(df)} rows, ({n_bytes/1e6:.2f}MB)')

        # Dump it
        print('  Saving CSV')
        df.to_csv(file_path)


