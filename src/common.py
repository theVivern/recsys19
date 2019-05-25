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


def load_metadata(path: Path):
    df = pd.read_csv( str(path),
        header=0,
        dtype={
        })

    return df


def load_dataset(nrows: int = None):
    df_train = load_sessions(data_dir / 'train.csv', nrows)
    df_test = load_sessions(data_dir / 'test.csv', nrows)
    df_meta = load_metadata(data_dir / 'metadata.csv')

    return df_train, df_test, df_meta

df_train = load_sessions(data_dir / 'train.csv', 1000)
df_meta = load_metadata(data_dir / 'item_metadata.csv')




#%% Data preprocessing pipelines
# sessions_pipe =
