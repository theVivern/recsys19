# %%
import numpy as np
import pandas as pd
from pathlib import Path
import sys
root_dir = Path().resolve()
sys.path.append(str(root_dir / 'src'))

from recsys_common import *


# %% Read the file
df_full = pd.read_csv(
    Path('outputs/popularity_full.csv')
)

# %% Get all ids part of the validation set
sessions_kept = pd.read_csv(
    Path('cache/df_sessions_small.csv')
)

sessions_kept = sessions_kept.loc[sessions_kept.is_train == False]
sessions_kept = sessions_kept.session_id.unique()

# %%
df_small = df_full.loc[df_full.session_id.isin(sessions_kept)]

display(df_full.shape)
display(df_small.shape)

# %% Dump
df_small.to_csv('out_small.csv', index=False)
