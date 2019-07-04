import numpy as np
import pandas as pd
from typing import List
from collections import defaultdict
from pathlib import Path


def find_uniques(df: pd.DataFrame):
    return set(map(tuple, df.itertuples(index=False)))


def ensemble(frames: List[pd.DataFrame]):
    partials = {}

    # Scan
    for ii, df in enumerate(frames):
        print(f'Merging dataframe {ii}')

        for row in df.itertuples():
            key = (
                row.user_id,
                row.session_id,
                row.timestamp,
                row.step,
            )


            items = np.array(row.item_recommendations.strip().split(' '))

            probs = np.array(
                [float(p) for p in row.item_probs.strip().split(' ')],
                dtype=np.float32
            )

            assert len(items) == len(probs), (len(items), len(probs))

            # Standardize the order
            ind = np.argsort(items)
            probs = probs[ind]

            # Standardize the probabilities
            sm = np.sum(probs)
            if sm == 0:
                continue

            probs /= sm


            # Save the result
            try:
                old_keys, old_probs = partials[key]
            except KeyError:
                partials[key] = (items[ind], probs)
            else:
                old_probs += probs

    # Write out
    print('Generating result')

    result = pd.DataFrame(
        columns=['user_id','session_id','timestamp','step','item_recommendations', 'item_probs']
    )

    for key, data in partials.items():
        items, probs = data

        ind = np.argsort(probs)[::-1]
        items = items[ind]
        probs = probs[ind]

        row = (
            key[0],
            key[1],
            key[2],
            key[3],
            ' '.join(items),
            ' '.join(map(str, probs)),
        )

        result.loc[len(result)] = row

        # result.append(
        #     {
        #         'user_id': key[0],
        #         'session_id': key[1],
        #         'timestamp': key[2],
        #         'step': key[3],
        #         'item_recommendations': items
        #     }
        # )


    return result


def create_test_csv():
    return pd.DataFrame(
        data={
            'user_id':
            ['u_1', 'u_2', 'u_1'],

            'session_id':
            ['sess_1', 'sess_2', 'sess_1'],

            'timestamp':
            ['t_1', 't_2', 't_1'],

            'step':
            [1, 9, 1],

            'item_recommendations':
            [
                'i1 i2 i3 i4',
                'i1 i2',
                'i1 i2 i3 i4',
            ],

            'item_probs':
            [
                '0.1 0.2 0.3 0.4',
                '0.8 0.2',
                '1.0 0.0 0.0 0.0',
            ],
        }
    )


def main():
    pth = Path(__file__).resolve().parent.parent / 'outputs'

    frames = []

    for fpath in pth.glob('*.csv'):
        if fpath.name == 'ensemble.csv':
            print('Skipping', fpath.name)
            continue

        print('Reading ', fpath.name)
        frame = pd.read_csv(fpath)

        print(' ', len(frame), 'rows')

        frames.append(frame)

    result = ensemble(frames)

    result.to_csv(
        pth / 'ensemble.csv',
        index=False
    )



if __name__ == '__main__':
    main()
