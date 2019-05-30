# Copyright 2019 Jakob Pinterits

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from typing import Iterable, Union
import pickle
from pathlib import Path
import time
import hashlib


Pathlike = Union[str, Path]
Pathlist = Union[Pathlike, Iterable[Pathlike]]


class DeppyError(Exception):
    pass


def as_pathlist(raw: Pathlist):
    if raw is None:
        raw = []

    if isinstance(raw, (str, Path)):
        raw = [raw]

    return [
        Path(x).resolve() for x in raw
    ]


def deppy_hash(value, max_depth=3) -> int:
    if max_depth < 0:
        return 834  # Random number, no relevance

    # Simple numeric types
    if isinstance(value, (int, float, bool)):
        return int(value * 10000)

    # List-like
    if isinstance(value, (list, tuple)):
        accu = 744 # Random number, no relevance

        for child in value:
            accu ^= deppy_hash(child, max_depth-1)
            accu *= 3  # Make the order of children important

        return accu

    # Dict
    if isinstance(value, dict):
        # Make sure the order _doesn't_ matter
        accu = 427 # Random number, no relevance

        for key, val in value.items():
            accu ^= deppy_hash(key, max_depth-1)
            accu ^= deppy_hash(val, max_depth-1)

        return accu

    # Strings
    if isinstance(value, str):
        digest = hashlib.sha1(value.encode('UTF-8')).hexdigest()
        return int(digest, 16)

    # Unhandled
    print(f'Deppy: Warning: Cannot hash {type(value)} instance. Cached results may be invalid!')
    return 239  # Random number, no relevance


def cache(cache_dir: Path = Path(),
        dump_function: callable = lambda obj, path: pickle.dump(obj, path.open('wb')),
        load_function: callable = lambda path: pickle.load(path.open('rb')),
        file_extension: str = '.pickle'):

    def decorator(func: callable):

        def load_func(*args, **kwargs):
            arg_hash = deppy_hash(args) ^ deppy_hash(kwargs)
            arg_hash_str = hex(arg_hash)[2:]

            cache_file_name = \
                f'deppy_cache__{func.__name__}__{arg_hash_str}{file_extension}'

            cache_file_path = cache_dir / cache_file_name

            # Load the cached result, if available
            if cache_file_path.exists():
                return load_function(cache_file_path)

            # Regenerate and cache the results
            # print(f'Deppy: Generating {path.name}')
            result = func(*args, **kwargs)

            cache_file_path.resolve().parent.mkdir(
                parents=True, exist_ok=True
            )

            dump_function(result, cache_file_path)
            return result

        return load_func

    return decorator



