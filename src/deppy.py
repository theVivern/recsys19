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
from pathlib import Path
import time


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


# List of all Generator objects
generators = []


def find_generator_for_path(path: Path):
    """
    Find the appropriate generator for a path. A KeyError is raised if there
    is no matching generator.
    """

    # This function performs a linear search over all generators. While this
    # could be sped up using a dict this would introduce a second copy of some
    # data. As there is likely a small number of files and generators a linear
    # search is preferable over potentially inconsistent data.

    # Paths in Generators are resolved as well, so resolve this one to ensure
    # matches.
    path = path.resolve()

    for generator in generators:
        if path in generator.creates:
            return generator

    raise DeppyError(
        f'The file at {path} is listed as dependency but not provided by any generator')


def generator(
        needs: Pathlist = None,
        creates: Pathlist = None,
        check_time: bool = True):

    """
    Decorator: Functions decorated with this can specify needed as well as
    created files. Whenever the function is called Deppy ensures that all needed
    files exist by calling the appropriate other decorators.
    """

    def decorator(func: callable):
        return Generator(
            needs,
            creates,
            check_time,
            func
        )

    return decorator


def toposort(generator: 'Generator'):
    """
    Returns a topological sorting of the generator and all it's dependencies.
    """

    result = []
    to_do = [generator]
    visiting = []
    visited = set()


    def visit(gen: Generator):
        # Detect dependency cycles
        if gen in visiting:
            # TODO better error message
            raise DeppyError('Dependency cycle')

        # Mark the generator as currently visiting
        visiting.append(gen)

        # Chain to dependencies
        for dep in gen.needs_generators():
            visit(dep)

        # Add the generator to the result, if it wasn't added by the
        # dependencies already
        if gen not in visited:
            result.append(gen)

        # Mark the generator as visited
        visited.add(gen)

    visit(generator)

    return result


def generate(paths: Pathlist):
    """ Ensures that all given files exist, generating them if necessary """
    paths = as_pathlist(paths)

    for path in paths:
        gen = find_generator_for_path(path)

        # Get a topological sorting of the generators
        order = toposort(gen)

        for gen in order:
            outdated = gen.outdated_creates()

            for path, reasons in outdated.items():
                print(f'Deppy: Generating {path.name} ({", ".join(reasons)})')

            if outdated:
                gen.create_function()

                # Make sure the user supplied function actually generated the files
                gen.validate_outputs_exist()


class Generator:
    def __init__(
            self,
            needs: Pathlist,
            creates: Pathlist,
            check_time: bool,
            create_function: callable):

        self.needs = as_pathlist(needs)
        self.creates = as_pathlist(creates)
        self.check_time = check_time
        self.create_function = create_function

        # Register the generator
        generators.append(self)

    def __hash__(self):
        return id(self)

    def outdated_creates(self):
        """
        Finds any outputs that need to be regenerated. The output is a
        Dictionary mapping paths to lists of human readable reasons why
        regeneration in necessary.
        """

        result = {}

        def add(path: Path, reason: str):
            try:
                old = result[path]
            except KeyError:
                old = []
                result[path] = old

            old.append(reason)

        for path in self.creates:
            # File doesn't exist at all
            if not path.exists():
                add(path, 'does not exist')

        # Dependencies are younger than outputs
        if self.check_time:
            newest_need_time = None
            for path in self.needs:
                try:
                    file_mtime = path.stat().st_mtime
                except FileNotFoundError:
                    pass

                if newest_need_time is None:
                    newest_need_time = file_mtime
                else:
                    newest_need_time = min(file_mtime, newest_need_time)

            if newest_need_time is not None:
                for path in self.creates:
                    try:
                        if path.stat().st_mtime < newest_need_time:
                            add(path, 'out of date')
                    except FileNotFoundError:
                        pass

        return result

    def validate_outputs_exist(self):
        """ Ensures all outputs exist, raising a DeppyError otherwise """

        for path in self.creates:
            if not path.exists():
                raise DeppyError(
                    f'Generator failed to create the output file at {path}')

    def needs_generators(self):
        """ Returns the set of all generators this one depends on """
        return set(
            find_generator_for_path(p) for p in self.needs
        )

    def __call__(self, *args, **kwargs):
        # Generate all dependencies
        generate(self.needs)

        # Run the user function
        result = self.create_function(*args, **kwargs)

        # Make sure the user supplied function actually generated the files
        self.validate_outputs_exist()

        return result

