from molSimplify.Scripts import io
from molSimplify.utils.decorators import deprecated  # noqa F401
from inspect import getmembers, isfunction
import functools  # noqa F401


# The following loop searches for all functions in Scipts.io and
# defines them again in this file after marking them deprecated.
# Since the use of exec() is somewhat unelegant it would be great to
# remove this whole file as soon as possible. RM 2022/03/03


for func in getmembers(io, isfunction):
    exec('@deprecated("Scripts.molSimplify_io is deprecated and will be '
         'removed in a future version. Use Scripts.io instead.")\n'
         f'@functools.wraps(io.{func[0]})\n'
         f'def {func[0]}(*args, **kwargs):\n'
         f'    return io.{func[0]}(*args, **kwargs)')
