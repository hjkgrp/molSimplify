import warnings
import functools


def deprecated(message):
    """Returns a decorator that adds a DeprecationWarning with
    the provided message."""

    def decorator(func):
        # Use functools.wraps to preserve docstrings etc.
        @functools.wraps(func)
        def inner(*args, **kwargs):
            warnings.warn(message, DeprecationWarning)
            return func(*args, **kwargs)
        return inner
    return decorator
