import os
import warnings


def tensorflow_silence():
    # thanks to
    # stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints
    try:
        from tensorflow.compat.v1 import logging
        logging.set_verbosity(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['KMP_WARNINGS'] = '0'

        def deprecated(date, instructions, warn_once=False):
            def deprecated_wrapper(func):
                return func
            return deprecated_wrapper

        from tensorflow.python.util import deprecation
        deprecation.deprecated = deprecated

    except ImportError:
        warnings.warn('Failed to silence tensorflow')
