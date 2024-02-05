from typing import Any, Dict


class Singleton(type):
    """Follows Method #2 in
    https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python"""
    _instances: Dict[type, Any] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
