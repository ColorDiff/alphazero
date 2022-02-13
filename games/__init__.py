from games.connect_x import ConnectX
from games.crosskalah import CrossKalah


def make_game(name, *args, **kwargs):
    if name == ConnectX.__name__:
        return ConnectX(*args, **kwargs)
    elif name == CrossKalah.__name__:
        return CrossKalah(*args, **kwargs)
    raise ValueError(name)
