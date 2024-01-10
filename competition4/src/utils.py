from functools import partial, reduce


def compose(*functions):
    return partial(reduce, lambda x, f: f(x), [*functions])
