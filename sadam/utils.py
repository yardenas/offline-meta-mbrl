from itertools import zip_longest


# https://docs.python.org/3/library/itertools.html#itertools-recipes
def grouper(iterable, n, *, incomplete="fill", fillvalue=None):
    """Collect data into non-overlapping fixed-length chunks or blocks"""
    # grouper('ABCDEFG', 3, fillvalue='x') --> ABC DEF Gxx
    # grouper('ABCDEFG', 3, incomplete='ignore') --> ABC DEF
    args = [iter(iterable)] * n
    if incomplete == "fill":
        return zip_longest(*args, fillvalue=fillvalue)
    if incomplete == "ignore":
        return zip(*args)
    else:
        raise ValueError("Expected fill or ignore")
