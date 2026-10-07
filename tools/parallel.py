"""Work that splits into pieces, done on several cores at once.

The photo is 186 million pixels, and most of what is done to it is done a tile, a band or a
patch at a time, each piece on its own. scipy's filters let go of Python's lock while they
run, so plain threads are enough: the photo is not copied and nothing has to be passed
between processes.

Two rules keep the result the same whatever the number of cores, down to the last bit. A
piece is worked out from the photo as it was before any piece was changed, never from a
neighbour's result. And the results are put back in the pieces' own order.

SPEEDWAY_CORES in the environment sets how many cores are used. The default is all of them,
up to twelve.
"""
import os
from concurrent.futures import ThreadPoolExecutor

CORES = max(1, int(os.environ.get("SPEEDWAY_CORES", min(os.cpu_count() or 1, 12))))


def each(work, pieces, most=None):
    """[work(piece) for piece in pieces], several at a time.

    most caps how many run at once, for work that needs a lot of memory for each piece.
    """
    pieces = list(pieces)
    count = min(CORES if most is None else min(most, CORES), len(pieces))
    if count <= 1:
        return [work(piece) for piece in pieces]
    with ThreadPoolExecutor(count) as pool:
        return list(pool.map(work, pieces))


def each_into(work, pieces, use, most=None):
    """use(work(piece)) for every piece, in the pieces' own order, with the work done several at a time.

    For work whose results are large: each is handed to use as soon as its turn comes and
    then let go, so no more than a few are held at once.
    """
    pieces = list(pieces)
    count = min(CORES if most is None else min(most, CORES), len(pieces))
    if count <= 1:
        for piece in pieces:
            use(work(piece))
        return
    with ThreadPoolExecutor(count) as pool:
        waiting = []
        for piece in pieces:
            waiting.append(pool.submit(work, piece))
            if len(waiting) >= 2 * count:           # no more than this many started or held at a time
                use(waiting.pop(0).result())
        for result in waiting:
            use(result.result())

