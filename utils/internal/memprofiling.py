from cparsing import parse_file


def heapyprofile(filepath: str) -> None:
    """Profile with guppy/heapy."""

    # pip install guppy
    # [works on python 2.7, AFAIK]
    import gc

    from guppy import hpy

    hp = hpy()
    _ = parse_file(filepath)
    gc.collect()
    h = hp.heap()
    print(h)


def memprofile(filepath: str) -> None:
    """Profile memory with `resource` and `tracemalloc`."""

    import resource
    import tracemalloc

    tracemalloc.start()

    _ = parse_file(filepath)

    print(f"Memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss} (kb)")

    snapshot = tracemalloc.take_snapshot()
    print("[ tracemalloc stats ]")
    for stat in snapshot.statistics("lineno")[:20]:
        print(stat)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser("Memory Profiling")
    parser.add_argument("filepath", default="zc.c")
    filepath: str = parser.parse_args().filepath

    memprofile(filepath)
    heapyprofile(filepath)


if __name__ == "__main__":
    raise SystemExit(main())
