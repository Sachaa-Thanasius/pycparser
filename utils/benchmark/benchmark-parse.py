# -----------------------------------------------------------------
# Benchmarking utility for internal use.
#
# Use with Python 3.6+
#
# Eli Bendersky [https://eli.thegreenplace.net/]
# License: BSD
# -----------------------------------------------------------------
import statistics
import time
from collections.abc import Callable
from pathlib import Path

from cparsing import c_ast, parse


def measure_parse(text: str, n: int, progress_cb: Callable[[int], None]) -> list[float]:
    """Measure the parsing of text with pycparser.

    Parameters
    ----------
    text: str
        Representation of a full file.
    n: int
        Number of iterations to measure.
    progress_cb: Callable[[int], None]
        Callback claled with the iteration number each time an iteration completes.

    Returns
    -------
    times: list[float]
        A list of elapsed times, one per iteration.
    """

    times: list[float] = []
    for i in range(n):
        start = time.perf_counter()
        ast = parse(text)
        elapsed = time.perf_counter() - start

        assert isinstance(ast, c_ast.File)
        times.append(elapsed)
        progress_cb(i)
    return times


def measure_file(filename: Path, n: int) -> None:
    def progress_cb(_: int) -> None:
        return print(".", sep="", end="", flush=True)

    with open(filename) as f:
        print(f"{filename.name:25}", end="", flush=True)
        text = f.read()
        times = measure_parse(text, n, progress_cb)

    print(f"    Mean:   {statistics.mean(times):.3f}")
    print(f"    Stddev: {statistics.stdev(times):.3f}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input-dir", type=Path, help="dir with input files")
    parser.add_argument("--num-runs", default=5, type=int, help="number of runs")
    args = parser.parse_args()

    input_dir: Path = args.filename
    num_runs: int = args.num_runs

    for filename in input_dir.iterdir():
        measure_file(filename, num_runs)


if __name__ == "__main__":
    raise SystemExit(main())
