import gc
import logging
import time
import tracemalloc
from typing import Any, Callable

from nuscenes import NuScenes
from nuscenes._lib import Tables as NuScenesRust

DEBUG = True


def profile_memory(func: Callable, *args: Any, **kwargs: Any) -> None:
    tracemalloc.start()
    func(*args, **kwargs)
    curr, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Function {func.__name__}")
    print(f"Current memory usage: {curr / 10**6:.6f} MB")
    print(f"Peak memory usage: {peak / 10**6:.6f} MB")
    print("-" * 30)


def nuscenes_python(version: str):
    nusc = NuScenes(version, "../data/", verbose=DEBUG)
    return nusc


def nuscenes_rust(version: str):
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("nuscenes")
    logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
    logger.addHandler(handler)
    nusc = NuScenesRust(version, "../data/")
    return nusc


def main() -> None:
    version = "v1.0-trainval"

    nuscenes_python(version)
    time.sleep(5)

    gc.collect()
    time.sleep(5)

    nuscenes_rust(version)
    time.sleep(5)

    # print("Timing Python implementation")
    # python_time = timeit.timeit(lambda: nuscenes_python(version), number=5) / 5

    # print("Timing Rust implementation")
    # rust_time = timeit.timeit(lambda: nuscenes_rust(version), number=5) / 5

    # print(f"Python time: {python_time}\nRust time: {rust_time}")
    # print(f"Speed ups: {python_time / rust_time:.2f}x")

    # profile_memory(nuscenes_python, version)
    # gc.collect()
    # profile_memory(nuscenes_rust, version)


if __name__ == "__main__":
    main()
