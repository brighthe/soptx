"""Generator-based timing utility kept numerically behavior-compatible."""

from __future__ import annotations

from time import time
from typing import Generator


def _timer_core() -> Generator[
    None,
    str | None,
    list[tuple[str | None, float]],
]:
    tags: list[str | None] = [None]
    times = [time()]
    while True:
        tag = yield
        if tag is None:
            break
        tags.append(tag)
        times.append(time())
    return list(zip(tags, times))


def timer(title: str = "") -> Generator[None, str | None, None]:
    """Measure elapsed time between labels sent to the generator."""

    while True:
        result = yield from _timer_core()
        print(f"\n{title} Timer.")
        print(
            "===========================================================\n"
            "   ID       Time        Proportion(%)    Label\n"
            "--------------------------------------------------------"
        )
        previous = result[0][1]
        total = result[-1][1] - previous
        for index in range(1, len(result)):
            label, timestamp = result[index]
            delta = timestamp - previous
            if delta > 1.0:
                time_text = f"{delta:.3f}".rjust(7) + " [s] "
            elif delta > 0.001:
                time_text = f"{delta * 1e3:.3f}".rjust(7) + " [ms]"
            else:
                time_text = f"{delta * 1e6:.3f}".rjust(7) + " [us]"
            previous = timestamp
            proportion = delta / total * 100
            print(
                "  "
                + "    ".join(
                    [
                        f"{index}".rjust(3),
                        time_text,
                        f"{proportion:.3f}".rjust(12),
                        str(label),
                    ]
                )
            )
        print("===========================================================")
        print(f"Total Time: {total:.3f} sec")
        print("===========================================================\n")
