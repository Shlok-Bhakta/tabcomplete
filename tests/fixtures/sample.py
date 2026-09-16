"""Shared fixture source for dataset generation and notebook demos."""

import os
import sys
from pathlib import Path

CONSTANT = 42


def add(a, b):
    total = a + b
    return total


def greet(name):
    if name:
        return "hi " + name
    return "hi"


def main(argv):
    target = Path(argv[1]) if len(argv) > 1 else Path("out.txt")
    target.write_text(f"{greet('world')} {add(1, 2)}{os.linesep}")


if __name__ == "__main__":
    main(sys.argv)
