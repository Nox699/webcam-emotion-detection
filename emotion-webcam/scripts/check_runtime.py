"""Check Python from a file so Windows PowerShell cannot strip code quotes."""
import argparse
import struct
import sys


def validate_runtime(version=None, pointer_bytes=None):
    version = sys.version_info[:2] if version is None else tuple(version[:2])
    pointer_bytes = struct.calcsize("P") if pointer_bytes is None else pointer_bytes
    if not (3, 11) <= version < (3, 14):
        raise RuntimeError("Use Python 3.11, 3.12, or 3.13 for this project.")
    if pointer_bytes != 8:
        raise RuntimeError("Use a 64-bit Python installation for this project.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check the supported Python runtime")
    parser.add_argument("--print-executable", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        validate_runtime()
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 1
    if arguments.print_executable:
        print(sys.executable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
