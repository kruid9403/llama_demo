import argparse
import os

from .server import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the hypothesis monitor UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.getenv("HYPOTHESIS_MONITOR_PORT", "8030")))
    args = parser.parse_args()
    run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
