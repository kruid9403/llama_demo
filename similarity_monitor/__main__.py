import argparse
import os

from .server import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the similarity search monitor UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.getenv("SIMILARITY_MONITOR_PORT", "8020")))
    args = parser.parse_args()
    run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
