import argparse

from .server import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the autonomous spider monitor UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
