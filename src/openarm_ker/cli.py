"""Command-line interface utilities for OpenArm KER."""

import argparse
import json
import sys
import time
from typing import NoReturn

from .ker_stream import KERStream


def main() -> NoReturn | None:
    """Run the KER CLI.

    Provides diagnostic utilities such as pinging the device and raw streaming.
    """
    parser = argparse.ArgumentParser(
        description="KERStream Command-Line Interface (CLI) Utility",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "command",
        choices=["ping", "stream"],
        help="Command to execute: 'ping' to fetch schema and device metadata, 'stream' to test continuous data reception.",
    )
    parser.add_argument(
        "--transport",
        type=str,
        default="usb",
        choices=["usb", "serial"],
        help="Transport protocol connection type.",
    )
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyACM0",
        help="Serial port path (only applicable when transport is set to 'serial').",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=2000000,
        help="Baud rate speed (only applicable when transport is set to 'serial').",
    )
    args = parser.parse_args()

    stream = KERStream(transport=args.transport, port=args.port, baud=args.baud)

    if args.command == "ping":
        metadata = stream.ping_only()
        if metadata:
            print(json.dumps(metadata, indent=2))
            sys.exit(0)
        else:
            print(
                "Error: Failed to fetch metadata or no response from the device.",
                file=sys.stderr,
            )
            sys.exit(1)

    elif args.command == "stream":
        print(f"[Info] Starting data stream via {args.transport.upper()}...")
        print("[Info] Press Ctrl+C to terminate the stream safely.\n")
        try:
            with stream:
                while stream.is_connected:
                    data = stream.latest()
                    if data:
                        print(f"\r[Stream Data] {data}", end="", flush=True)
                    time.sleep(0.01)
                print("\n[Warning] Stream loop terminated. Device connection lost.")
        except KeyboardInterrupt:
            print("\n[Info] Stream terminated by user. Cleaning up resources.")
            sys.exit(0)
        except Exception as e:
            print(
                f"\n[Critical] Stream crashed with an unexpected error: {e}",
                file=sys.stderr,
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
