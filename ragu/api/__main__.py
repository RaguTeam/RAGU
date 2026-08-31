"""Entry point: ``python -m api``."""

import argparse
import logging

import uvicorn

from ragu.api.app import create_app
from ragu.api.config import ServiceSettings


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGU search service")
    parser.add_argument(
        "--host", default=None, help="Bind address (default from RAGU_HOST)"
    )
    parser.add_argument(
        "--port", type=int, default=None, help="Bind port (default from RAGU_PORT)"
    )
    parser.add_argument(
        "--backend", choices=["ragu", "stub"], default=None, help="Search backend"
    )
    parser.add_argument(
        "--storage-folder",
        default=None,
        help="RAGU storage folder with the built graph",
    )
    parser.add_argument("--log-level", default="info", help="Log level")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper())

    overrides = {
        key: value
        for key, value in (
            ("host", args.host),
            ("port", args.port),
            ("backend", args.backend),
            ("storage_folder", args.storage_folder),
        )
        if value is not None
    }
    settings = ServiceSettings(**overrides)

    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
