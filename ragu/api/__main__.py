"""Entry point: ``python -m ragu.api``."""

import argparse

import uvicorn

from ragu.api.app import create_app
from ragu.api.config import ServiceSettings
from ragu.api.logging_setup import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGU search service")
    parser.add_argument(
        "--host", default=None, help="Bind address (default from RAGU_API_HOST)"
    )
    parser.add_argument(
        "--port", type=int, default=None, help="Bind port (default from RAGU_API_PORT)"
    )
    parser.add_argument(
        "--backend",
        choices=["ragu", "stub"],
        default=None,
        help="Search backend (default from RAGU_API_BACKEND)",
    )
    parser.add_argument(
        "--storage-folder",
        default=None,
        help="RAGU storage folder with the built graph (default from RAGU_API_STORAGE_FOLDER)",
    )
    parser.add_argument("--log-level", default="info", help="Log level")
    args = parser.parse_args()

    configure_logging(args.log_level)

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
        # configure_logging already routed stdlib logging into loguru; uvicorn's
        # own dictConfig would install a second set of handlers on top.
        log_config=None,
    )


if __name__ == "__main__":
    main()
