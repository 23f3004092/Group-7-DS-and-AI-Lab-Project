"""Tiny shared logging setup. Logs go to stdout, so you read them with:
    docker compose logs -f gateway
Set LOG_LEVEL=DEBUG (in runtime.env / docker-compose) for more detail; default INFO.
Every module does:  from .log import get ; log = get("<module>")
"""
import logging
import os
import sys

_configured = False


def _setup():
    global _configured
    if _configured:
        return
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-5s [%(name)s] %(message)s", datefmt="%H:%M:%S"))
    root = logging.getLogger("fv")
    root.setLevel(level)
    root.handlers[:] = [handler]
    root.propagate = False          # don't double-log through uvicorn's root
    _configured = True


def get(name: str) -> logging.Logger:
    _setup()
    return logging.getLogger(f"fv.{name}")

