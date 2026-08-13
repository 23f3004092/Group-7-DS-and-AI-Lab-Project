"""
FarmerVision Backend - Development entrypoint.

Run from project root:
    python run.py
    # or
    ./venv/bin/python run.py

This avoids relative-import errors that occur when uvicorn is launched
from inside backend/app/ directly.
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["backend"],
    )
