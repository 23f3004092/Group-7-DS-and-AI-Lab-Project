#!/usr/bin/env python3
"""
NOTE: This script has been moved to scripts/download_data.py.
Please use scripts/download_data.py instead.
"""
import sys
from pathlib import Path

if __name__ == "__main__":
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "download_data.py"
    print(f"ℹ️  download_data.py has been moved to: {script_path}")
    print(f"👉 Please run: python scripts/download_data.py {' '.join(sys.argv[1:])}")
    sys.exit(1)
