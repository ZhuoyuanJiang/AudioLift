#!/usr/bin/env python3
"""Run the existing Stage-2 inference launcher with unchanged arguments."""
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    launcher = Path(__file__).resolve().parent / "stage2" / "inference.sh"
    os.execvp("bash", ["bash", str(launcher), *sys.argv[1:]])
