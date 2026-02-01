#!/usr/bin/env python
"""
Linked Data Explorer - Modularized entrypoint.
Author: Huw Sandaver
Version: 2.1.4
Date: 2025-02-18
"""

from src.ui import render_app


def main() -> None:
    render_app()


if __name__ == "__main__":
    main()