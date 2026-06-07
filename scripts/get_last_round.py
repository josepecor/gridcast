#!/usr/bin/env python3
"""
Imprime el número de la última ronda completada de la temporada activa.
Usado por los workflows de GitHub Actions para determinar --as-of-round.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gridcast.config import get_settings
from gridcast.ingest.jolpica_client import get_schedule


def main() -> None:
    cfg = get_settings()
    schedule = get_schedule(cfg.season.active)
    today = date.today()
    completed = [r for r in schedule if r.date < today]
    if completed:
        print(completed[-1].round)
    else:
        print(0)


if __name__ == "__main__":
    main()
