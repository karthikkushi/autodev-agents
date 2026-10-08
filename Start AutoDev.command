#!/bin/bash
# Double-click to start AutoDev: the dashboard opens in your browser and the
# pipeline worker starts — unfinished projects resume where they stopped.
# Closing this window stops everything; progress is saved after every step.
cd "$(dirname "$0")"
exec python3 main.py
