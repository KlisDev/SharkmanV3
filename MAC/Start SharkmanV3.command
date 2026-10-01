#!/bin/bash
cd -- "$(dirname -- "$0")" || exit 1
if command -v python3.12 >/dev/null 2>&1; then
    python3.12 easy_run_mac.py "$@"
elif command -v python3 >/dev/null 2>&1; then
    python3 easy_run_mac.py "$@"
else
    echo "Install Python 3.12 with Tk from python.org first."
fi
echo
read -r -p "Press Return to close this launcher... "
