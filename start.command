#!/usr/bin/env bash
# Double-clickable launcher for macOS. Finder opens a plain .sh in a text
# editor, but runs a .command file in Terminal on double-click -- this is
# the macOS equivalent of start.bat. It just hands off to install.sh.
#
# First time only: macOS may need this marked executable. From Terminal:
#   chmod +x start.command
# (a `git clone` already sets that bit; a ZIP download strips it.)
set -uo pipefail
cd "$(dirname "$0")" || exit 1

./run.sh "$@"
status=$?
if [ "$status" -ne 0 ] && [ "$status" -ne 130 ] && [ "$status" -ne 143 ]; then
  echo
  echo "Afterimage didn't start (exit $status). Read the message above for what to fix."
  echo "Press any key to close this window."
  read -r -n 1 -s || true
fi
