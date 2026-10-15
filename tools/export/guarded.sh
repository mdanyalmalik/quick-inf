#!/bin/sh
# Runs a command and stops it if the Mac starts swapping heavily: when swap in use grows more than
# MAX_SWAP_GROWTH_MB (default 2048) above its level at the start, the command's process group is killed.
# Usage: tools/export/guarded.sh <command> [args...]
limit=${MAX_SWAP_GROWTH_MB:-2048}
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
start=$(swap_mb)
set -m
"$@" &
pid=$!
while kill -0 "$pid" 2>/dev/null; do
  now=$(swap_mb)
  if [ $((now - start)) -gt "$limit" ]; then
    echo "guarded.sh: swap grew from ${start} MB to ${now} MB; stopping $*" >&2
    kill -TERM -"$pid" 2>/dev/null
    sleep 5
    kill -KILL -"$pid" 2>/dev/null
    exit 137
  fi
  sleep 5
done
wait "$pid"
