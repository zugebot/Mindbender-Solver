#!/bin/bash
# python/scheduler/fleet_worker/run_unit.sh NAME SOLVER-ARGS...: runs one fleet work unit on a
# rented machine, from ~/mbfleet. mbq starts it detached (setsid nohup) and polls the files:
#   work/NAME.log            the solver's output
#   work/NAME.progress.json  its progress (--progress-file)
#   work/NAME.mid            the unit's middles (written only when the unit is complete)
#   work/NAME.rc             the exit code, written last
#   work/NAME.pid            while it runs
# One unit at a time per machine: a second start while one runs exits with code 75.
cd "$(dirname "$0")"
ROOT=$(pwd)
name=$1
shift
mkdir -p work
exec 9> work/.unit.lock
flock -n 9 || exit 75
rm -f "work/$name.rc"
echo $$ > "work/$name.pid"
"$ROOT/bin/experimentDiskDepth6" "$@" --out "$ROOT/work/tables" --middles-out "$ROOT/work/$name.mid" --no-recover \
    --progress-file "$ROOT/work/$name.progress.json" --progress-interval 10 > "work/$name.log" 2>&1
rc=$?
rm -f "work/$name.pid"
echo $rc > "work/$name.rc.part" && mv "work/$name.rc.part" "work/$name.rc"
exit $rc
