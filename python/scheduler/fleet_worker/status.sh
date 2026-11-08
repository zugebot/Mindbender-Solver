#!/bin/bash
# python/scheduler/fleet_worker/status.sh: what mbq reads from a fleet machine on each poll, one
# fact per line:
#   setup ok|failed|running|absent
#   live NAME              a unit whose process is running
#   dead NAME              a unit whose process is gone without an exit code
#   rc NAME CODE           a finished unit (not yet collected)
#   progress NAME JSON     the running unit's progress file
#   info KEY=VALUE         setup.ok's machine facts
cd "$(dirname "$0")"
if [ -e setup.ok ]; then echo "setup ok"; sed 's/^/info /' setup.ok
elif [ -e setup.failed ]; then echo "setup failed"; sed 's/^/info reason=/' setup.failed
elif [ -e setup.log ]; then echo "setup running"
else echo "setup absent"; fi
[ -d work ] || exit 0
for f in work/*.pid; do
    [ -e "$f" ] || continue
    name=$(basename "$f" .pid)
    if kill -0 "$(cat "$f")" 2>/dev/null; then
        echo "live $name"
        [ -e "work/$name.progress.json" ] && echo "progress $name $(tr -d '\n' < "work/$name.progress.json")"
    elif [ ! -e "work/$name.rc" ]; then
        echo "dead $name"
    fi
done
for f in work/*.rc; do
    [ -e "$f" ] || continue
    echo "rc $(basename "$f" .rc) $(cat "$f")"
done
