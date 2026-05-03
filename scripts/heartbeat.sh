#!/usr/bin/env bash
# Bulletproof D19 heartbeat. Emits every 300 s.
#  - tolerates missing / unreadable log
#  - detects training process death and yells
#  - self-recovers from transient failures (any error -> emit warning, keep going)

LOG=/c/dev/vocos_rt/checkpoints/finetune/training.log

while true; do
    ts=$(date +%H:%M:%S 2>/dev/null || echo "??:??:??")

    # check training process alive
    py_count=$(tasklist 2>/dev/null | grep -ci 'python\.exe' || echo 0)
    if [ "$py_count" -lt 1 ]; then
        echo "[$ts] HEARTBEAT ALERT: NO python.exe processes detected -- training likely died"
        sleep 300 2>/dev/null || true
        continue
    fi

    # read last training log line
    last_line=$(tail -1 "$LOG" 2>/dev/null || echo "")
    if [ -z "$last_line" ]; then
        echo "[$ts] HEARTBEAT WARNING: log unreadable; py procs alive=$py_count"
    else
        # strip log timestamp + module info, prepend our wall-clock
        clean=$(echo "$last_line" | sed "s|^.*\[__main__:[0-9]*\] |[$ts] |")
        echo "$clean"
    fi

    # sleep 300s, never let sleep failure kill the loop
    sleep 300 2>/dev/null || sleep 300 2>/dev/null || sleep 60 2>/dev/null || true
done
