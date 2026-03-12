#!/usr/bin/env bash

source venv/bin/activate
python -m http.server 8080 --directory Frontend/ --bind 127.0.0.1 &
PID1=$!

echo "App lounched on http://localhost:8080"
python -m uvicorn main:app --host 127.0.0.1 --port 2137 &
PID2=$!

trap "kill $PID1 $PID2" EXIT INT TERM

wait