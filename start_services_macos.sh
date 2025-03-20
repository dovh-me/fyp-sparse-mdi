#!/bin/bash

# Get the current working directory
WORKDIR=$(pwd)

# Array to store process IDs (PIDs)
PIDS=()

# Function to clean up processes
cleanup() {
    echo "Terminating all running instances of run_server.py and run_node.py..."
    for PID in "${PIDS[@]}"; do
        kill "$PID" 2>/dev/null || true
    done
    exit 0
}

# Trap SIGINT (Ctrl+C) and call the cleanup function
trap cleanup SIGINT

# Activate the virtual environment
source .venv/bin/activate

# Start the server and capture its PID
python3 "$WORKDIR/run_server.py" &
PIDS+=($!)

# Wait 1 second to ensure the server starts
sleep 1

# Start the first node and capture its PID
python3 "$WORKDIR/run_node.py" &
PIDS+=($!)

# Wait 2 seconds before starting the second node
sleep 2
python3 "$WORKDIR/run_node.py" &
PIDS+=($!)

# Wait 2 seconds before starting the third node
sleep 2
python3 "$WORKDIR/run_node.py" &
PIDS+=($!)

# Wait 2 seconds before starting the third node
sleep 2
python3 "$WORKDIR/run_node.py" &
PIDS+=($!)

# Wait 2 seconds before starting the third node
sleep 2
python3 "$WORKDIR/run_node.py" &
PIDS+=($!)

# Wait 2 seconds before starting the third node
sleep 2
python3 "$WORKDIR/run_node.py" &
PIDS+=($!)

# Wait 2 seconds before starting the third node
sleep 2
python3 "$WORKDIR/run_node.py" &
PIDS+=($!)

# Wait indefinitely to allow the user to cancel the script
while true; do
    sleep 1
done
