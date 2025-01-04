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

# Start the server in a new terminal and capture its PID
osascript <<EOF
tell application "Terminal"
    do script "cd $WORKDIR; source .venv/bin/activate; python3 run_server.py"
end tell
EOF
sleep 1 # Wait for the terminal to start the process

# Start the first node in a new terminal and capture its PID
osascript <<EOF
tell application "Terminal"
    do script "cd $WORKDIR; source .venv/bin/activate; python3 run_node.py"
end tell
EOF
sleep 2 # Wait before starting the next node

# Start the second node in a new terminal and capture its PID
osascript <<EOF
tell application "Terminal"
    do script "cd $WORKDIR; source .venv/bin/activate; python3 run_node.py"
end tell
EOF
sleep 2 # Wait before starting the next node

# Start the third node in a new terminal and capture its PID
osascript <<EOF
tell application "Terminal"
    do script "cd $WORKDIR; source .venv/bin/activate; python3 run_node.py"
end tell
EOF

# Wait indefinitely to allow user to cancel the script
while true; do
    sleep 1
done
