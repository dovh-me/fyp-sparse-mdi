#!/bin/bash

# Ensure the script is executed with Bash
if [ -z "$BASH_VERSION" ]; then
	    echo "This script must be run with Bash."
	        exit 1
fi

# Get the current working directory
WORKDIR=$(pwd)

# Array to store process IDs (PIDs)
PIDS=()

# Function to clean up processes
cleanup() {
	    echo "Terminating all running instances of run_server.py and run_node.py..."
	        for PID in "${PIDS[@]}"; do
			        if kill -0 "$PID" 2>/dev/null; then
					            kill "$PID" 2>/dev/null || true
						            fi
							        done
								    echo "All processes terminated."
								        exit 0
								}

							# Trap SIGINT (Ctrl+C) and call the cleanup function
							trap cleanup SIGINT

							# Check if Python 3 and virtualenv are installed
							if ! command -v python3 &>/dev/null; then
								    echo "Python 3 is not installed. Install it with: sudo apt install python3"
								        exit 1
							fi

							if ! command -v virtualenv &>/dev/null; then
								    echo "virtualenv is not installed. Install it with: sudo apt install python3-virtualenv"
								        exit 1
							fi

							# Activate the virtual environment
							if [ -d ".venv" ]; then
								    source .venv/bin/activate
							    elif [ -d "$HOME/.venv" ]; then
								        source "$HOME/.venv/bin/activate"
								else
									    echo "No virtual environment found. Create one with: python3 -m venv .venv"
									        exit 1
							fi

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

							# Display PIDs for debugging purposes
							echo "Started processes with PIDs: ${PIDS[*]}"

							# Wait indefinitely to allow the user to cancel the script
							while true; do
								    sleep 1
							    done

