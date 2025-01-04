#!/bin/bash

# Get the current working directory
WORKDIR=$(pwd)

# Start the server in a new terminal
gnome-terminal -- bash -c "cd $WORKDIR; source .venv/bin/activate; python3 run_server.py; exec bash"

# Wait 1 second to ensure the server starts
sleep 1

# Start the first node in a new terminal
gnome-terminal -- bash -c "cd $WORKDIR; source .venv/bin/activate; python3 run_node.py; exec bash"

# Wait 2 seconds before starting the second node
sleep 2
gnome-terminal -- bash -c "cd $WORKDIR; source .venv/bin/activate; python3 run_node.py; exec bash"

# Wait 2 seconds before starting the third node
sleep 2
gnome-terminal -- bash -c "cd $WORKDIR; source .venv/bin/activate; python3 run_node.py; exec bash"
