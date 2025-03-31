#!/bin/bash

# Define session name
SESSION_NAME="script-runner"

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null || true

# Create a new tmux session
tmux new-session -d -s $SESSION_NAME

# Split window horizontally
tmux split-window -h -t $SESSION_NAME:0

# Function to run the cycle
run_cycle() {
  local cycle=$1
  
  # Send commands to the left pane (shell script)
  tmux send-keys -t $SESSION_NAME:0.0 "echo 'Cycle $cycle: Starting shell script...'" C-m
  tmux send-keys -t $SESSION_NAME:0.0 "./start_services_macos.sh" C-m
  
  # Wait 10 seconds
  sleep 10
  
  # Send commands to the right pane (Python script with venv activation)
  tmux send-keys -t $SESSION_NAME:0.1 "echo 'Cycle $cycle: Activating virtual environment and starting Python script...'" C-m
  tmux send-keys -t $SESSION_NAME:0.1 "source .venv/bin/activate && python run_inference_resnet50.py" C-m
  
  # Wait for Python script to complete (this is a simplified approach)
  # Adjust the sleep time based on how long your Python script typically runs
  sleep 90  # Assuming Python script takes less than 10 seconds
  
  # Kill the shell script
  tmux send-keys -t $SESSION_NAME:0.0 C-c
  sleep 0.5
  
  # Clear both panes for the next cycle
  tmux send-keys -t $SESSION_NAME:0.0 "clear" C-m
  tmux send-keys -t $SESSION_NAME:0.1 "clear" C-m
}

# Run 5 cycles
for cycle in {1..5}; do
  echo "Starting cycle $cycle"
  run_cycle $cycle
  
  # Small delay between cycles
  sleep 1
done

# Attach to the tmux session to see the results
tmux attach-session -t $SESSION_NAME
