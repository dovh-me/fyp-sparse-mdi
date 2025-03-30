#!/bin/sh

# Add latency of 100ms with 10ms jitter
tc qdisc add dev eth0 root netem delay 100ms 10ms

# Keep running the main process
exec "$@"
