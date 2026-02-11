#!/bin/bash
# Start Redis server bound to a specific network interface
# Usage: ./start_redis.sh [bind_address] [port]

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# This should be set in env.sh
# REDIS_STABLE="${SCRIPT_DIR}/redis-stable"

# Get bind address (default to compute node network)
# On Aurora, you'll want the HSN (high-speed network) address
BIND_ADDRESS=${1:-$(getent hosts $(hostname).hsn.cm.aurora.alcf.anl.gov | awk '{ print $1 }' | head -n 1)}
REDIS_PORT=${2:-6379}

# If BIND_ADDRESS is still empty, fall back to primary IP
if [ -z "$BIND_ADDRESS" ]; then
    BIND_ADDRESS=$(hostname -i | awk '{print $1}')
fi

echo "$(date) Starting Redis server"
echo "$(date)   Bind address: ${BIND_ADDRESS}"
echo "$(date)   Port: ${REDIS_PORT}"

# Start Redis with command-line overrides
# This overrides the config file settings
${REDIS_STABLE}/src/redis-server ${REDIS_STABLE}/redis.conf \
    --bind ${BIND_ADDRESS} \
    --port ${REDIS_PORT} \
    --protected-mode no \
    --daemonize yes \
    --logfile ${SCRIPT_DIR}/redis-server.log \
    --pidfile ${SCRIPT_DIR}/redis-server.pid

# Wait a moment and check if it started
sleep 2

if [ -f "${SCRIPT_DIR}/redis-server.pid" ]; then
    PID=$(cat ${SCRIPT_DIR}/redis-server.pid)
    if kill -0 $PID 2>/dev/null; then
        echo "$(date) Redis server started successfully (PID: $PID)"
        echo "$(date) Connect with: redis-cli -h ${BIND_ADDRESS} -p ${REDIS_PORT}"
        echo "$(date) Logs at: ${SCRIPT_DIR}/redis-server.log"
        exit 0
    else
        echo "$(date) ERROR: Redis server failed to start"
        cat ${SCRIPT_DIR}/redis-server.log
        exit 1
    fi
else
    echo "$(date) ERROR: Redis server PID file not created"
    exit 1
fi
