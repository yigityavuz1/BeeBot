#!/bin/bash
set -e

# Wait for Weaviate to be ready
echo "Waiting for Weaviate to be ready..."
timeout=60
counter=0
while ! curl -s http://${WEAVIATE_HOST:-weaviate}:${WEAVIATE_PORT:-8080}/v1/.well-known/ready > /dev/null; do
    if [ $counter -gt $timeout ]; then
        echo "Timed out waiting for Weaviate to be ready"
        exit 1
    fi
    counter=$((counter + 1))
    echo "Waiting for Weaviate... ($counter/$timeout)"
    sleep 1
done

echo "Weaviate is ready! Starting BeeBot..."

# Run the specified command
exec "$@"