#!/bin/bash
set -e

echo "$(date) - BeeBot startup script initialized"

# Wait for Weaviate to be ready
echo "$(date) - Waiting for Weaviate to be ready..."
timeout=60
counter=0
while ! curl -s http://${WEAVIATE_HOST:-weaviate}:${WEAVIATE_PORT:-8080}/v1/.well-known/ready > /dev/null; do
    if [ $counter -gt $timeout ]; then
        echo "$(date) - ERROR: Timed out waiting for Weaviate to be ready"
        exit 1
    fi
    counter=$((counter + 1))
    echo "$(date) - Waiting for Weaviate... ($counter/$timeout)"
    sleep 1
done

echo "$(date) - Weaviate is ready! Starting BeeBot..."

# Set Python unbuffered output to ensure logs are immediately visible
export PYTHONUNBUFFERED=1

# Configure logging level
export LOGLEVEL=${LOGLEVEL:-INFO}
echo "$(date) - Setting Python log level to $LOGLEVEL"

# Print environment details (excluding sensitive values)
echo "$(date) - Starting BeeBot with configuration:"
echo "WEAVIATE_HOST: ${WEAVIATE_HOST:-weaviate}"
echo "WEAVIATE_PORT: ${WEAVIATE_PORT:-8080}"
echo "RAW_DATA_DIR: ${RAW_DATA_DIR:-data/raw}"
echo "PROCESSED_DATA_DIR: ${PROCESSED_DATA_DIR:-data/processed}"
echo "PDF_TEXT_DIR: ${PDF_TEXT_DIR:-data/pdf_text}"

# Run the specified command
echo "$(date) - Executing: $@"
exec "$@"