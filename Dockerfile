FROM python:3.12.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create data directories
RUN mkdir -p /app/data/raw /app/data/processed /app/data/pdf_text

# Set Python to run in unbuffered mode to ensure logs are visible immediately
#ENV PYTHONUNBUFFERED=1

# Copy the application code
COPY . .

# Make entrypoint script executable
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose the port that Streamlit runs on
EXPOSE 8501

# Set entrypoint to wait for Weaviate
ENTRYPOINT ["/entrypoint.sh"]

# Command to run the application with hot reloading enabled
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.runOnSave=true"]