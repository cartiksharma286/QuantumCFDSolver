FROM python:3.9-slim

WORKDIR /app

# Install system dependencies if any (none major needed for this setup)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy local code to the container image.
COPY QuantumCFD/ /app/QuantumCFD/

# Install python dependencies
RUN pip install --no-cache-dir numpy matplotlib qiskit

# Set entrypoint to run the main script
# Runs a default simulation
ENTRYPOINT ["python3", "QuantumCFD/main.py"]
CMD ["--steps", "100", "--nx", "64", "--ny", "64"]
