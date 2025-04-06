# Base image
FROM python:3.14-rc-alpine3.20

# Install iproute2 for tc command
RUN apt-get update && apt-get install -y iproute2 && apt-get clean

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Ensure the latency script is executable
RUN chmod +x setup_tc.sh

# Install dependencies (Assumes you have a requirements.txt)
RUN pip install -r requirements-server.txt

# Expose any necessary ports (if required)
## Adjust based on your application
EXPOSE 5000  
EXPOSE 50052
EXPOSE 50053
EXPOSE 50054
EXPOSE 50055
EXPOSE 4010

# Default command (this will be overridden in docker-compose)
CMD ["python3", "run_server.py"]
