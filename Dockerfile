# Base image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install dependencies (Assumes you have a requirements.txt)
RUN pip install --no-cache-dir -r requirements-server.txt

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
