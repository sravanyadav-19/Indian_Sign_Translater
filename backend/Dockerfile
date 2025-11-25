# Use Ubuntu-based Python image so apt installs work correctly
FROM python:3.11-bullseye

WORKDIR /app

# Install dependencies including Node.js and FFmpeg
RUN apt-get update && \
    apt-get install -y curl ffmpeg libsm6 libxext6 && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy remaining backend code
COPY . .

# Install Node dependencies for server.js
RUN npm install

EXPOSE 3000

# Run the backend server
CMD ["node", "server.js"]
