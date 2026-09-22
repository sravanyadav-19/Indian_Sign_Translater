# Use Ubuntu-based Python image so apt installs work correctly
FROM python:3.11-bullseye

WORKDIR /app

# Install dependencies including Node.js and FFmpeg
RUN apt-get update && \
    apt-get install -y curl ffmpeg libsm6 libxext6 && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs

# Install Python dependencies without retaining pip's download cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Node dependencies from the lockfile for reproducible deploys.
COPY package*.json ./
RUN npm ci --omit=dev

# Copy remaining backend code, including the prediction scripts and frontend.
COPY . .

ENV NODE_ENV=production
ENV PYTHON_BIN=python

EXPOSE 7240

# Run the backend server
CMD ["node", "server.js"]
