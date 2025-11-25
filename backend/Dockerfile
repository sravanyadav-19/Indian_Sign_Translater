FROM python:3.11

WORKDIR /app

RUN apt-get update && apt-get install -y nodejs ffmpeg libsm6 libxext6

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN npm install

CMD ["node", "server.js"]
