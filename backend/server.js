const express = require('express');
const path = require('path');
const { spawn } = require('child_process');

const app = express();
const port = 7240;

app.use(express.static('public'));
app.use(express.json({ limit: '10mb' }));

app.post('/predict', (req, res) => {
    const { image } = req.body;
    if (!image) {
        return res.status(400).json({ error: 'No image data provided' });
    }

    const pythonProcess = spawn('python', ['ai/isl_predict.py']);

    let predictionData = '';
    let errorData = '';

    pythonProcess.stdout.on('data', (data) => {
        predictionData += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
        errorData += data.toString();
    });

    pythonProcess.on('close', (code) => {
        if (code !== 0 || errorData) {
            console.error(`Python script error: ${errorData}`);
            return res.status(500).json({ error: 'Prediction failed', details: errorData });
        }
        res.json({ prediction: predictionData.trim() });
    });

    pythonProcess.stdin.write(image);
    pythonProcess.stdin.end();
});
app.get('/', (req, res) => res.send("ISL Backend Running"));

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
