const express = require('express');
const path = require('path');
const cors = require('cors');
const { spawn } = require('child_process');

const app = express();
const port = process.env.PORT || 7240;
// Python interpreter name differs across systems (Windows: python, Linux/Mac: python3).
const PYTHON = process.env.PYTHON_BIN || 'python3';

// Allow the page (whether served by us or opened separately) to call the API.
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// Serve the learning-hub UI and the sign videos.
const FRONTEND_DIR = path.join(__dirname, '..', 'frontend');
const PUBLIC_DIR = path.join(__dirname, 'public');
app.use(express.static(PUBLIC_DIR));    // /signs/*.mp4
app.use(express.static(FRONTEND_DIR));  // /index.html

app.get('/', (req, res) => res.sendFile(path.join(FRONTEND_DIR, 'index.html')));
app.get('/health', (req, res) => res.json({ status: 'ok' }));

app.post('/predict', (req, res) => {
    const { image } = req.body;
    if (!image) {
        return res.status(400).json({ error: 'No image data provided' });
    }

    const scriptPath = path.join(__dirname, 'ai', 'isl_predict.py');
    const pythonProcess = spawn(PYTHON, [scriptPath], { cwd: __dirname });

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

app.listen(port, () => {
    console.log(`ISL server running at http://localhost:${port}`);
});
