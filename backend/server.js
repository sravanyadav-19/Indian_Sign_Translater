const express = require('express');
const path = require('path');
const cors = require('cors');
const { spawn } = require('child_process');

const app = express();
const port = process.env.PORT || 7240;
// Python interpreter name differs across systems (Windows: python, Linux/Mac: python3).
const PYTHON = process.env.PYTHON_BIN || 'python3';
// A model process must finish within a bounded time so one bad frame cannot
// keep a browser request open forever. Override for slower local machines.
const PREDICTION_TIMEOUT_MS = Number(process.env.PREDICTION_TIMEOUT_MS || 10000);
const MAX_IMAGE_DATA_LENGTH = 5 * 1024 * 1024; // Protect the inference process from oversized frames.

// Keep local development convenient while requiring an explicit allowlist in production.
const configuredOrigins = (process.env.ALLOWED_ORIGINS || '')
    .split(',')
    .map((origin) => origin.trim())
    .filter(Boolean);
const allowedOrigins = new Set(configuredOrigins);
if (process.env.NODE_ENV !== 'production') {
    allowedOrigins.add(`http://localhost:${port}`);
    allowedOrigins.add(`http://127.0.0.1:${port}`);
}

app.use(cors({
    origin(origin, callback) {
        // Non-browser tools such as curl do not send an Origin header.
        if (!origin || allowedOrigins.has(origin)) return callback(null, true);
        return callback(new Error('Origin is not allowed by the ISL API'));
    },
}));
app.use(express.json({ limit: '10mb' }));

// Serve the learning-hub UI and the sign videos.
const FRONTEND_DIR = path.join(__dirname, '..', 'frontend');
const PUBLIC_DIR = path.join(__dirname, 'public');
app.use(express.static(PUBLIC_DIR));    // /signs/*.mp4
app.use(express.static(FRONTEND_DIR));  // /index.html

app.get('/', (req, res) => res.sendFile(path.join(FRONTEND_DIR, 'index.html')));
app.get('/health', (req, res) => {
    const modelPath = path.join(__dirname, 'ai', 'isl_model.h5');
    const labelMapPath = path.join(__dirname, 'ai', 'label_map.json');
    const modelReady = require('fs').existsSync(modelPath);
    const labelsReady = require('fs').existsSync(labelMapPath);

    res.status(modelReady && labelsReady ? 200 : 503).json({
        status: modelReady && labelsReady ? 'ok' : 'degraded',
        model_ready: modelReady,
        labels_ready: labelsReady,
    });
});

app.post('/predict', (req, res) => {
    const { image } = req.body;
    if (!image) {
        return res.status(400).json({ error: 'No image data provided' });
    }
    if (typeof image !== 'string' || !image.startsWith('data:image/')) {
        return res.status(400).json({ error: 'Image must be a valid data URL' });
    }
    if (image.length > MAX_IMAGE_DATA_LENGTH) {
        return res.status(413).json({ error: 'Image data is too large' });
    }

    const scriptPath = path.join(__dirname, 'ai', 'isl_predict.py');
    const pythonProcess = spawn(PYTHON, [scriptPath], { cwd: __dirname });
    let settled = false;
    let predictionData = '';
    let errorData = '';

    const timeout = setTimeout(() => {
        pythonProcess.kill();
        if (!settled) {
            settled = true;
            res.status(504).json({ error: 'Prediction timed out' });
        }
    }, PREDICTION_TIMEOUT_MS);

    pythonProcess.stdout.on('data', (data) => {
        predictionData += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
        errorData += data.toString();
    });

    pythonProcess.on('error', (error) => {
        clearTimeout(timeout);
        if (settled) return;
        settled = true;
        console.error(`Unable to start prediction process: ${error.message}`);
        res.status(500).json({ error: 'Prediction service unavailable' });
    });

    pythonProcess.on('close', (code) => {
        clearTimeout(timeout);
        if (settled) return;
        settled = true;
        if (code !== 0 || errorData) {
            console.error(`Python script error: ${errorData}`);
            return res.status(500).json({ error: 'Prediction failed' });
        }
        try {
            const result = JSON.parse(predictionData.trim());
            res.json({
                prediction: result.prediction || '',
                confidence: Number(result.confidence) || 0,
            });
        } catch (parseError) {
            console.error(`Invalid prediction response: ${parseError.message}`);
            res.status(500).json({ error: 'Invalid prediction response' });
        }
    });

    pythonProcess.stdin.write(image);
    pythonProcess.stdin.end();
});

app.listen(port, () => {
    console.log(`ISL server running at http://localhost:${port}`);
});
