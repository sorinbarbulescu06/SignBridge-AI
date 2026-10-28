import os
import sys
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

# Adăugăm folderul 'src' în path pentru a importa direct modulele tale fără să le modificăm
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if os.path.exists(SRC_DIR):
    sys.path.insert(0, SRC_DIR)
else:
    sys.path.insert(0, BASE_DIR)

from translation.sign_translator import SignTranslator

app = FastAPI(title="AI Sign Language Hub - Web Edition")

# Verificăm la pornire ce cuvinte are învățate modelul tău din data/sign_model.pkl
_startup_translator = SignTranslator()
LEARNED_CLASSES = _startup_translator.classes


@app.get("/api/info")
async def get_model_info():
    return {
        "model_loaded": _startup_translator.model is not None,
        "classes": LEARNED_CLASSES,
    }


@app.websocket("/ws/translate")
async def websocket_translate(websocket: WebSocket):
    await websocket.accept()
    # Fiecare utilizator primește propria instanță SignTranslator (fereastră deque izolată)
    translator = SignTranslator()
    try:
        while True:
            sign_input_template = await websocket.receive_json()
            translated_word = translator.translate(sign_input_template)
            if translated_word:
                await websocket.send_json({"word": translated_word})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS Error] {e}")


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Sign Language Translator</title>
    <!-- MediaPipe Hands JS pentru rulare 60 FPS direct în browser fără lag de rețea -->
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js" crossorigin="anonymous"></script>
    <style>
        :root {
            --bg-main: #181818;
            --bg-panel: #242424;
            --bg-viewport: #1a1a1a;
            --blue: #3B8ED0;
            --blue-hover: #1F6AA5;
            --red: #c93b2b;
            --red-hover: #9e2a1b;
            --gray-btn: #4a4a4a;
            --text: #f0f0f0;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', system-ui, sans-serif; }
        body { background: var(--bg-main); color: var(--text); height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
        
        /* Header identic cu SignLanguageApp */
        header {
            background: var(--bg-panel);
            padding: 14px 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
            border-bottom: 1px solid #333;
        }
        header h1 { font-size: 22px; font-weight: 700; }
        .segmented-btn {
            display: flex;
            background: #1a1a1a;
            border-radius: 8px;
            padding: 4px;
            width: 340px;
        }
        .segmented-btn button {
            flex: 1;
            background: transparent;
            border: none;
            color: #aaa;
            padding: 8px 0;
            font-size: 14px;
            font-weight: 700;
            border-radius: 6px;
            cursor: pointer;
            transition: 0.2s;
        }
        .segmented-btn button.active {
            background: var(--blue);
            color: white;
        }

        /* Grid Layout */
        .mode-container {
            display: none;
            flex: 1;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            padding: 20px;
            min-height: 0;
        }
        .mode-container.active { display: grid; }

        .panel {
            background: var(--bg-panel);
            border-radius: 15px;
            padding: 18px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            min-height: 0;
        }
        .panel-title { text-align: center; font-size: 16px; font-weight: 700; letter-spacing: 0.5px; }
        
        .toolbar { display: flex; align-items: center; gap: 10px; }
        .toolbar label { font-size: 13px; color: #ccc; white-space: nowrap; }
        .toolbar select {
            flex: 1;
            background: #333;
            color: white;
            border: 1px solid #444;
            border-radius: 6px;
            padding: 6px 10px;
            font-size: 13px;
        }

        .viewport {
            flex: 1;
            background: var(--bg-viewport);
            border-radius: 10px;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            min-height: 0;
        }
        .viewport canvas {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }
        .placeholder-text {
            position: absolute;
            text-align: center;
            color: #aaa;
            font-size: 14px;
            line-height: 1.6;
            pointer-events: none;
        }

        .btn-row { display: flex; justify-content: center; gap: 16px; padding-top: 4px; }
        .btn {
            background: var(--blue);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 9px 22px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: 0.15s;
        }
        .btn:hover { background: var(--blue-hover); }
        .btn-red { background: var(--red); }
        .btn-red:hover { background: var(--red-hover); }
        .btn-secondary { background: #2b5b84; }
        .btn-gray { background: var(--gray-btn); }
        .btn-gray:hover { background: #333; }

        textarea.output-box {
            flex: 1;
            background: var(--bg-viewport);
            color: #fff;
            border: 1px solid #333;
            border-radius: 10px;
            padding: 15px;
            font-size: 16px;
            resize: none;
            line-height: 1.5;
        }
        .classes-badge {
            font-size: 12px;
            color: #999;
            text-align: center;
        }
        #speechParagraphs {
            padding: 24px;
            font-size: 19px;
            font-weight: 600;
            text-align: center;
            color: #eaeaea;
            white-space: pre-wrap;
            overflow-y: auto;
            max-height: 100%;
        }
        .status-italic {
            text-align: center;
            font-size: 14px;
            font-style: italic;
            color: #bbb;
        }
        @media (max-width: 850px) {
            body { overflow: auto; height: auto; }
            .mode-container { grid-template-columns: 1fr; }
            .viewport { min-height: 300px; }
        }
    </style>
</head>
<body>

<header>
    <h1>AI Sign Language Hub</h1>
    <div class="segmented-btn">
        <button id="tabSignToVoice" class="active" onclick="switchMode('s2v')">Sign -&gt; Voice</button>
        <button id="tabVoiceToSign" onclick="switchMode('v2s')">Voice -&gt; Sign</button>
    </div>
</header>

<!-- MODUL 1: SIGN -> VOICE -->
<div id="modeS2V" class="mode-container active">
    <div class="panel">
        <div class="panel-title">SIGN INPUT SOURCE</div>
        <div class="toolbar">
            <label>Camera:</label>
            <select id="cameraSelect"></select>
            <button class="btn btn-gray" style="padding: 5px 10px;" onclick="loadDevices()">⟳</button>
        </div>
        <div class="viewport">
            <video id="hiddenVideo" playsinline style="display:none;"></video>
            <canvas id="videoCanvas" width="640" height="480"></canvas>
            <div id="videoPlaceholder" class="placeholder-text">[ Camera / Video Feed ]<br><br>Waiting for input...</div>
        </div>
        <div class="btn-row">
            <button id="btnCamera" class="btn" onclick="toggleCamera()">Open Camera</button>
            <button class="btn btn-secondary" onclick="document.getElementById('videoFileInput').click()">Upload Video File</button>
            <input type="file" id="videoFileInput" accept="video/*" style="display:none;" onchange="loadVideoFile(event)">
        </div>
    </div>

    <div class="panel">
        <div class="panel-title">OUTPUT TEXT &amp; VOICE</div>
        <div class="toolbar">
            <label>Audio Out:</label>
            <select id="voiceOutputToggle">
                <option value="on">[0] Browser Text-to-Speech (Active)</option>
                <option value="off">[1] Mute Voice Output</option>
            </select>
        </div>
        <textarea id="translatedOutput" class="output-box" readonly>Translated text will appear here&#10;</textarea>
        <div id="learnedClassesInfo" class="classes-badge">Loading AI model classes...</div>
        <div class="btn-row">
            <button class="btn btn-gray" onclick="clearTranslatedText()">Delete Text</button>
        </div>
    </div>
</div>

<!-- MODUL 2: VOICE -> SIGN -->
<div id="modeV2S" class="mode-container">
    <div class="panel">
        <div class="panel-title">VOICE INPUT SOURCE</div>
        <div class="toolbar">
            <label>Language:</label>
            <select id="sttLangSelect">
                <option value="en-US">English (en-US)</option>
                <option value="ro-RO">Română (ro-RO)</option>
            </select>
        </div>
        <div class="viewport">
            <canvas id="waveCanvas" width="600" height="360"></canvas>
            <div id="audioPlaceholder" class="placeholder-text">[ Status Audio / Waveform ]<br><br>Waiting for input...</div>
        </div>
        <div class="btn-row">
            <button id="btnMic" class="btn btn-red" onclick="toggleMic()">Use Microphone</button>
        </div>
    </div>

    <div class="panel">
        <div class="panel-title">OUTPUT 3D MODEL</div>
        <div class="viewport">
            <div id="speechParagraphs">[ Viewport Model 3D ]<br><br>Aspect-Ratio Locked Ready</div>
        </div>
        <div id="recognizedVoiceLabel" class="status-italic">No Word Detected</div>
    </div>
</div>

<script>
    // ==========================================
    // 1. CONEXIUNE WEBSOCKET CU BACKEND-UL PYTHON
    // ==========================================
    let ws = null;
    function connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${protocol}//${window.location.host}/ws/translate`);
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.word) {
                const box = document.getElementById('translatedOutput');
                box.value += data.word + "\\n";
                box.scrollTop = box.scrollHeight;

                // Rostim cuvântul tradus dacă Audio Out este activ
                if (document.getElementById('voiceOutputToggle').value === 'on' && 'speechSynthesis' in window) {
                    const cleanWord = data.word.split('(')[0].trim();
                    const utter = new SpeechSynthesisUtterance(cleanWord);
                    window.speechSynthesis.speak(utter);
                }
            }
        };
        ws.onclose = () => setTimeout(connectWebSocket, 1500);
    }
    connectWebSocket();

    fetch('/api/info').then(r => r.json()).then(info => {
        const badge = document.getElementById('learnedClassesInfo');
        if (info.classes && info.classes.length > 0) {
            badge.textContent = `Active AI Signs (${info.classes.length}): ${info.classes.join(', ')}`;
        } else {
            badge.textContent = 'Model status: Ready';
        }
    });

    // ==========================================
    // 2. MEDIAPIPE HANDS LA 60 FPS ÎN BROWSER
    // ==========================================
    const videoEl = document.getElementById('hiddenVideo');
    const canvasEl = document.getElementById('videoCanvas');
    const ctx = canvasEl.getContext('2d');
    let cameraRunning = false;
    let isLiveCamera = true;
    let animFrameId = null;
    let lastInferenceTime = 0;

    const hands = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });
    hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 0,
        minDetectionConfidence: 0.6,
        minTrackingConfidence: 0.6,
        selfieMode: true // Oglindește exact ca cv2.flip(frame, 1) din VideoEngine
    });

    hands.onResults((results) => {
        if (!cameraRunning) return;
        canvasEl.width = results.image.width || 640;
        canvasEl.height = results.image.height || 480;

        ctx.save();
        ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
        ctx.drawImage(results.image, 0, 0, canvasEl.width, canvasEl.height);

        const signTemplate = { hands_detected: 0, hands: [] };

        if (results.multiHandLandmarks && results.multiHandedness) {
            signTemplate.hands_detected = results.multiHandLandmarks.length;
            for (let i = 0; i < results.multiHandLandmarks.length; i++) {
                const landmarks = results.multiHandLandmarks[i];
                const handedness = results.multiHandedness[i].label;

                drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {color: '#00FF00', lineWidth: 3});
                drawLandmarks(ctx, landmarks, {color: '#FF0000', lineWidth: 1, radius: 3});

                signTemplate.hands.push({
                    type: handedness,
                    landmarks: landmarks.map((lm, idx) => ({
                        id: idx,
                        x: lm.x,
                        y: lm.y,
                        z: lm.z
                    }))
                });
            }
        }
        ctx.restore();

        // Trimitem template-ul la fiecare ~30ms (exact ca self.after(30) din aplicația desktop)
        const now = performance.now();
        if (now - lastInferenceTime >= 30 && ws && ws.readyState === WebSocket.OPEN) {
            lastInferenceTime = now;
            ws.send(JSON.stringify(signTemplate));
        }
    });

    async function processVideoLoop() {
        if (!cameraRunning) return;
        if (videoEl.readyState >= 2 && !videoEl.paused && !videoEl.ended) {
            await hands.send({ image: videoEl });
        } else if (videoEl.ended && !isLiveCamera) {
            stopCameraFeed("[ Video Finished ]");
            return;
        }
        animFrameId = requestAnimationFrame(processVideoLoop);
    }

    async function toggleCamera() {
        if (cameraRunning) {
            stopCameraFeed("[ Camera / Video Feed ]\\n\\nCamera stopped.");
        } else {
            try {
                const deviceId = document.getElementById('cameraSelect').value;
                const constraints = {
                    video: deviceId ? { deviceId: { exact: deviceId }, width: 640, height: 480 } : { width: 640, height: 480 }
                };
                const stream = await navigator.mediaDevices.getUserMedia(constraints);
                stopCameraFeed();
                isLiveCamera = true;
                hands.setOptions({ selfieMode: true });
                videoEl.srcObject = stream;
                await videoEl.play();

                cameraRunning = true;
                document.getElementById('videoPlaceholder').style.display = 'none';
                const btn = document.getElementById('btnCamera');
                btn.textContent = 'Stop Camera';
                btn.classList.add('btn-red');
                processVideoLoop();
            } catch (err) {
                alert("Please allow camera access in your browser: " + err.message);
            }
        }
    }

    function loadVideoFile(event) {
        const file = event.target.files[0];
        if (!file) return;
        stopCameraFeed();
        isLiveCamera = false;
        hands.setOptions({ selfieMode: false });
        videoEl.srcObject = null;
        videoEl.src = URL.createObjectURL(file);
        videoEl.play().then(() => {
            cameraRunning = true;
            document.getElementById('videoPlaceholder').style.display = 'none';
            const box = document.getElementById('translatedOutput');
            box.value += `\\n[INFO] Playing video: ${file.name}...\\n`;
            processVideoLoop();
        });
    }

    function stopCameraFeed(placeholderMsg = "[ Camera / Video Feed ]<br><br>Waiting for input...") {
        cameraRunning = false;
        if (animFrameId) cancelAnimationFrame(animFrameId);
        if (videoEl.srcObject) {
            videoEl.srcObject.getTracks().forEach(t => t.stop());
            videoEl.srcObject = null;
        }
        videoEl.pause();
        ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
        const ph = document.getElementById('videoPlaceholder');
        ph.innerHTML = placeholderMsg.replace(/\\n/g, '<br>');
        ph.style.display = 'block';
        const btn = document.getElementById('btnCamera');
        btn.textContent = 'Open Camera';
        btn.classList.remove('btn-red');
    }

    function clearTranslatedText() {
        document.getElementById('translatedOutput').value = '';
    }

    // ==========================================
    // 3. VOICE -> SIGN (WAVEFORM 60 FPS + STT)
    // ==========================================
    let micRunning = false;
    let audioCtx = null;
    let analyser = null;
    let micStream = null;
    let waveAnimId = null;
    let wavePhase = 0.0;
    let recognition = null;
    let recognizedHistory = [];

    function renderWaveform() {
        if (!micRunning) return;
        const canvas = document.getElementById('waveCanvas');
        const wCtx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        const dataArray = new Float32Array(analyser.fftSize);
        analyser.getFloatTimeDomainData(dataArray);
        let sumSq = 0;
        for (let i = 0; i < dataArray.length; i++) sumSq += dataArray[i] * dataArray[i];
        const rms = Math.sqrt(sumSq / dataArray.length);
        const volume = Math.min(rms * 35.0, 1.0); // Exact formula din AudioEngine

        // Exact matematica din compute_waveform_Render_data (ui_helpers.py)
        const midY = height / 2;
        const maxAmp = (height / 2) - 15;
        const amplitude = 3 + (volume * maxAmp);
        wavePhase += 0.15 + (volume * 0.35);

        let color = "#3B8ED0";
        let lineWidth = 2;
        if (volume > 0.75) { color = "#ff3b30"; lineWidth = 4; }
        else if (volume > 0.08) { color = "#30d158"; lineWidth = 3; }

        wCtx.clearRect(0, 0, width, height);
        wCtx.beginPath();
        wCtx.strokeStyle = color;
        wCtx.lineWidth = lineWidth;

        for (let x = 0; x < width; x += 4) {
            const normX = x / width;
            const envelope = Math.sin(normX * Math.PI);
            const yOffset = Math.sin((x * 0.03) + wavePhase) * 0.7 + Math.sin((x * 0.07) - wavePhase * 1.5) * 0.3;
            const y = midY + (yOffset * amplitude * envelope);
            if (x === 0) wCtx.moveTo(x, y);
            else wCtx.lineTo(x, y);
        }
        wCtx.stroke();
        waveAnimId = requestAnimationFrame(renderWaveform);
    }

    async function toggleMic() {
        if (micRunning) {
            stopMicFeed();
        } else {
            try {
                micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                const source = audioCtx.createMediaStreamSource(micStream);
                analyser = audioCtx.createAnalyser();
                analyser.fftSize = 1024;
                source.connect(analyser);

                micRunning = true;
                document.getElementById('audioPlaceholder').style.display = 'none';
                const btn = document.getElementById('btnMic');
                btn.textContent = 'Stop Microphone';
                btn.classList.replace('btn-red', 'btn-gray');
                document.getElementById('recognizedVoiceLabel').textContent = 'Listening on Microphone...';
                renderWaveform();

                const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
                if (SpeechRec) {
                    recognition = new SpeechRec();
                    recognition.lang = document.getElementById('sttLangSelect').value;
                    recognition.continuous = true;
                    recognition.interimResults = false;
                    recognition.onresult = (e) => {
                        const transcript = e.results[e.results.length - 1][0].transcript.trim();
                        if (transcript) {
                            const formatted = transcript.charAt(0).toUpperCase() + transcript.slice(1) + '.';
                            recognizedHistory.push(formatted);
                            document.getElementById('speechParagraphs').textContent = recognizedHistory.slice(-6).join('\\n\\n');
                            document.getElementById('recognizedVoiceLabel').textContent = 'Ready / Listening...';
                        }
                    };
                    recognition.onend = () => { if (micRunning) recognition.start(); };
                    recognition.start();
                }
            } catch (err) {
                alert("Microphone access error: " + err.message);
            }
        }
    }

    function stopMicFeed() {
        micRunning = false;
        if (waveAnimId) cancelAnimationFrame(waveAnimId);
        if (recognition) { recognition.onend = null; recognition.stop(); recognition = null; }
        if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
        if (audioCtx) { audioCtx.close(); audioCtx = null; }
        const canvas = document.getElementById('waveCanvas');
        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
        document.getElementById('audioPlaceholder').style.display = 'block';
        const btn = document.getElementById('btnMic');
        btn.textContent = 'Use Microphone';
        btn.classList.replace('btn-gray', 'btn-red');
        document.getElementById('recognizedVoiceLabel').textContent = 'No Word Detected';
    }

    function switchMode(mode) {
        stopCameraFeed();
        stopMicFeed();
        document.getElementById('tabSignToVoice').classList.toggle('active', mode === 's2v');
        document.getElementById('tabVoiceToSign').classList.toggle('active', mode === 'v2s');
        document.getElementById('modeS2V').classList.toggle('active', mode === 's2v');
        document.getElementById('modeV2S').classList.toggle('active', mode === 'v2s');
    }

    async function loadDevices() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const camSelect = document.getElementById('cameraSelect');
            camSelect.innerHTML = '';
            const cams = devices.filter(d => d.kind === 'videoinput');
            cams.forEach((c, idx) => {
                const opt = document.createElement('option');
                opt.value = c.deviceId;
                opt.textContent = `[${idx}] ${c.label || 'Camera ' + idx}`;
                camSelect.appendChild(opt);
            });
            if (cams.length === 0) {
                camSelect.innerHTML = '<option value="">[0] Default Camera</option>';
            }
        } catch (e) {}
    }
    loadDevices();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return HTML_PAGE


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)