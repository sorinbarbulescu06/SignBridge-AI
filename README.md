# AI Sign Language Bidirectional Translator

A modular, scalable, and high-performance AI-powered bidirectional sign language translation system supporting Sign to Voice and Voice to Sign operations, developed in Python.
You can test the application live here: [Sign Language Translator - Web App]([https://linkul-tau.onrender.com](https://signbridge-ai-2-082g.onrender.com))

> ** Cold Start Notice:** The live demo is hosted on a free-tier Render instance. If it hasn't been accessed recently, the server enters sleep mode to conserve resources. Please allow **60-80 seconds** for the container to wake up and load on your first click.


## Project Overview

<img src="assets/Demo.gif" alt="Learning Bird" width="600">

This project is designed as an accessible communication solution bridging the gap between sign language users and verbal communicators. The architecture strictly separates the client-facing application, optimized for low-latency real-time inference, from a dedicated developer studio used for data ingestion and model training.

### Core Features

1. Sign to Voice (Real-Time Gesture Recognition):
   - Optimized Hand Tracking: Utilizes MediaPipe configured in Lite mode for maximum real-time frame rates.
   - Spatio-Temporal Feature Extraction: Extracts an 88-dimensional anatomical feature vector per frame (normalized 3D coordinates relative to the wrist, inter-fingertip distances, and 3D joint angles) along with sequence-level footprints, ensuring scale, position, and rotation invariance.
   - Advanced Machine Learning: Powered by an Extra-Trees Ensemble classifier optimized for sub-millisecond inference latency.

2. Voice to Sign (Speech Recognition and Live Subtitles):
   - Real-time microphone audio capture using sounddevice and numpy.
   - Asynchronous Speech-to-Text (STT) processing engine designed to convert spoken audio into a continuous, fluid paragraph view in real time.

3. Modular and Clean Architecture:
   - Organized into dedicated packages: ui/, hardware/, utils/, ai/, translation/, and data/.
   - Separation of Concerns: The final client application excludes training utilities, isolating developer tools within a dedicated environment (dev_trainer_studio.py).

## Architecture Overview

```text
src/
├── ai/                 # AI engines (HandTracker, FeatureExtractor, VoiceRecognizer)
├── hardware/           # Hardware streams (VideoEngine, AudioEngine, DeviceScanner)
├── translation/        # Translation logic (SignTranslator, VoiceToSignTranslator)
├── ui/                 # Graphical user interface (AppWindow, UI Helpers)
├── data/               # Trained models (.pkl) and training datasets (.json)
├── dev_trainer_studio.py # Developer-only studio for bulk dataset ingestion and training
└── main.py             # Entry point for the final application
```

## Installation and Setup

### Linux

Makefile commands:
- make = setups virtual environment
```bash
make
```

- make run = runs the application
```bash
make run
```

- make train = runs the dev training tool
```bash
make train
```

- make = deletes the virtual environment
```bash
make clean
```

### Windows

### 1. Repository Clone and Virtual Environment
Ensure Python 3.12 is installed, then run the following commands in your terminal:
```powershell
python3.12 -m venv venv
.\venv\Scripts\activate
```

### 2. Dependencies Installation
```powershell
pip install customtkinter opencv-python pillow sounddevice numpy joblib SpeechRecognition scikit-learn mediapipe==0.10.14
```

### 3. Running the Final Application
To launch the bidirectional translation hub:
```powershell
python src\main.py
```

### 4. Running the Developer Trainer Studio
To perform bulk dataset ingestion (via folders or media files) and retrain the AI model:
```powershell
python src\dev_trainer_studio.py
```

## Performance Optimizations
- Single-Thread Inference: The machine learning classifier is configured for rapid single-threaded inference (n_jobs=1), completely preventing thread synchronization overhead and lag on Windows environments.
- Data Augmentation: The system automatically generates synthetic variations (3D rotations and micro-jitter) from provided samples, ensuring robust recognition performance.
