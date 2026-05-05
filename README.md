# SkinLesionAI

SkinLesionAI is a mobile and web app for real-time skin lesion analysis. The Expo React Native client streams live camera frames to a FastAPI backend over WebSockets, where YOLOv8 detects lesion regions and a MobileNet classifier layer ranks seven supported skin-condition categories.

This is an educational computer-vision project, not a medical diagnosis tool. Users should consult a dermatologist for clinical guidance.

## Features

- Live camera streaming from iOS, Android, and web through Expo Camera.
- FastAPI WebSocket backend with concurrent client tracking and frame throttling for an 8 FPS processing target.
- YOLOv8 lesion detection using `models/skin_lesion_yolov8.pt`.
- MobileNetV3 classification wrapper for seven conditions: Melanoma, Nevus, Basal Cell Carcinoma, Actinic Keratosis, Benign Keratosis, Dermatofibroma, and Vascular Lesion.
- Connection recovery, recoverable frame-level error handling, and live diagnostics for FPS, latency, active clients, device, and model status.

## Project Structure

```text
backend/
  main.py                 FastAPI realtime analysis API
  requirements.txt        Backend runtime dependencies
models/
  skin_lesion_yolov8.pt   YOLOv8 detector checkpoint
SkinLesionAI-frontend/
  App.tsx                 Expo mobile/web client
  package.json            Frontend dependencies and scripts
```

## Backend

```bash
cd /Users/surshah/Desktop/SkinLesionAI
source venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

The backend loads the YOLOv8 detector at startup. If `models/skin_lesion_mobilenet.pt` is present, it also loads the fine-tuned MobileNet classifier weights. Without that checkpoint, the API still returns ranked seven-condition predictions with `source: fallback_mobile_classifier` and reports `mobilenet_loaded: false` in diagnostics.

## Frontend

```bash
cd /Users/surshah/Desktop/SkinLesionAI/SkinLesionAI-frontend
npm install
npm run web
```

For mobile devices, update `DEFAULT_SERVER_HOST` in `SkinLesionAI-frontend/App.tsx` to the LAN IP address of the machine running FastAPI, then start Expo:

```bash
npm start
```

The app connects to:

```text
ws://<server-host>:8000/ws/analyze
```

## Verification

Current checks:

```bash
python -m py_compile backend/main.py
venv/bin/python -c "from backend.main import app; print(app.title)"
cd SkinLesionAI-frontend && npx tsc --noEmit
```
