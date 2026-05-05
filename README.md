# SkinLesionAI

Real-time skin lesion analysis using a YOLOv8 detector and MobileNetV3 classifier, streamed over WebSockets from a React Native mobile camera to a Python FastAPI backend.

> **Disclaimer:** This is an educational computer-vision project. It is not a substitute for professional medical diagnosis. Always consult a licensed dermatologist.

---

## What It Does

Point your phone camera at a skin lesion. The app streams live frames to a Python backend, which runs a two-stage inference pipeline:

1. **Detection** — YOLOv8 localizes lesion regions in the frame and draws bounding boxes in real-time.
2. **Classification** — A MobileNetV3 model crops each detected region and ranks it across seven dermatological categories with a confidence score and risk level.

Results are pushed back to the phone in under ~150ms, overlaid on the live camera feed.

---

## Architecture

```
┌─────────────────────────────────────┐       WebSocket (ws://host:8010)
│         React Native (Expo)         │ ◄────────────────────────────────┐
│                                     │                                  │
│  Camera → base64 frame @ ~4 FPS  ──►│ ─── JSON {image: "<base64>"} ───►│
│                                     │                                  │
│  SVG bounding boxes + labels   ◄───  │ ◄── {detections, predictions}   │
└─────────────────────────────────────┘                                  │
                                                                         │
┌────────────────────────────────────────────────────────────────────────┤
│                        FastAPI Backend (Python)                        │
│                                                                        │
│  1. decode_frame()    — base64 → NumPy (BGR via OpenCV)                │
│  2. YOLOv8 inference  — localize lesion regions, filter conf ≥ 0.35   │
│  3. MobileNetV3       — classify each crop → top-3 conditions          │
│  4. Broadcast JSON    — {detections, predictions, diagnostics}         │
└────────────────────────────────────────────────────────────────────────┘
```

**Frame throttling:** The backend targets 8 FPS (125ms minimum interval between processed frames) to keep inference responsive on CPU.

---

## ML Pipeline

### Stage 1 — YOLOv8 Lesion Detector

- Model: `models/skin_lesion_yolov8.pt` (custom-trained YOLOv8 checkpoint)
- Framework: [Ultralytics](https://github.com/ultralytics/ultralytics)
- Input: raw camera frame (any resolution — YOLO handles resizing internally)
- Output: bounding boxes `(x, y, w, h)` + confidence scores
- Confidence threshold: `0.35` (filters weak detections)

### Stage 2 — MobileNetV3 Classifier

- Architecture: MobileNetV3-Small (torchvision pretrained backbone, fine-tuned classifier head)
- Input: lesion crop extracted from each YOLO bounding box
- Output: softmax probabilities over 7 conditions
- Falls back to a lightweight heuristic classifier if the fine-tuned checkpoint is not present

**Supported conditions:**

| Condition | Risk Level |
|---|---|
| Melanoma | HIGH |
| Basal Cell Carcinoma | HIGH |
| Actinic Keratosis | MEDIUM-HIGH |
| Benign Keratosis | LOW |
| Nevus (mole) | LOW |
| Dermatofibroma | LOW |
| Vascular Lesion | LOW |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Mobile client | React Native (Expo), TypeScript |
| Camera streaming | Expo Camera, WebSocket API |
| Bounding box overlay | react-native-svg |
| Backend API | FastAPI, Python 3.11 |
| Object detection | YOLOv8 (Ultralytics) |
| Classification | MobileNetV3 (PyTorch / torchvision) |
| Image decoding | OpenCV, Pillow, NumPy |
| Async server | Uvicorn with WebSocket concurrency |

---

## Project Structure

```
backend/
  main.py                 FastAPI WebSocket server, YOLO + MobileNet inference
  requirements.txt        Backend Python dependencies

models/
  skin_lesion_yolov8.pt   Custom YOLOv8 detector checkpoint
  skin_lesion_mobilenet.pt  (Optional) Fine-tuned MobileNetV3 weights

SkinLesionAI-frontend/
  App.tsx                 React Native camera client — streaming, overlay, state
  package.json            Expo + npm dependencies

train.py                  Model training scaffold
```

---

## Model Weights

The YOLOv8 checkpoint (`models/skin_lesion_yolov8.pt`) is not stored in this repo due to file size. Place the model file at `models/skin_lesion_yolov8.pt` before starting the backend. The optional fine-tuned MobileNetV3 weights (`models/skin_lesion_mobilenet.pt`) follow the same convention — if absent, the backend uses a fallback classifier and reports `mobilenet_loaded: false` in diagnostics.

---

## Running Locally

### Backend

```bash
cd SkinLesionAI
python -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt

uvicorn backend.main:app --host 0.0.0.0 --port 8010
```

Verify the backend is running and models loaded:

```bash
curl http://localhost:8010/health
```

Expected response:
```json
{
  "active_connections": 0,
  "performance": { "processed_fps": 0.0, "avg_inference_ms": 0.0 },
  "models": { "yolo": true, "mobilenet": false }
}
```

`mobilenet: false` is normal if `models/skin_lesion_mobilenet.pt` is not present — the backend uses a fallback classifier and still returns predictions.

### Frontend

```bash
cd SkinLesionAI-frontend
npm install

# Run in browser
npm run web

# Run on phone (requires Expo Go app)
npm start
```

The app auto-detects the backend host from Expo's dev server and connects to port `8010`. For physical devices, the laptop and phone must be on the same Wi-Fi network.

---

## How the Real-Time Loop Works

1. The React Native app captures a camera frame every **250ms** at 85% JPEG quality.
2. The frame is base64-encoded and sent as a JSON payload over a persistent WebSocket connection.
3. The backend decodes the frame, runs YOLO detection, then classifies each detected region with MobileNetV3.
4. The result is broadcast back as JSON within the same WebSocket message cycle.
5. The frontend updates bounding boxes and condition labels on the live camera overlay — all in under ~150ms round-trip.

Auto-reconnect is built in with exponential backoff (800ms → 6s) so the stream recovers from network interruptions without user interaction.

<img width="3010" height="542" alt="image" src="https://github.com/user-attachments/assets/8fbd877e-c2cf-4b6e-b5da-9b16ae5fb7f7" />

---

## Development

Built with [Claude Code](https://claude.ai/code) and [Cursor](https://cursor.sh) as AI coding assistants throughout development.

---

## API Reference

### `GET /`

Returns backend status:

```json
{
  "device": "cpu",
  "conditions_supported": 7,
  "mobilenet_loaded": false,
  "target_fps": 8
}
```

### `GET /health`

Returns live performance metrics and model load status.

### `WS /ws/analyze`

Send:
```json
{ "image": "<base64-encoded JPEG/PNG>" }
```

Receive:
```json
{
  "type": "analysis",
  "detections": [
    { "x": 120, "y": 85, "width": 90, "height": 95, "confidence": 0.71, "label": "lesion", "risk": "MEDIUM" }
  ],
  "predictions": [
    { "name": "Melanoma", "confidence": 0.62, "risk": "HIGH", "source": "mobilenet" }
  ],
  "top_prediction": { "name": "Melanoma", "confidence": 0.62, "risk": "HIGH" },
  "diagnostics": { "inference_ms": 87, "processed_fps": 7.4, "active_connections": 1 }
}
```
