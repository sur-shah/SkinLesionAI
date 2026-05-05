from fastapi import FastAPI, WebSocket
import cv2
import torch
import numpy as np
import threading
import time
import json 
import base64
import random
import time
import asyncio
from contextlib import asynccontextmanager
from ultralytics import YOLO


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU available, using CPU")

yolo_model = YOLO("../models/skin_lesion_yolov8.pt")
yolo_model.to(DEVICE)
print(f"Model moved to device: {yolo_model.device}")

class PerformanceTracker:
    def __init__(self):
        self.inference_time = 0
        self.frame_count = 0

    def log_inference_time(self, duration_ms):
        self.inference_time.append(duration_ms)
        self.frame_count += 1
    
        if self.frame_count % 100 == 0:
            avg_time = np.mean(self.inference_time[-100])
            fps = 1000 / avg_time if avg_time > 0 else 0
            print(f"📊 Last 100 frames: {avg_time:.1f}ms avg, {fps:.1f} FPS")

performance_tracker = PerformanceTracker()


class StateManager:
    def __init__(self):
        self.latest_frame = None
        self.current_predictions = {}  # Fixed typo
        self.is_analyzing = False
        self.error_message = None
        self._lock = threading.Lock()

    def set_frame(self, frame):
        """Store latest video frame"""
        with self._lock:
            self.latest_frame = frame

    def get_frame(self):
        """Get latest video frame"""
        with self._lock:
            return self.latest_frame

    def set_predictions(self, predictions):
        """Store latest predictions"""  # Fixed typo
        with self._lock:
            self.current_predictions = predictions

    def get_predictions(self):
        """Get latest predictions"""  # Fixed typo
        with self._lock:
            return self.current_predictions

    def set_analyzing(self, status):
        """Set analyzing state"""
        with self._lock:
            self.is_analyzing = status

    def is_currently_analyzing(self):
        """Get analyzing state"""
        with self._lock:
            return self.is_analyzing

    def set_error(self, error):
        """Set error message"""
        with self._lock:
            self.error_message = error

    def get_error(self):
        """Get error message"""
        with self._lock:
            return self.error_message
        
state_manager = StateManager()


def detect_lesion_with_yolo(frame):
    """Real YOLO lesion detection"""
    try:
        start_time = time.time()

        results = yolo_model(frame, device = DEVICE)
        inference_time_ms = (time.time() - start_time) * 1000
        performance_tracker.log_inference_time(inference_time_ms)
        print(f"🔍 YOLO Results: {len(results)} results")  # Add this
        detections = []
        for result in results:
            boxes = result.boxes
            print(f"📦 Boxes: {boxes}")  # Add this

            if boxes is not None:
                print(f"📦 Number of boxes: {len(boxes)}")  # Add this
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()

                    detections.append({
                        'x': int(x1),
                        'y': int(y1), 
                        'width': int(x2 - x1),
                        'height': int(y2 - y1),
                        'confidence': float(confidence),
                        'label': 'lesion_detected',
                        'risk': 'MEDIUM' 
                    })
        if len(detections) > 0:
            print(f" Inference: {inference_time_ms:.1f}ms) ({len(detections)} detections)")
        else:
            print("No lesions detected")
            #mock detections
            # Add a mock detection for testing
            detections.append({
                'x': 100,
                'y': 100, 
                'width': 150,
                'height': 150,
                'confidence': 0.9,
                'label': 'test_lesion',
                'risk': 'HIGH' 
            })
        print("🧪 Added mock detection for testing")
        return detections
    except Exception as e:
        print(f"YOLO detection error: {e}")
        return []


#BATCH DETECTION FOR GPU
def detect_lesion_batch(frames):
    try:
        start_time = time.time()
        results = yolo_model(frames, device = DEVICE)
        inference_time_ms = (time.time() - start_time) * 1000
        print(f"Batch inference: {inference_time_ms:.1f}ms) for {len(frames)} frames ({inference_time_ms / len(frames):.1f} ms per frame)")
        all_detections = []
        for result in results:
            frame_detections = []
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    frame_detections.append({
                        'x': int(x1),
                        'y': int(y1),
                        'width': int(x2 - x1),
                        'height': int(y2 - y1),
                        'confidence': float(confidence),
                        'label': 'lesion_detected',
                        'risk': 'MEDIUM'
                    })
            all_detections.append(frame_detections)

        return all_detections
    except Exception as e:
        print(f"Batch detection error: {e}")
        return [[] for _ in frames]

def start_video_thread():
    video_thread = threading.Thread(target = video_processing_thread, daemon=True)
    video_thread.start()
    print("Video thread started")

def video_processing_thread():
    #background thread to process video frames

    cap = cv2.VideoCapture(0) #opens camera

    while True:
        ret, frame = cap.read()
        if ret:
            #dummy prediciton for now
            detections = detect_lesion_with_yolo(frame)
            state_manager.set_predictions(detections)
            state_manager.set_frame(frame)
        time.sleep(0.1)




LESION_TYPES = [
    {"name": "Melanoma", "risk": "HIGH"},
    {"name": "Nevus", "risk": "LOW"}, 
    {"name": "Basal Cell Carcinoma", "risk": "MEDIUM"},
    {"name": "Psoriasis", "risk": "MEDIUM"},
    {"name": "Dermatofibroma", "risk": "LOW"},
    {"name": "Vascular Lesion", "risk": "LOW"},
    {"name": "Actinic Keratosis", "risk": "MEDIUM"}
]


def generate_mock_prediction():
    predictions = []
    remaining_confidence = 1.0

    shuffled_lesions = random.sample(LESION_TYPES, 3)
    for i , lesion in enumerate(shuffled_lesions):
        if i == 0:
            confidence = round(random.uniform(0.4, 0.9), 2)
        elif i == 1:
            confidence = round(random.uniform(0.1, remaining_confidence - 0.05), 2)
        else:
            confidence = round(remaining_confidence, 2)
        if confidence > 0.5:
            predictions.append({
                "name": lesion["name"],
                "risk": lesion["risk"],
                "confidence": confidence
            })
        remaining_confidence -= confidence

    return predictions










@asynccontextmanager
async def lifespan(app: FastAPI):
    start_video_thread()
    print("Video processing thread started")
    yield
    print("Shutting down")

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def read_root():
    return{"message": "Hello World"}
    
@app.websocket("/ws/analyze")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Camera connected")

    try:
        while True:

            predictions = state_manager.get_predictions()

            if predictions:
                response = {
                    "message": "Live analysis!",
                    "predictions": predictions,
                    "top_prediction": predictions[0]
                }
                await websocket.send_json(response)
                try:
                    top_prediction = predictions[0]
                    print(f"🔍 Top prediction keys: {list(top_prediction.keys())}")
                    print(f"🔍 Top prediction: {top_prediction}")
                    print(f"Sent: {top_prediction['label']} ({top_prediction['confidence']*100:.1f}%)")
                except Exception as e:
                    print(f"❌ WebSocket error: {e}")
                    print(f"❌ Predictions: {predictions}")
            else:
                await websocket.send_json({"message": "Starting analysis..."})
            await asyncio.sleep(0.5)

            #TODO prcoess image with CNN + LLM
            # For now, send back a simple response

            # response = {"message": "Image received successfuly!"}
            # await websocket.send_json(response)
    except Exception as e:
        print(f"Connection lost: {e}")