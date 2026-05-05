# SkinLesionAI - Project Development History

## Project Overview
Building a real-time skin lesion analysis app using computer vision and AI explanations. The app captures live video frames, classifies lesions using a pretrained CNN (EfficientNet or YOLOv8), and provides AI-generated explanations with risk assessments.

### Key Features
- Real-time skin lesion classification from live camera feed
- 7+ lesion categories from HAM10000 dataset (+ psoriasis from PAD-UFES-20)
- Top-3 predictions with confidence scores
- Risk level mapping (Low/Medium/High)
- AI explanations via LLM (GPT-4o-mini or Claude 3.5 Haiku)
- Optional Grad-CAM heatmap overlay
- Medical disclaimer included

### Target Performance
- <200ms inference per frame
- Demo-ready within one week

---

## Development Journey

### Phase 1: Initial Setup and Planning

**Goal**: Start slow, learn by doing, work step-by-step

#### Machine Learning Foundation (`train.py`)

**Created**: Basic PyTorch training script scaffolding

```python
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Data transforms for preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Commented out for now
])

# Paths to training and validation data (placeholder)
train_dir = 'data/HAM10000/train' # Placeholder, needs data organization
val_dir = 'data/HAM10000/val'   # Placeholder, needs data organization

# Dataset/DataLoader setup (conceptual - needs actual data organization)
# train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
# val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

**Key Learnings**:
- **Transforms**: Image preprocessing pipeline (resize, normalize, convert to tensor)
- **HAM10000 Dataset**: Contains 7 skin lesion types, split into two parts
- **Missing Psoriasis**: HAM10000 doesn't include psoriasis, need PAD-UFES-20 dataset
- **Data Organization**: Need to structure datasets into train/val folders

#### Dataset Research

**HAM10000**: 
- 10,015 dermatoscopic images
- 7 categories: Melanoma, Nevus, Basal Cell Carcinoma, Dermatofibroma, Vascular Lesion, Actinic Keratosis, Benign Keratosis
- Split into two parts on Kaggle
- Size: ~1.5-2GB total

**PAD-UFES-20**:
- Additional dataset for psoriasis classification
- Can be "daisy-chained" with HAM10000

**Fine-tuning Timeline**: 2-6 hours on GPU for pretrained EfficientNet

### Phase 2: Frontend Development

#### Environment Setup

**Virtual Environment**:
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
```

**Expo Installation**:
```bash
npm install -g expo-cli
# Note: Got warning about legacy expo-cli not supporting Node +17
# Solution: Use npx expo instead of global expo-cli
```

**Project Creation**:
```bash
npx create-expo-app SkinLesionAI-frontend --template blank-typescript
cd SkinLesionAI-frontend
npx expo install expo-camera
```

**Dependency Installation**:
```bash
# Web support dependencies
npx expo install react-dom react-native-web @expo/metro-runtime
```

#### Camera Implementation (`App.tsx`)

**Iteration 1**: Basic camera setup with permissions
```tsx
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';

const [facing, setFacing] = useState<CameraType>('back');
const [permission, requestPermission] = useCameraPermissions();
```

**Iteration 2**: Added camera toggle functionality
```tsx
const [isCameraOn, setIsCameraOn] = useState<boolean>(true);

function toggleCamera() {
  setIsCameraOn(current => !current);
}
```

**Iteration 3**: Real-time frame capture
```tsx
async function captureAndAnalyze() {
  if(!cameraRef.current || !isCameraOn) return;
  try {
    setIsAnalyzing(true);
    
    const photo = await cameraRef.current.takePictureAsync({
      quality: 0.7,  // Lower quality for faster processing
      base64: true,  // Get base64 for easy transmission
      skipProcessing: true  // Skip expensive processing
    });
    
    console.log('Frame captured, size:', photo.uri);
    if (photo.base64) { 
      console.log('Base64 data length:', photo.base64.length);
    }
    
  } catch (error) {
    console.error('Error capturing frame:', error);
  } finally {
    setIsAnalyzing(false);
  }
}

// Real-time analysis - capture every 300ms
useEffect(() => {
  if(!isCameraOn) return;
  const interval = setInterval(async () => {
    await captureAndAnalyze();
  }, 300);
  return () => clearInterval(interval);
}, [isCameraOn]);
```

**Key Design Decisions**:
- Real-time continuous analysis vs. single-shot capture
- No permanent image storage (privacy + performance)
- 300ms capture interval for smooth real-time feel
- Base64 encoding for easy WebSocket transmission

### Phase 3: Backend Development

#### FastAPI WebSocket Server (`backend/main.py`)

**Created**: WebSocket-based inference server
```python
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import base64
import json
import random
import time

app = FastAPI()

# Mock data for testing
MOCK_LABELS = ["Melanoma", "Nevus", "Basal Cell Carcinoma", "Psoriasis", 
               "Dermatofibroma", "Vascular Lesion", "Actinic Keratosis"]
RISK_LEVELS = ["Low", "Medium", "High"]

def get_mock_prediction():
    label = random.choice(MOCK_LABELS)
    confidence = round(random.uniform(0.5, 0.99), 2)
    risk = random.choice(RISK_LEVELS)
    explanation = f"This looks like {label} with {confidence*100:.1f}% confidence. Risk appears {risk}. This is a mock explanation for testing purposes."
    return {"label": label, "confidence": confidence, "risk": risk, "explanation": explanation}

@app.websocket("/ws/infer")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected.")
    try:
        while True:
            data = await websocket.receive_text()
            if data.startswith("data:image/png;base64,"):
                base64_image = data.split(",")[1]
                print(f"Received image data (length: {len(base64_image)}). Simulating analysis...")
                
                time.sleep(0.1)  # Simulate 100ms processing
                
                mock_predictions = [get_mock_prediction() for _ in range(3)]
                response_data = {
                    "predictions": mock_predictions,
                    "disclaimer": "This is not a medical diagnosis. Consult a dermatologist."
                }
                await websocket.send_json(response_data)
            else:
                await websocket.send_json({"error": "Invalid data format. Expected base64 image."})
    except Exception as e:
        print(f"WebSocket disconnected: {e}")
```

**Dependencies** (`backend/requirements.txt`):
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
websockets==12.0
python-multipart==0.0.6
Pillow==10.1.0
```

**Features**:
- WebSocket endpoint for real-time communication
- Base64 image reception
- Mock prediction responses (3 predictions with confidence + risk + explanation)
- Medical disclaimer included
- 100ms simulated processing time

---

## Technical Challenges and Solutions

### 1. Import Typo Resolution
**Error**: `Import "tochvision" could not be resolved`
**Solution**: Fixed typo from `tochvision` to `torchvision`
**User Reaction**: "lol" 😄

### 2. TypeScript Camera Reference Typing
**Error**: `'Camera' refers to a value, but is being used as a type`
**Solution**: 
- Import `CameraView` for component reference
- Use `useRef<CameraView | null>(null)` for proper typing
- Import `Camera` separately for utility methods

### 3. Expo CLI Compatibility
**Warning**: `The legacy expo-cli does not support Node +17`
**Solution**: Use `npx expo` instead of globally installed `expo-cli`
**Note**: Not a breaking error, just a compatibility warning

### 4. Web Dependencies Missing
**Error**: `It looks like you're trying to use web support but don't have the required dependencies installed`
**Solution**: `npx expo install react-dom react-native-web @expo/metro-runtime`

### 5. Backend Server Connection Issues
**Error**: `curl: (7) Failed to connect to localhost port 8000`
**Root Cause**: Running `python main.py` from wrong directory
**Solution**: `cd backend` first, then `python main.py`

### 6. Public WiFi Limitations
**Issue**: Starbucks WiFi blocking local device communication
**Solutions**: 
- Use personal hotspot
- Use `npx expo start --tunnel` for public network compatibility

### 7. QR Code Scanning Confusion
**Issue**: Native camera app showing "no usable data found"
**Solution**: Scan QR code with Expo Go app directly, or use native camera which redirects to Expo Go

---

## Architecture Overview

```
SkinLesionAI/
├── train.py                    # ML training script (PyTorch + EfficientNet)
├── backend/
│   ├── main.py                 # FastAPI WebSocket server
│   └── requirements.txt        # Python dependencies
└── SkinLesionAI-frontend/
    ├── App.tsx                 # React Native camera app
    ├── package.json            # Node.js dependencies
    └── [Expo configuration files]
```

### Data Flow
1. **Frontend**: Camera captures frames every 300ms
2. **Encoding**: Convert to base64 for transmission
3. **WebSocket**: Send to backend via WebSocket connection
4. **Backend**: Receive base64, decode, run inference
5. **AI Processing**: CNN classification + LLM explanation
6. **Response**: Send back predictions with confidence + risk + explanation
7. **UI Update**: Display results in real-time

### Technology Stack
- **ML**: PyTorch, EfficientNet, torchcam (Grad-CAM)
- **Backend**: FastAPI, WebSockets, uvicorn
- **Frontend**: React Native, Expo Camera, TypeScript
- **LLM**: GPT-4o-mini or Claude 3.5 Haiku
- **Datasets**: HAM10000, PAD-UFES-20

---

## Next Steps (TODO)

### Immediate (Current Session)
- [ ] ~~Create basic backend~~ ✅ **COMPLETED**
- [ ] Test image sending from frontend to backend
- [ ] Add mock prediction responses
- [ ] Update UI to show live predictions

### Short Term (Next Few Days)
- [ ] Download and organize HAM10000 dataset
- [ ] Download PAD-UFES-20 dataset for psoriasis
- [ ] Implement actual CNN inference in backend
- [ ] Add LLM integration for explanations
- [ ] Implement Grad-CAM visualization

### Medium Term (This Week)
- [ ] Fine-tune EfficientNet on combined datasets
- [ ] Optimize inference speed (<200ms target)
- [ ] Add risk level mapping logic
- [ ] Improve UI/UX with color-coded predictions
- [ ] Add confidence threshold filtering

### Long Term (Polish)
- [ ] Deploy backend to cloud
- [ ] Add offline inference capability
- [ ] Implement data augmentation
- [ ] Add more lesion types
- [ ] Professional UI design
- [ ] Medical professional validation

---

## Key Learning Moments

1. **Step-by-Step Approach**: User requested slower pace to learn by doing - led to better understanding
2. **Real-time vs Static**: Decided on continuous real-time analysis over single-shot capture
3. **Privacy First**: No image storage - immediate analysis and discard
4. **Mock-First Development**: Start with mock responses, then add real ML
5. **Environment Separation**: Clear separation between Python (backend) and Node.js (frontend) environments
6. **WebSocket Choice**: Real-time bidirectional communication perfect for live video analysis

---

## Technical Specifications

### Performance Targets
- **Inference Time**: <200ms per frame
- **Frame Rate**: ~3.3 FPS (300ms intervals)
- **Image Quality**: 0.7 quality for speed optimization
- **Processing**: Skip expensive image processing for speed

### Model Architecture
- **Base Model**: EfficientNet (pretrained)
- **Alternative**: YOLOv8 for comparison
- **Classes**: 7 from HAM10000 + psoriasis from PAD-UFES-20
- **Output**: Top-3 predictions with confidence scores

### Deployment Architecture
- **Development**: Local FastAPI server + Expo dev server
- **Production**: Cloud-hosted backend + Expo build
- **Networking**: WebSocket for real-time communication
- **Image Processing**: Base64 encoding for transmission

---

## Lessons Learned

1. **Start Simple**: Mock responses first, real ML later
2. **Environment Management**: Separate Python/Node.js environments crucial
3. **Real-time Considerations**: Frame capture frequency affects UX
4. **Network Limitations**: Public WiFi can block local development
5. **TypeScript Benefits**: Caught type errors early in development
6. **Expo Ecosystem**: Great for rapid mobile development
7. **WebSocket Power**: Perfect for real-time ML applications

---

*Project Status: Backend foundation complete, ready for ML integration*
*Next Session: Connect frontend to backend, test image transmission*



