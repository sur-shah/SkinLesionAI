import React, { useRef, useState, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import { Svg, Rect, Text as SvgText } from 'react-native-svg';


const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  message: {
    textAlign: 'center',
    paddingBottom: 10,
    fontSize: 16,
  },
  button: {
    backgroundColor: '#2196F3',
    padding: 15,
    borderRadius: 5,
    marginTop: 10,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  flipButton: {
    backgroundColor: '#4CAF50',
    padding: 10,
    borderRadius: 5,
  },
  flipText: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold',
  },
  cameraToggleButton: {
    backgroundColor: '#FF6B6B',
    padding: 12,
    borderRadius: 5,
    marginBottom: 10,
  },
  cameraToggleText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  // Add the new styles
  analysisPanel: {
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    padding: 15,
    borderRadius: 10,
    marginTop: 10,
    minWidth: 250,
  },
  panelTitle: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 8,
    textAlign: 'center',
  },
  predictionText: {
    color: 'white',
    fontSize: 12,
    marginBottom: 4,
    textAlign: 'center',
  },
});





export default function App() {
  const [facing, setFacing] = useState<CameraType>('back');
  const [permission, requestPermission] = useCameraPermissions();
  const [isCameraOn, setIsCameraOn] = useState<boolean>(true);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [wsConnection, setWsConnection] = useState<WebSocket | null>(null);
  const [serverResponse, setServerResponse] = useState<string>("Disconnected");
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [boundingBoxes, setBoundingBoxes] = useState<any[]>([]); 
  const [currentPredictions, setCurrentPredictions] = useState<any[]>([]);


  const cameraRef = useRef<CameraView | null>(null);

  // Functions

  function connectToServer() {
    try{
      const ws = new WebSocket('ws://172.20.10.2:8000/ws/analyze');      ws.onopen = () => {
        console.log("Connected to backend server!");
        setIsConnected(true);
        setWsConnection(ws);
      };

      ws.onmessage = (event) => {
        const response = JSON.parse(event.data);
        if (response.predictions && response.top_prediction) {
          const top = response.top_prediction;
          const displayText = `${top.label}: ${(top.confidence * 100).toFixed(1)}% (${top.risk} risk)`;          setServerResponse(displayText);
          setCurrentPredictions(response.predictions);
          
          // Create mock bounding boxes
          setBoundingBoxes(response.predictions);  // Use the actual detections!
          console.log("📊 Prediction:", displayText);
        } else {
          setServerResponse(response.message);
        }
      };

      ws.onclose = () => {
        console.log('Disconnected from backend server');
        setIsConnected(false);
        setWsConnection(null);
        setServerResponse("Disconnected - attempting reconnect...");

        setTimeout(() => {
          console.log('🔄 Auto-reconnecting...');
          connectToServer();
        }, 2000);

      };

      ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        setServerResponse("Error connecting to server");
      }
    } catch (error) {
      console.error("WebSocket connection failed:", error);
    }
  }


  function toggleCameraFacing() {
    setFacing(current => (current === 'back' ? 'front' : 'back'));
  }

  function toggleCamera() {
    setIsCameraOn(current => !current);
  }

  async function captureAndAnalyze() {
    if(!cameraRef.current || !isCameraOn || isAnalyzing) return;
    try {
      setIsAnalyzing(true);
      
      // Capture frame from camera
      const photo = await cameraRef.current.takePictureAsync({
        quality: 1.0,  // Lower quality for faster processing
        base64: true,  // Get base64 for easy transmission
        skipProcessing: true  // Skip expensive processing
      });
      
      console.log('Frame captured, size:', photo.uri);
      
      if(photo.base64 && wsConnection){
        const imageData = `data:image/jpeg;base64,${photo.base64}`;
        wsConnection.send(imageData);
        console.log("Image sent to backend", photo.base64.length, 'characters');
      } else {
        console.error("No image data or connection to send to server");
      }
      
      
    } catch (error) {
      console.error('Error capturing frame:', error);
    } finally {
      setIsAnalyzing(false);
    }
  }
  
  useEffect(() =>{
    connectToServer();
  }, []);

  // Real-time analysis effect
  useEffect(() => {
    if(!isCameraOn || !wsConnection) return; // Add wsConnection check
    const interval = setInterval(async () => {
      await captureAndAnalyze();
    }, 1000);
    return () => clearInterval(interval);
  }, [isCameraOn, wsConnection]); // <- Add wsConnection to dependencies

  if (!permission) {
    // Camera permissions are still loading
    return <View style={styles.container}><Text>Loading...</Text></View>;
  }

  if (!permission.granted) {
    // Camera permissions are not granted yet
    return (
      <View style={styles.container}>
        <Text style={styles.message}>We need your permission to show the camera</Text>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={{ flex: 1 }}>
      {isCameraOn ? (
        <View style={{ flex: 2, position: 'relative' }}>
         <CameraView ref={cameraRef} style={{ width: '100%', height: '100%' }} facing={facing} />
          
          {/* Bounding Box Overlays */}
          <Svg style={StyleSheet.absoluteFillObject}>
            {boundingBoxes.map((box, index) => (
              <React.Fragment key={index}>
                <Rect
                  x={box.x}
                  y={box.y}
                  width={box.width}
                  height={box.height}
                  stroke={box.risk.toLowerCase() === 'high' ? 'red' :box.risk.toLowerCase() === 'medium' ? 'orange' : 'green'}                  strokeWidth="3"
                  fill="transparent"
                />
                <SvgText
                  x={box.x}
                  y={box.y - 5}
                  fill={box.risk === 'HIGH' ? 'red' : box.risk === 'MEDIUM' ? 'orange' : 'green'}
                  fontSize="20"
                >
                  {box.label}: {(box.confidence * 100).toFixed(1)}%  {/* Use box.label consistently */}
                </SvgText>
              </React.Fragment>
            ))}
          </Svg>
        </View>
      ) : (
        <View style={{ flex: 2, backgroundColor: '#000', justifyContent: 'center', alignItems: 'center' }}>
          <Text style={{ color: 'white', fontSize: 18 }}>Camera is off</Text>
        </View>
      )}
      
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
        <TouchableOpacity style={styles.cameraToggleButton} onPress={toggleCamera}>
          <Text style={styles.cameraToggleText}>
            {isCameraOn ? 'Turn Camera Off' : 'Turn Camera On'}
          </Text>
        </TouchableOpacity>
        
        <TouchableOpacity style={styles.flipButton} onPress={toggleCameraFacing}>
          <Text style={styles.flipText}>Flip Camera</Text>
        </TouchableOpacity>
        
        <Text style={{ marginTop: 20, color: isConnected ? 'green' : 'red'}}>
          {isConnected ? 'Connected to server' : 'Disconnected from server'}
        </Text>
        
        {/* Live Analysis Panel */}
        <View style={styles.analysisPanel}>
          <Text style={styles.panelTitle}>🤖 Live Analysis</Text>
          {currentPredictions.map((prediction, index) => (
            <Text key={index} style={styles.predictionText}>
              {prediction.name}: {prediction.confidence*100}% ({prediction.risk})
            </Text>
          ))}
        </View>
        
        <Text style={{ color: 'red', marginTop: 10 }}>
          This is not a medical diagnosis. Consult a dermatologist.
        </Text>
      </View>
    </View>
  );
}
// Add this right after your imports, before the App component
