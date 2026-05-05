import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Platform, ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { CameraType, CameraView, useCameraPermissions } from 'expo-camera';
import Constants from 'expo-constants';
import { Rect, Svg, Text as SvgText } from 'react-native-svg';

type RiskLevel = 'LOW' | 'MEDIUM' | 'HIGH';

type Detection = {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  label: string;
  risk: RiskLevel;
};

type Prediction = {
  name: string;
  risk: RiskLevel;
  confidence: number;
  source: string;
};

type Diagnostics = {
  active_connections: number;
  frames_received: number;
  frames_skipped: number;
  processed_fps: number;
  inference_ms: number;
  mobilenet_loaded: boolean;
  device: string;
};

type FrameSize = {
  width: number;
  height: number;
};

const STREAM_INTERVAL_MS = 125;
const RECONNECT_BASE_DELAY_MS = 800;
const RECONNECT_MAX_DELAY_MS = 6000;
const DEFAULT_SERVER_HOST = resolveServerHost();
const WS_URL = `ws://${DEFAULT_SERVER_HOST}:8000/ws/analyze`;

const riskColor: Record<RiskLevel, string> = {
  LOW: '#26a269',
  MEDIUM: '#f59e0b',
  HIGH: '#dc2626',
};

function resolveServerHost() {
  if (Platform.OS === 'web' && typeof window !== 'undefined') {
    return window.location.hostname || 'localhost';
  }

  const expoConstants = Constants as typeof Constants & {
    expoConfig?: { hostUri?: string };
    manifest?: { debuggerHost?: string };
    manifest2?: { extra?: { expoGo?: { debuggerHost?: string; packagerOpts?: { dev?: boolean } } } };
  };
  const hostUri =
    expoConstants.expoConfig?.hostUri ??
    expoConstants.manifest?.debuggerHost ??
    expoConstants.manifest2?.extra?.expoGo?.debuggerHost;

  return hostUri?.split(':')[0] || 'localhost';
}

export default function App() {
  const [facing, setFacing] = useState<CameraType>('back');
  const [permission, requestPermission] = useCameraPermissions();
  const [isCameraOn, setIsCameraOn] = useState(true);
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(true);
  const [status, setStatus] = useState('Connecting to analyzer...');
  const [detections, setDetections] = useState<Detection[]>([]);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [diagnostics, setDiagnostics] = useState<Diagnostics | null>(null);
  const [frameSize, setFrameSize] = useState<FrameSize>({ width: 1, height: 1 });

  const cameraRef = useRef<CameraView | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const streamTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const captureInFlightRef = useRef(false);
  const isMountedRef = useRef(true);

  const clearReconnectTimer = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  }, []);

  const scheduleReconnect = useCallback(() => {
    clearReconnectTimer();
    const delay = Math.min(
      RECONNECT_BASE_DELAY_MS * 2 ** reconnectAttemptRef.current,
      RECONNECT_MAX_DELAY_MS
    );
    reconnectAttemptRef.current += 1;
    setStatus(`Disconnected. Reconnecting in ${(delay / 1000).toFixed(1)}s...`);
    reconnectTimerRef.current = setTimeout(() => {
      connectToServer();
    }, delay);
  }, [clearReconnectTimer]);

  const handleServerMessage = useCallback((event: WebSocketMessageEvent) => {
    const response = JSON.parse(event.data);
    if (response.type === 'connection') {
      setStatus(response.message);
      return;
    }
    if (response.type === 'error') {
      setStatus(response.message || 'Analyzer reported a recoverable error.');
      return;
    }
    setDetections(response.detections ?? []);
    setPredictions(response.predictions ?? []);
    setDiagnostics(response.diagnostics ?? null);
    const top = response.top_prediction as Prediction | null;
    if (top) {
      setStatus(`${top.name}: ${(top.confidence * 100).toFixed(1)}% (${top.risk})`);
    } else {
      setStatus('No lesion detected in the current frame.');
    }
  }, []);

  const connectToServer = useCallback(() => {
    clearReconnectTimer();
    wsRef.current?.close();

    try {
      const socket = new WebSocket(WS_URL);
      wsRef.current = socket;

      socket.onopen = () => {
        reconnectAttemptRef.current = 0;
        setIsConnected(true);
        setStatus('Connected. Starting live analysis...');
      };

      socket.onmessage = handleServerMessage;

      socket.onerror = () => {
        setStatus('Connection error. Retrying...');
      };

      socket.onclose = () => {
        setIsConnected(false);
        wsRef.current = null;
        if (isMountedRef.current) {
          scheduleReconnect();
        }
      };
    } catch {
      setIsConnected(false);
      scheduleReconnect();
    }
  }, [clearReconnectTimer, handleServerMessage, scheduleReconnect]);

  const sendFrame = useCallback(async () => {
    const socket = wsRef.current;
    if (
      !cameraRef.current ||
      !socket ||
      socket.readyState !== WebSocket.OPEN ||
      !isCameraOn ||
      !isStreaming ||
      captureInFlightRef.current
    ) {
      return;
    }

    captureInFlightRef.current = true;
    try {
      const photo = await cameraRef.current.takePictureAsync({
        base64: true,
        quality: 0.35,
        skipProcessing: true,
      });

      if (photo.base64) {
        setFrameSize({
          width: photo.width || 1,
          height: photo.height || 1,
        });
        socket.send(`data:image/jpeg;base64,${photo.base64}`);
      }
    } catch {
      setStatus('Frame capture failed. Streaming will continue.');
    } finally {
      captureInFlightRef.current = false;
    }
  }, [isCameraOn, isStreaming]);

  useEffect(() => {
    isMountedRef.current = true;
    connectToServer();
    return () => {
      isMountedRef.current = false;
      clearReconnectTimer();
      wsRef.current?.close();
    };
  }, [clearReconnectTimer, connectToServer]);

  useEffect(() => {
    const tick = async () => {
      await sendFrame();
      streamTimerRef.current = setTimeout(tick, STREAM_INTERVAL_MS);
    };

    streamTimerRef.current = setTimeout(tick, STREAM_INTERVAL_MS);
    return () => {
      if (streamTimerRef.current) {
        clearTimeout(streamTimerRef.current);
      }
    };
  }, [sendFrame]);

  if (!permission) {
    return (
      <View style={styles.centered}>
        <Text style={styles.message}>Loading camera...</Text>
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.centered}>
        <Text style={styles.message}>Camera access is required for live lesion analysis.</Text>
        <TouchableOpacity style={styles.primaryButton} onPress={requestPermission}>
          <Text style={styles.buttonText}>Grant Access</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.cameraArea}>
        {isCameraOn ? (
          <>
            <CameraView ref={cameraRef} style={styles.camera} facing={facing} />
            <Svg
              style={StyleSheet.absoluteFillObject}
              viewBox={`0 0 ${frameSize.width} ${frameSize.height}`}
              preserveAspectRatio="none"
            >
              {detections.map((box, index) => (
                <React.Fragment key={`${box.x}-${box.y}-${index}`}>
                  <Rect
                    x={box.x}
                    y={box.y}
                    width={box.width}
                    height={box.height}
                    stroke={riskColor[box.risk] ?? riskColor.MEDIUM}
                    strokeWidth="4"
                    fill="transparent"
                  />
                  <SvgText
                    x={box.x}
                    y={Math.max(box.y - 8, 22)}
                    fill={riskColor[box.risk] ?? riskColor.MEDIUM}
                    fontSize="22"
                    fontWeight="700"
                  >
                    {`${box.label} ${(box.confidence * 100).toFixed(0)}%`}
                  </SvgText>
                </React.Fragment>
              ))}
            </Svg>
          </>
        ) : (
          <View style={styles.cameraOff}>
            <Text style={styles.cameraOffText}>Camera off</Text>
          </View>
        )}
      </View>

      <View style={styles.controls}>
        <View style={styles.statusRow}>
          <View style={[styles.statusDot, isConnected ? styles.connectedDot : styles.disconnectedDot]} />
          <View style={styles.statusCopy}>
            <Text style={styles.statusText}>{status}</Text>
            <Text style={styles.endpointText}>{WS_URL}</Text>
          </View>
        </View>

        <View style={styles.buttonRow}>
          <TouchableOpacity style={styles.secondaryButton} onPress={() => setIsCameraOn((current) => !current)}>
            <Text style={styles.buttonText}>{isCameraOn ? 'Camera Off' : 'Camera On'}</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.secondaryButton} onPress={() => setFacing((current) => (current === 'back' ? 'front' : 'back'))}>
            <Text style={styles.buttonText}>Flip</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.primaryButton} onPress={() => setIsStreaming((current) => !current)}>
            <Text style={styles.buttonText}>{isStreaming ? 'Pause' : 'Resume'}</Text>
          </TouchableOpacity>
        </View>

        <ScrollView style={styles.panel} contentContainerStyle={styles.panelContent}>
          <Text style={styles.panelTitle}>Condition Analysis</Text>
          {predictions.length > 0 ? (
            predictions.map((prediction) => (
              <View key={prediction.name} style={styles.predictionRow}>
                <Text style={styles.predictionName}>{prediction.name}</Text>
                <Text style={[styles.predictionRisk, { color: riskColor[prediction.risk] }]}>
                  {`${(prediction.confidence * 100).toFixed(1)}% ${prediction.risk}`}
                </Text>
              </View>
            ))
          ) : (
            <Text style={styles.emptyText}>Waiting for the first analyzed frame.</Text>
          )}

          <View style={styles.metricsGrid}>
            <Metric label="FPS" value={diagnostics?.processed_fps?.toFixed(1) ?? '0.0'} />
            <Metric label="Latency" value={`${diagnostics?.inference_ms?.toFixed(0) ?? '0'} ms`} />
            <Metric label="Clients" value={`${diagnostics?.active_connections ?? 0}`} />
            <Metric label="Model" value={diagnostics?.mobilenet_loaded ? 'MobileNet' : 'Fallback'} />
          </View>
          <Text style={styles.disclaimer}>Not a medical diagnosis. Consult a dermatologist for clinical guidance.</Text>
        </ScrollView>
      </View>
    </View>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.metric}>
      <Text style={styles.metricLabel}>{label}</Text>
      <Text style={styles.metricValue}>{value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#111827',
  },
  centered: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#111827',
    padding: 24,
  },
  message: {
    color: '#f9fafb',
    fontSize: 16,
    marginBottom: 16,
    textAlign: 'center',
  },
  cameraArea: {
    flex: 1.35,
    minHeight: 360,
    backgroundColor: '#030712',
    position: 'relative',
  },
  camera: {
    width: '100%',
    height: '100%',
  },
  cameraOff: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  cameraOffText: {
    color: '#f9fafb',
    fontSize: 20,
    fontWeight: '700',
  },
  controls: {
    flex: 1,
    backgroundColor: '#f8fafc',
    paddingHorizontal: 16,
    paddingTop: 14,
  },
  statusRow: {
    minHeight: 44,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  statusDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  connectedDot: {
    backgroundColor: '#16a34a',
  },
  disconnectedDot: {
    backgroundColor: '#dc2626',
  },
  statusText: {
    color: '#111827',
    fontSize: 15,
    fontWeight: '600',
  },
  statusCopy: {
    flex: 1,
  },
  endpointText: {
    color: '#64748b',
    fontSize: 11,
    fontWeight: '600',
    marginTop: 2,
  },
  buttonRow: {
    flexDirection: 'row',
    gap: 10,
    marginVertical: 10,
  },
  primaryButton: {
    minHeight: 44,
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#2563eb',
    borderRadius: 8,
    paddingHorizontal: 12,
  },
  secondaryButton: {
    minHeight: 44,
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#334155',
    borderRadius: 8,
    paddingHorizontal: 12,
  },
  buttonText: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '700',
  },
  panel: {
    flex: 1,
  },
  panelContent: {
    paddingBottom: 20,
  },
  panelTitle: {
    color: '#111827',
    fontSize: 17,
    fontWeight: '800',
    marginBottom: 8,
  },
  predictionRow: {
    minHeight: 36,
    borderBottomColor: '#e5e7eb',
    borderBottomWidth: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
  },
  predictionName: {
    flex: 1,
    color: '#111827',
    fontSize: 14,
    fontWeight: '600',
  },
  predictionRisk: {
    fontSize: 13,
    fontWeight: '800',
  },
  emptyText: {
    color: '#64748b',
    fontSize: 14,
    paddingVertical: 10,
  },
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginTop: 12,
  },
  metric: {
    width: '48%',
    minHeight: 58,
    borderRadius: 8,
    backgroundColor: '#e2e8f0',
    justifyContent: 'center',
    padding: 10,
  },
  metricLabel: {
    color: '#475569',
    fontSize: 12,
    fontWeight: '700',
  },
  metricValue: {
    color: '#0f172a',
    fontSize: 16,
    fontWeight: '800',
    marginTop: 2,
  },
  disclaimer: {
    color: '#991b1b',
    fontSize: 12,
    fontWeight: '600',
    marginTop: 12,
  },
});
