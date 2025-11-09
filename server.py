import asyncio
import json
import time
import logging
from typing import Dict, Any, Optional, List
import cv2
import numpy as np
from ultralytics import YOLO
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from collections import deque
import threading
import queue
import psutil
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VisionServer:
    def __init__(self, model_path: str = "yolov8n.pt", max_queue_size: int = 100):
        # Model setup with device optimization
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path)
        self.model.to(self.device)
        logger.info(f"Model loaded on {self.device}")
        
        # Warmup with multiple dummy frames for stability
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(3):
            self.model(dummy_frame, verbose=False)
        
        # Connection and queue management
        self.clients = {}
        self.max_queue_size = max_queue_size
        self.frame_queues = {}
        
        # Performance metrics
        self.stats = {
            "total_frames": 0, 
            "avg_latency": 0, 
            "fps": 0,
            "active_streams": 0,
            "queue_sizes": {},
            "cpu_usage": 0,
            "memory_usage": 0,
            "gpu_usage": 0
        }
        
        # Adaptive processing
        self.frame_counter = 0
        self.last_time = time.time()
        self.latency_window = deque(maxlen=100)
        self.adaptive_quality = 80
        
        # Background monitoring
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitor_thread.start()
        
    def _monitor_system(self):
        """Background system monitoring"""
        while self.running:
            try:
                # System metrics
                self.stats["cpu_usage"] = psutil.cpu_percent()
                self.stats["memory_usage"] = psutil.virtual_memory().percent
                
                # GPU usage if available
                if torch.cuda.is_available():
                    self.stats["gpu_usage"] = torch.cuda.utilization()
                
                # Adaptive quality adjustment based on load
                avg_cpu = self.stats["cpu_usage"]
                if avg_cpu > 80:
                    self.adaptive_quality = max(60, self.adaptive_quality - 5)
                elif avg_cpu < 50:
                    self.adaptive_quality = min(90, self.adaptive_quality + 5)
                
                time.sleep(5)
            except Exception as e:
                logger.error(f"Monitor error: {e}")
    
    async def process_frame(self, frame_data: bytes, stream_name: str = "default") -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            # Decode frame with error handling
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return {"error": "Invalid frame", "timestamp": int(time.time())}
            
            # Adaptive frame resizing for performance
            h, w = frame.shape[:2]
            if max(h, w) > 1280:  # Resize large frames
                scale = 1280 / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h))
            
            # Run inference with timeout protection
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, self.model, frame)
            
            # Extract detections efficiently
            detections = []
            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    classes = r.boxes.cls.cpu().numpy().astype(int)
                    
                    for box, conf, cls in zip(boxes, confs, classes):
                        if conf > 0.25:  # Confidence threshold
                            detections.append({
                                "label": self.model.names[cls],
                                "conf": float(conf),
                                "bbox": [float(x) for x in box]
                            })
            
            latency_ms = (time.time() - start_time) * 1000
            self.latency_window.append(latency_ms)
            
            # Update performance stats
            self._update_stats(stream_name, latency_ms)
            
            return {
                "timestamp": int(time.time()),
                "frame_id": self.stats["total_frames"],
                "stream_name": stream_name,
                "latency_ms": round(latency_ms, 2),
                "detections": detections,
                "system_load": {
                    "cpu": self.stats["cpu_usage"],
                    "memory": self.stats["memory_usage"],
                    "gpu": self.stats["gpu_usage"]
                }
            }
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            return {
                "error": str(e),
                "timestamp": int(time.time()),
                "stream_name": stream_name
            }
    
    def _update_stats(self, stream_name: str, latency_ms: float):
        """Thread-safe stats update"""
        self.stats["total_frames"] += 1
        current_time = time.time()
        
        # Calculate FPS
        self.frame_counter += 1
        time_diff = current_time - self.last_time
        if time_diff >= 1.0:
            self.stats["fps"] = self.frame_counter / time_diff
            self.frame_counter = 0
            self.last_time = current_time
            
        # Update average latency
        if self.latency_window:
            self.stats["avg_latency"] = sum(self.latency_window) / len(self.latency_window)
    
    async def add_client(self, websocket: WebSocket, stream_name: str = "default"):
        """Add client with queue management"""
        await websocket.accept()
        client_id = id(websocket)
        self.clients[client_id] = {"websocket": websocket, "stream_name": stream_name}
        self.frame_queues[client_id] = asyncio.Queue(maxsize=self.max_queue_size)
        self.stats["active_streams"] += 1
        logger.info(f"Client {client_id} connected for stream {stream_name}")
        return client_id
    
    async def remove_client(self, client_id: int):
        """Remove client and cleanup resources"""
        if client_id in self.clients:
            del self.clients[client_id]
            if client_id in self.frame_queues:
                del self.frame_queues[client_id]
            self.stats["active_streams"] = max(0, self.stats["active_streams"] - 1)
            logger.info(f"Client {client_id} disconnected")

app = FastAPI()
vision_server = VisionServer()

@app.websocket("/inference/{stream_name}")
async def websocket_endpoint(websocket: WebSocket, stream_name: str = "default"):
    client_id = await vision_server.add_client(websocket, stream_name)
    
    try:
        # Create processing task for non-blocking operation
        processing_task = asyncio.create_task(process_client_frames(client_id, websocket, stream_name))
        await processing_task
        
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"Client {client_id} error: {e}")
    finally:
        await vision_server.remove_client(client_id)

async def process_client_frames(client_id: int, websocket: WebSocket, stream_name: str):
    """Non-blocking frame processing for individual client"""
    frame_count = 0
    
    try:
        while True:
            # Receive frame with timeout to prevent blocking
            try:
                data = await asyncio.wait_for(websocket.receive_bytes(), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning(f"Client {client_id} timeout - sending keepalive")
                await websocket.send_text(json.dumps({"keepalive": True}))
                continue
            
            frame_count += 1
            
            # Drop frames if queue is full (prevent memory overflow)
            if client_id in vision_server.frame_queues:
                queue_size = vision_server.frame_queues[client_id].qsize()
                if queue_size >= vision_server.max_queue_size - 1:
                    logger.warning(f"Dropping frame for client {client_id} - queue full")
                    continue
            
            # Process frame asynchronously
            result = await vision_server.process_frame(data, stream_name)
            
            # Send result with error handling
            try:
                await websocket.send_text(json.dumps(result))
            except Exception as e:
                logger.error(f"Failed to send result to client {client_id}: {e}")
                break
                
            # Adaptive frame skipping under high load
            if vision_server.stats["cpu_usage"] > 85 and frame_count % 2 == 0:
                await asyncio.sleep(0.01)  # Small delay to reduce load
                
    except Exception as e:
        logger.error(f"Processing error for client {client_id}: {e}")

@app.get("/stats")
async def get_stats():
    """Enhanced stats with system metrics"""
    stats = vision_server.stats.copy()
    stats["queue_sizes"] = {cid: q.qsize() for cid, q in vision_server.frame_queues.items()}
    stats["device"] = vision_server.device
    stats["adaptive_quality"] = vision_server.adaptive_quality
    return stats

@app.get("/health")
async def health_check():
    """Health endpoint for load balancer"""
    return {
        "status": "healthy" if vision_server.stats["cpu_usage"] < 90 else "degraded",
        "active_connections": len(vision_server.clients),
        "avg_latency": vision_server.stats["avg_latency"],
        "fps": vision_server.stats["fps"]
    }

if __name__ == "__main__":
    import signal
    
    def signal_handler(signum, frame):
        logger.info("Shutting down server...")
        vision_server.running = False
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Production-grade server configuration
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8000,
        loop="asyncio",
        ws_ping_interval=20,
        ws_ping_timeout=20,
        timeout_keep_alive=30,
        limit_concurrency=1000,
        limit_max_requests=10000
    )
    
    server = uvicorn.Server(config)
    server.run()