import asyncio
import json
import cv2
import websockets
import numpy as np
import argparse
import logging
import time
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class LiveViewer:
    def __init__(self, server_url: str = "ws://localhost:8000/inference", stream_name: str = "default"):
        self.server_url = f"{server_url}/{stream_name}"
        self.stream_name = stream_name
        self.display_window = f"Live Detections - {stream_name}"
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.latency_window = deque(maxlen=50)
        self.avg_latency = 0
        
        # Adaptive settings
        self.target_fps = 30
        self.frame_skip = 0
        self.quality = 80
        
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on frame with optimized rendering"""
        h, w = frame.shape[:2]
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            conf = det['conf']
            label = det['label']
            
            # Color coding by object type
            color_map = {
                'person': (0, 255, 0),
                'car': (255, 0, 0),
                'truck': (255, 0, 255),
                'bicycle': (0, 255, 255),
                'motorcycle': (128, 0, 255)
            }
            color = color_map.get(label, (0, 255, 0))
            
            # Draw bounding box with thickness based on confidence
            thickness = max(1, int(conf * 4))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw filled background for text
            text = f"{label}: {conf:.2f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1-text_size[1]-10), (x1+text_size[0], y1), color, -1)
            
            # Draw label with confidence
            cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
        
        return frame
    
    def draw_performance_overlay(self, frame, result):
        """Draw performance metrics on frame"""
        h, w = frame.shape[:2]
        overlay_y = 30
        
        # Current metrics
        latency = result.get('latency_ms', 0)
        det_count = len(result.get('detections', []))
        
        # Update performance tracking
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
        
        self.latency_window.append(latency)
        self.avg_latency = sum(self.latency_window) / len(self.latency_window)
        
        # Performance text
        perf_text = [
            f"FPS: {self.current_fps:.1f}",
            f"Latency: {latency:.1f}ms (avg: {self.avg_latency:.1f}ms)",
            f"Detections: {det_count}",
            f"Stream: {self.stream_name}"
        ]
        
        # System load if available
        if 'system_load' in result:
            load = result['system_load']
            perf_text.append(f"CPU: {load.get('cpu', 0):.1f}% GPU: {load.get('gpu', 0):.1f}%")
        
        # Draw performance overlay
        for i, text in enumerate(perf_text):
            y_pos = overlay_y + (i * 25)
            
            # Background rectangle
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (10, y_pos-20), (15+text_size[0], y_pos+5), (0, 0, 0), -1)
            
            # Text color based on performance
            if i == 1:  # Latency
                color = (0, 255, 0) if latency < 50 else (0, 255, 255) if latency < 100 else (0, 0, 255)
            else:
                color = (255, 255, 255)
            
            cv2.putText(frame, text, (12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
        
    async def stream_with_display(self, source, show_overlay: bool = True, adaptive_quality: bool = True):
        # Open video source with optimization
        if str(source).isdigit():
            cap = cv2.VideoCapture(int(source))
            # Optimize camera settings
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        else:
            cap = cv2.VideoCapture(source)
            
        if not cap.isOpened():
            logger.error(f"Cannot open video source: {source}")
            return
        
        # Get video properties for optimization
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Video source: {width}x{height}")
        
        # Create resizable window
        cv2.namedWindow(self.display_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.display_window, min(1280, width), min(720, height))
        
        frame_interval = 1.0 / self.target_fps
        last_frame_time = time.time()
        
        try:
            async with websockets.connect(
                self.server_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            ) as websocket:
                logger.info(f"Connected to {self.server_url}")
                frame_count = 0
                skip_count = 0
                
                while True:
                    current_time = time.time()
                    
                    # Frame rate limiting
                    if current_time - last_frame_time < frame_interval:
                        await asyncio.sleep(0.001)
                        continue
                    
                    ret, frame = cap.read()
                    if not ret:
                        if str(source).isdigit():  # webcam
                            continue
                        else:  # video file
                            break
                    
                    # Adaptive frame skipping based on latency
                    if adaptive_quality and self.avg_latency > 100:
                        skip_count += 1
                        if skip_count % 2 == 0:  # Skip every other frame
                            continue
                    else:
                        skip_count = 0
                    
                    # Adaptive quality adjustment
                    if adaptive_quality:
                        if self.avg_latency > 80:
                            self.quality = max(60, self.quality - 5)
                        elif self.avg_latency < 40:
                            self.quality = min(90, self.quality + 5)
                    
                    # Encode and send frame
                    try:
                        _, buffer = cv2.imencode('.jpg', frame, 
                                               [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                        await websocket.send(buffer.tobytes())
                        
                        # Receive result with timeout
                        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        result = json.loads(response)
                        
                        # Handle keepalive
                        if result.get('keepalive'):
                            continue
                        
                        # Handle errors gracefully
                        if 'error' in result:
                            logger.warning(f"Server error: {result['error']}")
                            continue
                        
                        # Draw detections and overlay
                        if 'detections' in result:
                            frame = self.draw_detections(frame, result['detections'])
                        
                        if show_overlay:
                            frame = self.draw_performance_overlay(frame, result)
                        
                        # Display frame
                        cv2.imshow(self.display_window, frame)
                        last_frame_time = current_time
                        
                        # Print stats periodically
                        frame_count += 1
                        if frame_count % 100 == 0:
                            logger.info(f"Processed {frame_count} frames - "
                                      f"FPS: {self.current_fps:.1f}, "
                                      f"Avg Latency: {self.avg_latency:.1f}ms, "
                                      f"Quality: {self.quality}%")
                        
                    except asyncio.TimeoutError:
                        logger.warning("Timeout waiting for server response")
                        continue
                    except Exception as e:
                        logger.error(f"Communication error: {e}")
                        await asyncio.sleep(0.1)
                        continue
                    
                    # Exit controls
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):  # Reset stats
                        self.fps_counter = 0
                        self.fps_start_time = time.time()
                        self.latency_window.clear()
                        logger.info("Performance stats reset")
                    elif key == ord('='):  # Increase target FPS
                        self.target_fps = min(60, self.target_fps + 5)
                        frame_interval = 1.0 / self.target_fps
                        logger.info(f"Target FPS: {self.target_fps}")
                    elif key == ord('-'):  # Decrease target FPS
                        self.target_fps = max(10, self.target_fps - 5)
                        frame_interval = 1.0 / self.target_fps
                        logger.info(f"Target FPS: {self.target_fps}")
                        
        except websockets.exceptions.ConnectionClosed:
            logger.error("Connection to server lost")
        except KeyboardInterrupt:
            logger.info("Stopping viewer...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Viewer stopped")

async def main():
    parser = argparse.ArgumentParser(description='Optimized Live Vision Viewer')
    parser.add_argument('--source', default='0', help='Video source (webcam index or file path)')
    parser.add_argument('--server', default='ws://localhost:8000/inference', help='Server URL')
    parser.add_argument('--stream', default='default', help='Stream name')
    parser.add_argument('--fps', type=int, default=30, help='Target FPS')
    parser.add_argument('--no-overlay', action='store_true', help='Hide performance overlay')
    parser.add_argument('--no-adaptive', action='store_true', help='Disable adaptive quality')
    
    args = parser.parse_args()
    
    # Print controls
    print("=== Live Vision Viewer ===")
    print("Controls:")
    print("  'q' - Quit")
    print("  'r' - Reset performance stats")
    print("  '+' - Increase target FPS")
    print("  '-' - Decrease target FPS")
    print("===========================")
    
    viewer = LiveViewer(args.server, args.stream)
    viewer.target_fps = args.fps
    
    await viewer.stream_with_display(
        args.source, 
        not args.no_overlay, 
        not args.no_adaptive
    )

if __name__ == "__main__":
    asyncio.run(main())