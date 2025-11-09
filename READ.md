# Ultra-Optimized Real-Time Vision Streaming System

## Quick Start

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start server:
```bash
python3 server.py
```

4. **Default Live Viewer (Recommended):**
```bash
python3 live_viewer.py --source 0
```

5. **Advanced Options:**
```bash
# Custom stream name and FPS
python3 live_viewer.py --source 0 --stream webcam_1 --fps 25

# Video file processing
python3 live_viewer.py --source path/to/video.mp4

# Minimal overlay
python3 live_viewer.py --source 0 --no-overlay --no-adaptive
```

## Alternative Clients

**JSON Results Client:**
```bash
python3 client.py --source 0
```

**Console Output Only:**
```bash
python3 console_viewer.py --source 0
```

## Architecture

- **server.py**: FastAPI + WebSocket server with YOLOv8 inference
- **client.py**: Async client for video streaming and result collection
- Model loaded once, reused for all frames
- JSON results saved continuously

## Performance Features

- Async processing pipeline
- Frame rate limiting
- Real-time latency tracking
- WebSocket for low-latency communication
- Efficient JPEG encoding for transmission