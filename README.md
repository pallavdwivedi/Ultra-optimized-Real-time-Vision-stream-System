Ultra-Optimized Real-Time Vision Streaming System

A high-performance, low-latency computer vision streaming pipeline built with FastAPI, WebSockets, and YOLOv8. Designed for real-time inference on video streams with minimal overhead and maximum throughput.

Features: 

-Real-time video streaming over WebSockets

-YOLOv8 inference server with single model load and reuse

-Asynchronous client for high-throughput frame delivery

-Latency-optimized JPEG encoding

-Adaptive FPS and frame-rate limiting

-JSON output support for downstream tasks

-Modular design with interchangeable clients

Project Structure


├── server.py              # FastAPI + WebSocket server with YOLOv8 inference

├── client.py              # Async client sending frames and receiving predictions

├── requirements.txt

└── README.md

Quick Start :

1. Create and activate virtual environment :
python3 -m venv venv
source venv/bin/activate      # Linux/Mac

# Mac or on Windows
venv\Scripts\activate

2. Install dependencies
pip install -r requirements.txt

Running the System
1. Start the YOLOv8 inference server
python3 server.py


This launches a FastAPI WebSocket server that accepts video frames and returns model predictions in real time.

Client Options
Default Client (Recommended)

Sends frames and receives YOLO predictions.

python3 client.py and storges json in results

Live Visualisation with bounding boxes.

python client_visualise.py

--------------------------------------------------------------------------------------------------------------------

Architecture Overview :

1.Server (server.py)

FastAPI WebSocket endpoint

YOLOv8 model initialized once

Per-frame inference

Returns bounding boxes, labels, scores in JSON format

2.Client (client.py)

Captures frames using OpenCV

Async message loop for sending frames and receiving results

Can run from webcam or video file

Streams results to log or downstream processing

--------------------------------------------------------------------------------------------------------------------
![YOLO Output 1](/Demo1.png)
![YOLO Output 2](/Demo.png)
