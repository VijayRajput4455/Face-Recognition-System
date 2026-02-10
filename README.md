# Face Recognition System

A comprehensive face recognition system built with FastAPI that processes videos, extracts faces, generates embeddings using CLIP, and stores them in a vector database (ChromaDB) for face recognition and identification.

## Table of Contents
- [Features](#features)
- [Project Overview](#project-overview)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Configuration](#configuration)
- [How to Run](#how-to-run)
- [API Endpoints](#api-endpoints)
- [Dependencies](#dependencies)
- [Workflow](#workflow)
- [Project Components](#project-components)
- [Directory Structure](#directory-structure)

## Features

✅ **Real-time Face Detection** - MTCNN-based face detection in videos and images  
✅ **Advanced Embeddings** - CLIP model for generating accurate face embeddings  
✅ **Vector Database** - ChromaDB for persistent storage and similarity search  
✅ **Automatic Training** - Background service for continuous model training on new data  
✅ **Metadata Management** - Support for name, age, gender, and employee ID  
✅ **Video Processing** - Extract frames from video files or URLs  
✅ **URL Support** - Download images and videos from URLs  
✅ **Comprehensive Logging** - Detailed logging to file and console  
✅ **Thread-safe Operations** - Safe concurrent processing with lock mechanisms  
✅ **Cross-platform Compatibility** - Works on Windows, macOS, and Linux  

## Project Overview

This Face Recognition System provides an end-to-end solution for:
1. **Video Processing** - Extract frames from videos or URLs
2. **Face Detection** - Detect faces in extracted frames using MTCNN
3. **Embedding Generation** - Convert face images to vector embeddings using CLIP
4. **Storage & Retrieval** - Store embeddings in ChromaDB and search for similar faces
5. **Automatic Training** - Continuously process new data in the background

## Technology Stack

### Core Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| **FastAPI** | 0.118.3 | Web framework for REST API |
| **ChromaDB** | 1.1.1 | Vector database for embeddings |
| **PyTorch** | 2.7.1 | Deep learning framework |
| **OpenCV** | 4.12.0.88 | Video and image processing |
| **MTCNN** | 0.1.1 | Face detection |
| **Transformers** | 4.57.0 | HuggingFace models (CLIP) |
| **Uvicorn** | 0.37.0 | ASGI server |
| **Pillow** | 11.0.0 | Image processing |
| **NumPy** | 2.0.2 | Numerical computations |

### Additional Dependencies
- TensorFlow & Keras - Deep learning utilities
- Google Cloud libraries - Model serving
- Pydantic - Data validation
- Python-dotenv - Environment configuration
- Coloredlogs - Enhanced logging
- ONNX Runtime - Model optimization

## Project Structure

```
Face-Recognition-System/
├── app.py                          # Main FastAPI application
├── config.py                       # Configuration and directory setup
├── face_recognition_system.py      # Core face recognition logic
├── face_embedding_manager.py       # Embedding generation and storage
├── face_data_manager.py            # Data operations and monitoring
├── video_frame_extractor.py        # Video processing and frame extraction
├── autotrain_service.py            # Automatic training service
├── logger.py                       # Logging configuration
├── requirements.txt                # Python dependencies
└── Data/                           # Data directories (auto-created)
    ├── Face_Data/                  # Main face dataset storage
    ├── Received_Image_Data/        # Temporary image storage from videos
    └── Received_Videos_Data/       # Video storage
├── Database/
    └── chromaDB/                   # ChromaDB persistent storage
└── face_logs/                      # Application logs
```

## Installation & Setup

### Prerequisites
- **Python 3.10** (Recommended)
- **Git** (for version control)
- **pip** or **conda** (for package management)
- **CUDA** (Optional, for GPU acceleration - NVIDIA GPUs only)

### Step 1: Clone or Download the Repository
```bash
cd /path/to/Face-Recognition-System
```

### Step 2: Create a Virtual Environment (Recommended)

**Using venv:**
```bash
# Create virtual environment
python -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

**Using conda:**
```bash
# Create conda environment
conda create -n face_recognition python=3.10

# Activate environment
conda activate face_recognition
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

**Note:** The first installation may take 5-10 minutes due to large model files.

### Step 4: Verify Installation

```bash
# Test Python and key libraries
python -c "import torch, cv2, chromadb, fastapi; print('All libraries installed successfully!')"
```

## Configuration

### Directory Setup
The system automatically creates necessary directories on first run:
- `Data/Face_Data/` - Main dataset storage
- `Data/Received_Image_Data/` - Temporary frame storage
- `Data/Received_Videos_Data/` - Video storage
- `Database/chromaDB/` - Vector database
- `face_logs/` - Application logs

### Configuration File (config.py)
Edit `config.py` to customize paths:

```python
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()

# Data directories
MAIN_DIR = BASE_DIR / "Data" / "Face_Data"
FRAME_STORAGE_DIR = BASE_DIR / "Data" / "Received_Image_Data"
VIDEO_STORAGE_DIR = BASE_DIR / "Data" / "Received_Videos_Data"
CHROMA_DB_PATH = BASE_DIR / "Database" / "chromaDB"

# Logs
LOG_DIR = "face_logs"
LOG_FILE = "face_recognition.log"
```

## How to Run

### Running the Application

**Start the FastAPI server:**
```bash
# Default: runs on http://localhost:8000
python app.py

# Or using uvicorn directly
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Expected output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
Application started.
Auto-train thread started.
```

### Accessing the API

**Interactive API Documentation:**
```
http://localhost:8000/docs
```

**OpenAPI Schema:**
```
http://localhost:8000/openapi.json
```

## API Endpoints

### 1. Health Check
**GET** `/`
```bash
curl http://localhost:8000/
```
**Response:**
```json
{
  "Message": "Recognition Project"
}
```

### 2. Process Video
**GET** `/process_video`

Processes a video file, extracts frames, and stores them for embedding generation.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| name | string | Yes | Person's full name |
| age | string | Yes | Person's age |
| gender | string | Yes | Person's gender (M/F) |
| employee_id | string | Yes | Unique employee identifier |
| video_url | string | Yes | URL or local path to video file |

**Example Request:**
```bash
curl "http://localhost:8000/process_video?name=John%20Doe&age=30&gender=M&employee_id=EMP001&video_url=/path/to/video.mp4"
```

**Successful Response (200):**
```json
{
  "message": "Video processed successfully. Frames extracted: 50"
}
```

**Error Response (400/500):**
```json
{
  "detail": "Error message describing the issue"
}
```

## Dependencies

### Core Libraries

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | 0.118.3 | Web framework |
| uvicorn | 0.37.0 | ASGI server |
| opencv-python | 4.12.0.88 | Video/image processing |
| torch | 2.7.1+cu118 | Deep learning |
| torchvision | 0.22.1+cu118 | Computer vision utilities |
| transformers | 4.57.0 | Hugging Face models |
| chromadb | 1.1.1 | Vector database |
| mtcnn | 0.1.1 | Face detection |
| pillow | 11.0.0 | Image operations |
| numpy | 2.0.2 | Numerical operations |

### Complete Dependency List

The `requirements.txt` file contains all 120+ dependencies, including:

**Web Framework & Server:**
- fastapi==0.118.3
- uvicorn==0.37.0
- starlette==0.48.0
- pydantic==2.12.0
- pydantic_core==2.41.1

**Deep Learning & AI Models:**
- torch==2.7.1+cu118
- torchvision==0.22.1+cu118
- tensorflow==2.20.0
- keras==3.10.0
- transformers==4.57.0
- safetensors==0.6.2
- tokenizers==0.22.1

**Computer Vision:**
- opencv-python==4.12.0.88
- pillow==11.0.0
- mtcnn==0.1.1
- scikit-image (via dependencies)

**Vector Database & Storage:**
- chromadb==1.1.1
- onnxruntime==1.19.2
- h5py==3.14.0

**Utilities & Support:**
- numpy==2.0.2
- scipy (via dependencies)
- python-dotenv==1.1.1
- coloredlogs==15.0.1
- requests==2.32.5
- httpx==0.28.1

**Google Cloud & AI:**
- google-auth==2.41.1
- googleapis-common-protos==1.70.0
- google-pasta==0.2.0
- grpcio==1.75.1

**Monitoring & Telemetry:**
- opentelemetry-api==1.37.0
- opentelemetry-sdk==1.37.0
- opentelemetry-exporter-otlp-proto-grpc==1.37.0

**Install all at once:**
```bash
pip install -r requirements.txt
```

## Workflow

### Data Processing Pipeline

```
1. USER SUBMITS VIDEO
   ↓
2. VideoFrameExtractor
   ├── Downloads video (if URL) or validates local path
   ├── Extracts frames (up to 50, sampling every 15th frame)
   └── Saves frames with metadata folder structure
   ↓
3. Auto-Training Service (Background)
   ├── Monitors Received_Image_Data folder
   ├── Detects new frames from new videos
   └── Triggers embedding generation
   ↓
4. FaceEmbeddingManager
   ├── Reads images from person folders
   ├── Detects faces using MTCNN
   ├── Generates embeddings using the CLIP model
   └── Stores embeddings in ChromaDB
   ↓
5. FaceDataManager
   ├── Moves processed data to Face_Data
   └── Organizes by person metadata
   ↓
6. SYSTEM READY FOR RECOGNITION
   ├── Embeddings stored in ChromaDB
   ├── Can search/match new faces
   └── Available for similarity queries
```

### Folder Naming Convention

When processing videos, folders are created with this format:
```
PERSON_NAME_AGE_GENDER_EMPLOYEE_ID
Example: JOHN_DOE_30_M_EMP001
```

This structure allows the system to track metadata automatically.

## Project Components

### 1. **app.py** - FastAPI Application
- Main entry point for the system
- Handles HTTP requests
- Manages lifespan events (startup/shutdown)
- Routes video processing requests
- Manages auto-training background thread

**Key Functions:**
- `home()` - Health check endpoint
- `process_video_request()` - Video processing endpoint with metadata

---

### 2. **face_recognition_system.py** - Core Recognition Engine
- Initializes MTCNN face detector
- Sets up ChromaDB vector database
- Loads CLIP model for embeddings
- Implements face detection and search algorithms

**Key Classes & Methods:**
- `FaceRecognitionSystem` - Main class
  - `extract_faces()` - Detect faces in images
  - `search_face()` - Search similar faces in the database

---

### 3. **face_embedding_manager.py** - Embedding Generation
- Loads CLIP model (`openai/clip-vit-base-patch32`)
- Generates embeddings from face images
- Stores embeddings in ChromaDB
- Processes batches of images automatically

**Key Methods:**
- `generate_embedding_from_image()` - Convert image to vector
- `process_and_store_embeddings()` - Batch process folder of images

---

### 4. **face_data_manager.py** - Data Operations
- Monitors folders for new files
- Downloads images from URLs
- Detects and crops faces from images
- Organizes and moves processed data

**Key Methods:**
- `urls_to_image()` - Download image from URL
- `monitor_folder()` - Track new files
- `move_new_face_data()` - Organize processed data
- `detect_and_crop_faces()` - Extract face regions

---

### 5. **video_frame_extractor.py** - Video Processing
- Downloads videos from URLs
- Extracts frames at intervals
- Handles video rotation/orientation
- Saves frames with metadata

**Key Methods:**
- `download_video()` - Get video from URL or local path
- `extract_frames()` - Extract up to 50 frames
- `get_video_rotation()` - Detect video orientation
- `correct_frame_rotation()` - Correct frame orientation

---

### 6. **autotrain_service.py** - Background Training
- Runs as a daemon thread (polls every 10 minutes)
- Monitors for new video extracts
- Automatically generates embeddings
- Moves processed data to the main dataset
- Thread-safe with a lock mechanism

**Key Function:**
- `autotrain()` - Main auto-training loop

---

### 7. **logger.py** - Logging System
- Centralized logging configuration
- Logs to both file and console
- Format: `[TIMESTAMP] [LEVEL] [MODULE] MESSAGE`
- Stored in `face_logs/face_recognition.log.`

---

### 8. **config.py** - Configuration Management
- Defines all directory paths
- Auto-creates required directories
- Centralized configuration management

---

## Directory Structure

After the first run, your project will have this structure:

```
Face-Recognition-System/
├── app.py
├── config.py
├── face_recognition_system.py
├── face_embedding_manager.py
├── face_data_manager.py
├── video_frame_extractor.py
├── autotrain_service.py
├── logger.py
├── requirements.txt
├── README.md
│
├── Data/
│   ├── Face_Data/                    # Main dataset (processed faces)
│   │   ├── JOHN_DOE_30_M_EMP001/
│   │   │   ├── frame_1.jpg
│   │   │   ├── frame_2.jpg
│   │   │   └── ...
│   │   └── ...
│   │
│   ├── Received_Image_Data/          # Temporary frames from videos
│   │   ├── JOHN_DOE_30_M_EMP001/
│   │   │   ├── frame_1.jpg
│   │   │   └── ...
│   │   └── ...
│   │
│   └── Received_Videos_Data/         # Downloaded videos
│       ├── JOHN_DOE_30_M_EMP001_video.mp4
│       └── ...
│
├── Database/
│   └── chromaDB/                     # Vector database (persistent)
│       ├── chroma.parquet
│       ├── embeddings/
│       └── ...
│
├── face_logs/
│   └── face_recognition.log          # Application logs
│
└── videos/                           # Temporary video storage
    └── ...
```

## Usage Examples

### Example 1: Process Local Video
```bash
curl "http://localhost:8000/process_video?name=John%20Doe&age=30&gender=M&employee_id=EMP001&video_url=/path/to/local/video.mp4"
```

### Example 2: Process Video from URL
```bash
curl "http://localhost:8000/process_video?name=Jane%20Smith&age=28&gender=F&employee_id=EMP002&video_url=https://example.com/video.mp4"
```

### Example 3: Using Python Requests
```python
import requests

url = "http://localhost:8000/process_video"
params = {
    "name": "John Doe",
    "age": "30",
    "gender": "M",
    "employee_id": "EMP001",
    "video_url": "/path/to/video.mp4"
}

response = requests.get(url, params=params)
print(response.json())
```

## Troubleshooting

### Issue: Import errors on startup
**Solution:** Ensure all dependencies are installed
```bash
pip install -r requirements.txt
```

### Issue: ChromaDB initialization error
**Solution:** Check write permissions in the project directory
```bash
chmod -R 755 ./Database
```

### Issue: MTCNN not finding faces
**Solution:** Ensure image quality is good and faces are clearly visible

### Issue: Out of memory errors
**Solution:** Reduce `max_frames` in `VideoFrameExtractor` or increase system RAM

### Issue: Video not processing
**Solution:** Verify video codec is supported by OpenCV (MP4, AVI, MOV)

## Performance Tips

1. **GPU Acceleration**: Install CUDA-enabled PyTorch for faster embeddings
2. **Batch Processing**: Process multiple videos sequentially
3. **Frame Sampling**: Adjust `max_frames` in video_frame_extractor.py
4. **Poll Interval**: Adjust `poll_interval` in autotrain_service.py
5. **Model Size**: Consider using smaller models for faster inference

## Logging

Logs are saved to `face_logs/face_recognition.log` with the following format:
```
[2026-01-27 10:30:45] [INFO] [app] Processing video request: name=John Doe...
[2026-01-27 10:30:46] [INFO] [video_frame_extractor] Downloaded video from...
[2026-01-27 10:30:50] [INFO] [face_embedding_manager] Successfully generated embedding...
```

View logs in real-time:
```bash
tail -f face_logs/face_recognition.log
```

## Development & Contributing

### Running in Development Mode
```bash
# With hot-reload
uvicorn app:app --reload

# With debug logging
export LOG_LEVEL=DEBUG
python app.py
```

### Testing API Endpoints
Use the interactive Swagger UI at: `http://localhost:8000/docs`

## System Requirements

### Minimum
- **CPU**: Dual-core processor
- **RAM**: 8GB
- **Storage**: 10GB
- **Python**: 3.8+

### Recommended
- **CPU**: Quad-core or better (Intel i5/AMD Ryzen 5 or equivalent)
- **RAM**: 16GB
- **GPU**: NVIDIA GPU with CUDA support (GTX 1050 or better)
- **Storage**: 50GB SSD
- **Python**: 3.10 or 3.11

### For GPU Acceleration
- NVIDIA CUDA Toolkit 11.8 or higher
- NVIDIA cuDNN
- PyTorch with CUDA support (already in requirements.txt)

## License

This project is provided as-is for educational and commercial use.

## Support & Contact

For issues, questions, or contributions, please contact the development team.

---

**Last Updated:** January 27, 2026  
**Version:** 1.0.0
