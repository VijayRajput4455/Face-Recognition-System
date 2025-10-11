from pathlib import Path

# ------------------- Base Directories ------------------------
BASE_DIR = Path(__file__).parent.resolve()  # Base directory (project root)

# Define data directories (ensure cross-platform compatibility)
MAIN_DIR = BASE_DIR / "Data" / "Face_Data"
FRAME_STORAGE_DIR = BASE_DIR / "Data" / "Received_Image_Data"
VIDEO_STORAGE_DIR = BASE_DIR / "Data" / "Received_Videos_Data"

# Path to ChromaDB database
CHROMA_DB_PATH = BASE_DIR / "Database" / "chromaDB"

# Path to Logs directories
LOG_DIR = "face_logs"
LOG_FILE = "face_recognition.log"

# ------------------- Ensure Directories Exist ----------------
for directory in [MAIN_DIR, FRAME_STORAGE_DIR, VIDEO_STORAGE_DIR, CHROMA_DB_PATH.parent]:
    directory.mkdir(parents=True, exist_ok=True)
