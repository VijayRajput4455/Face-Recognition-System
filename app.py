# ------------------- Standard Library ------------------------
import os
import threading
from pathlib import Path
from contextlib import asynccontextmanager

# ------------------- Third-Party Libraries -------------------
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Query, HTTPException

# ------------------- Local Modules ---------------------------
from logger import get_logger
from autotrain_service import autotrain
from video_frame_extractor import VideoFrameExtractor
from face_recognition_system import FaceRecognitionSystem
from face_embedding_manager import FaceEmbeddingManager
from face_data_manager import FaceDataManager

# ------------------- Logger Setup ----------------------------
logger = get_logger()

# ------------------- Initialize Face Recognition --------------
face_recognition = FaceRecognitionSystem()
face_embedding = FaceEmbeddingManager()
face_manager = FaceDataManager()


@asynccontextmanager
async def lifespan(app:FastAPI):
    logger.info("Application started.")
    global interpreter, auto_train_thread
    try:
        auto_train_thread = threading.Thread(target=autotrain, daemon=True)
        auto_train_thread.start()
        logger.info("Auto-train thread started.")

        yield

    finally:
        if auto_train_thread:
            auto_train_thread.join()
            logger.info("Auto-train thread stopped.")

# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)

@app.get("/")
def home():
    """Root endpoint to check if the server is running."""
    logger.info("Root endpoint accessed.")
    return {"Message": "Recognition Project"}

@app.get("/process_video")
def process_video_request(
    name: str = Query(..., title="Person's Name"),
    age: str = Query(..., title="Person's Age"),
    gender: str = Query(..., title="Person's Gender"),
    employee_id: str = Query(..., title="Employee ID"),
    video_url: str = Query(..., title="Video URL or Path")
):
    """
    API endpoint to process a video file and extract frames.

    Parameters:
    - name: Name of the person
    - age: Age of the person
    - gender: Gender of the person
    - employee_id: Employee identifier
    - video_url: URL or local path of the video file

    Returns:
    JSON response with success message or error message.
    """
    try:
        logger.info(f"Processing video request: name={name}, age={age}, gender={gender}, employee_id={employee_id}, video_url={video_url}")

        # Process the video and extract frames
        extractor = VideoFrameExtractor(
        name=name,
        age=age,
        gender=gender,
        employee_id=employee_id,
        video_url=video_url
        )

        frame_count = extractor.extract_frames()

        if isinstance(frame_count, str):  # If an error message is returned
            logger.error(f"Error processing video: {frame_count}")
            raise HTTPException(status_code=400, detail=frame_count)

        logger.info(f"Video processed successfully. Frames extracted: {frame_count}")
        return {"message": f"Video processed successfully. Frames extracted: {frame_count}"}

    except Exception as e:
        logger.exception("Unexpected server error in /process_video.")
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {str(e)}")
    
@app.get("/face_recognizer")
def face_recognizer_api(image_url: str = Query(..., title="Image URL or Path")):
    """
    Recognize a face from a local image or remote URL.
    Returns recognized person's metadata or 'Unknown' if not found.
    """
    try:
        logger.info(f"Face recognition request received for image: {image_url}")

        # Load the image (local or URL)
        if os.path.exists(image_url):
            image = cv2.imread(image_url)
            logger.info(f"Loaded image from local path: {image_url}")
        else:
            image = face_manager.urls_to_image(image_url)
            if image is None:
                raise HTTPException(status_code=400, detail="Failed to load image from URL")
            logger.info(f"Loaded image from URL: {image_url}")

        # Perform face recognition
        if isinstance(image, np.ndarray):
            results = face_recognition.search_face(image, top_k=1)

            if results == "No match found" or not results:
                logger.warning("No matching face found in the database.")
                return {
                    "Person_Name": "Unknown",
                    "Person_Age": "Unknown",
                    "Person_Gender": "Unknown",
                    "Person_Employee_Code": "Unknown"
                }

            # Extract metadata
            matched_data = results[0]['metadata']
            response = {
                "Person_Name": matched_data.get("Person Name", "Unknown"),
                "Person_Age": matched_data.get("Person Age", "Unknown"),
                "Person_Gender": matched_data.get("Person Gender", "Unknown"),
                "Person_Employee_Code": matched_data.get("Person Employee Code", "Unknown"),
            }
            logger.info(f"Face recognized: {response}")
            return response

        else:
            logger.error("Invalid image format or failed processing.")
            raise HTTPException(status_code=400, detail="Invalid image or processing error.")

    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        logger.exception("Unexpected server error in /face_recognizer.")
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {str(e)}")


@app.post("/delete_embeddings")
def delete_embeddings_api(employee_code: str = Query(..., title="Employee Code")):
    """
    Delete face embeddings from the database by employee code.
    """
    try:
        logger.info(f"Request to delete embeddings for employee_code: {employee_code}")
        response = face_recognition.delete_embeddings(employee_code)
        logger.info(f"Delete response: {response}")
        return {"message": response}
    except Exception as e:
        logger.exception("Error deleting embeddings.")
        raise HTTPException(status_code=500, detail=f"Failed to delete embeddings: {str(e)}")


# Run the FastAPI application with Uvicorn
if __name__ == "__main__":
    logger.info("Starting FastAPI application...")
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)