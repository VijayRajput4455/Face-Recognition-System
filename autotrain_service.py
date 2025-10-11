# ------------------- Standard Library ------------------------
import time
import threading

# ------------------- Local Modules ---------------------------
from config import FRAME_STORAGE_DIR, MAIN_DIR
from logger import get_logger
from face_embedding_manager import FaceEmbeddingManager
from face_data_manager import FaceDataManager  # Ensure these are implemented

# ------------------- Initialization --------------------------
logger = get_logger(__name__)
face_embed = FaceEmbeddingManager()
face_manager = FaceDataManager()

# Thread lock to prevent concurrent training processes
lock = threading.Lock()

def autotrain(poll_interval: int = 600):
    """
    Continuously monitors FRAME_STORAGE_DIR and automatically generates embeddings
    for new files, then moves processed data to MAIN_DIR.

    Args:
        poll_interval (int): Time (in seconds) to wait between folder checks. Default: 10 minutes.
    """
    logger.info(f"Auto-training service started. Monitoring folder: {FRAME_STORAGE_DIR}")

    while True:
        try:
            # Detect new files
            new_files = face_manager.monitor_folder(FRAME_STORAGE_DIR)  # Returns list of new files

            if new_files:
                with lock:
                    logger.info(f"Detected {len(new_files)} new file(s): {new_files}")
                    logger.info("Starting embedding training pipeline...")

                    try:
                        # Process embeddings
                        face_embed.process_and_store_embeddings(FRAME_STORAGE_DIR)
                        logger.info("Embedding generation completed successfully.")

                        # Move processed files to main dataset
                        face_manager.move_new_face_data(FRAME_STORAGE_DIR, MAIN_DIR)
                        logger.info(f"Processed files moved to: {MAIN_DIR}")

                    except Exception as train_err:
                        logger.error(f"Error during training process: {train_err}", exc_info=True)
                    else:
                        logger.info("Auto-training cycle completed successfully.")

            else:
                logger.info(f"No new files detected. Next check in {poll_interval} seconds...")

        except Exception as e:
            logger.error(f"Unexpected error in auto-training loop: {e}", exc_info=True)

        # Wait before the next folder scan
        time.sleep(poll_interval)


# if __name__ == "__main__":
#     try:
#         autotrain()
#     except KeyboardInterrupt:
#         logger.info("Auto-training service stopped manually.")
