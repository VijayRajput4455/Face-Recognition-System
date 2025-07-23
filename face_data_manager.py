# ------------------- Standard Library ------------------------
import os
import shutil
from pathlib import Path
from urllib.request import urlopen

# ------------------- Third-Party Libraries -------------------
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

# ------------------- Local Modules ---------------------------
from logger import get_logger
from config import FRAME_STORAGE_DIR, VIDEO_STORAGE_DIR, MAIN_DIR

class FaceDataManager:
    """
    Handles face data operations:
    - Monitoring folders for new files
    - Downloading images from URLs
    - Moving processed data
    - Detecting and cropping faces
    """

    def __init__(self, frame_dir=FRAME_STORAGE_DIR, main_dir=MAIN_DIR, video_dir=VIDEO_STORAGE_DIR):
        self.logger = get_logger()
        self.frame_dir = Path(frame_dir)
        self.main_dir = Path(main_dir)
        self.video_dir = Path(video_dir)

        self.face_detector = MTCNN()

        # Track last seen files per folder
        self._last_seen_files = {}

        # Ensure directories exist
        for folder in [self.frame_dir, self.main_dir, self.video_dir]:
            folder.mkdir(parents=True, exist_ok=True)

    def urls_to_image(self, image_url: str):
        """
        Loads an image from a URL and decodes it into a NumPy array.
        """
        try:
            req = urlopen(image_url)
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            self.logger.info(f"Successfully loaded image from URL: {image_url}")
            return image
        except Exception as e:
            self.logger.error(f"Error loading image from URL {image_url}: {e}")
            return None

    def move_new_face_data(self, source_folder: Path, destination_folder: Path):
        """
        Moves all subfolders from the source folder to the destination folder.
        """
        try:
            source_folder = Path(source_folder)
            destination_folder = Path(destination_folder)
            destination_folder.mkdir(parents=True, exist_ok=True)

            for folder_name in os.listdir(source_folder):
                folder_path = source_folder / folder_name
                if folder_path.is_dir():
                    shutil.move(str(folder_path), str(destination_folder))
                    self.logger.info(f"Moved {folder_path} to {destination_folder}")

            self.logger.info("All folders moved successfully!")
        except Exception as e:
            self.logger.error(f"Error moving folders: {e}")

    def monitor_folder(self, folder_path: Path, skip_existing_on_first_check=True):
        """
        Monitors a folder and returns a list of new files since the last check.
        """
        folder_path = Path(folder_path)

        if not folder_path.exists():
            self.logger.error(f"Folder does not exist: {folder_path}")
            return []

        current_files = set(os.listdir(folder_path))

        if folder_path not in self._last_seen_files:
            self._last_seen_files[folder_path] = current_files
            if skip_existing_on_first_check:
                self.logger.info(f"Initial scan of {folder_path}. Ignoring existing files.")
                return []
            return list(current_files)

        last_seen = self._last_seen_files[folder_path]
        new_files = current_files - last_seen
        self._last_seen_files[folder_path] = current_files

        if new_files:
            self.logger.info(f"New files detected in {folder_path}: {new_files}")
            return list(new_files)

        self.logger.info(f"No new files detected in {folder_path}.")
        return []

    def detect_and_crop_face(self, image: np.ndarray):
        """
        Detects and crops the first face from an image using MTCNN.
        """
        if image is None:
            self.logger.error("Input image is None.")
            return None

        self.logger.info("Detecting faces in the image.")
        detected_faces = self.face_detector.detect_faces(image)

        if not detected_faces:
            self.logger.warning("No face detected in the image.")
            return None

        x, y, width, height = detected_faces[0]['box']
        x2, y2 = x + width, y + height

        img_height, img_width, _ = image.shape
        x1, y1, x2, y2 = max(0, x), max(0, y), min(img_width, x2), min(img_height, y2)

        self.logger.info(f"Face detected at: (x1={x1}, y1={y1}, x2={x2}, y2={y2})")
        face = image[y1:y2, x1:x2]
        self.logger.info("Face extraction successful.")
        return face
