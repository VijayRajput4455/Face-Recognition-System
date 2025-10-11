# ------------------- Standard Library ------------------------
from urllib.request import urlretrieve
from pathlib import Path

# ------------------- Third-Party Libraries -------------------
import cv2

# ------------------- Local Modules ---------------------------
from logger import get_logger
from config import FRAME_STORAGE_DIR

# ------------------- Initialization --------------------------
logger = get_logger(__name__)

class VideoFrameExtractor:
    def __init__(self, name, age, gender, employee_id, video_url, max_frames=50):
        self.name = name.upper()
        self.age = age.upper()
        self.gender = gender.upper()
        self.employee_id = employee_id.upper()
        self.video_url = video_url
        self.user_folder_name = f"{name.upper()}_{age.upper()}_{gender.upper()}_{employee_id.upper()}"
        self.frame_output_dir = Path(FRAME_STORAGE_DIR) / self.user_folder_name
        self.max_frames = max_frames
        self.frame_output_dir.mkdir(parents=True, exist_ok=True)

        print(self.name, self.age, self.gender, self.employee_id,"---------------------------------->")

    def get_video_rotation(self, video_path):
        """
        Detects the rotation of a video using OpenCV metadata.
        Returns rotation in degrees (0, 90, 180, 270).
        """
        cap = cv2.VideoCapture(str(video_path))
        rotation_angle = 0  # Default no rotation

        try:
            rotate_code = cap.get(cv2.CAP_PROP_ORIENTATION_META)
            if rotate_code == 90:
                rotation_angle = 90
            elif rotate_code == 180:
                rotation_angle = 180
            elif rotate_code == 270:
                rotation_angle = 270
        except Exception as e:
            logger.warning(f"Could not determine rotation metadata: {e}")

        cap.release()
        return rotation_angle

    def correct_frame_rotation(self, frame, rotation_angle):
        """
        Corrects the frame rotation based on the detected angle.
        """
        if rotation_angle == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_angle == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation_angle == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def download_video(self):
        """
        Downloads the video if it's a URL, else returns local path.
        """
        if self.video_url.startswith("http"):
            video_file_path = Path("videos") / f"{self.user_folder_name}_video.mp4"
            urlretrieve(self.video_url, video_file_path)
            logger.info(f"Downloaded video from {self.video_url} to {video_file_path}")
            return video_file_path
        else:
            video_file_path = Path(self.video_url)
            if not video_file_path.is_file():
                logger.error(f"Video file '{self.video_url}' not found.")
                return None
            return video_file_path

    def extract_frames(self):
        """
        Extracts frames from the video and saves up to `max_frames`, capturing every 15th frame.
        Returns the number of frames extracted or an error message.
        """
        try:
            video_path = self.download_video()
            if video_path is None:
                return f"Error: Video file '{self.video_url}' not found."
    
            # rotation_angle = self.get_video_rotation(video_path)

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error("Could not open video file.")
                return "Error: Could not open video file."

            total_frames = 0
            extracted_frames = 0

            while extracted_frames < self.max_frames:
                success, frame = cap.read()
                if not success:
                    break 

                if total_frames % 15 == 0:
                    # corrected_frame = self.correct_frame_rotation(frame, rotation_angle)
                    frame_filename = self.frame_output_dir / f"{self.user_folder_name}_{extracted_frames + 1}.jpg"
                    cv2.imwrite(str(frame_filename), frame)
                    extracted_frames += 1

                total_frames += 1

            cap.release()

            if extracted_frames == 0:
                logger.warning("No frames extracted (video might be too short or corrupt).")
                return "Error: No frames extracted."

            logger.info(f"Extracted {extracted_frames} frames from {video_path}")
            return extracted_frames

        except Exception as e:
            logger.error(f"Unexpected error in extract_frames: {e}")
            return f"Unexpected error occurred: {str(e)}"
        

if __name__ == "__main__":
    extractor = VideoFrameExtractor(
        name="Vijay",
        age="28",
        gender="M",
        employee_id="YT470",
        video_url=r"AANAND.MOV"
    )
    result = extractor.extract_frames()
    print("Result:", result)
