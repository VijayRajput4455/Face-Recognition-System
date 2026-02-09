# ------------------- Standard Library ------------------------
import os

# ------------------- Third-Party Libraries -------------------
import chromadb
from mtcnn.mtcnn import MTCNN

# ------------------- Local Modules ---------------------------
from logger import get_logger
from config import CHROMA_DB_PATH, COLLECTION_NAME, MODEL_NAME
from face_embedding_manager import FaceEmbeddingManager
from transformers import CLIPProcessor, CLIPModel

# ------------------- Initialization --------------------------
logger = get_logger(__name__)
face_embedding = FaceEmbeddingManager()

class FaceRecognitionSystem:
    def __init__(self, chroma_db_path=CHROMA_DB_PATH, model_name=MODEL_NAME):
        """
        Initialize the Face Recognition System with ChromaDB and CLIP.
        """
        # Setup logger
        self.logger = get_logger()
        self.logger.info("Initializing Face Recognition System...")

        # Disable TensorFlow logs
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

        # Initialize MTCNN for face detection
        self.face_detector = MTCNN()
        self.logger.info("MTCNN face detector initialized.")

        # Initialize ChromaDB
        try:
            self.chroma_client = chromadb.PersistentClient(
                                        path=str(chroma_db_path),
                                        settings=chromadb.config.Settings(
                                            anonymized_telemetry=False
                                        )
                                    )
            self.collection = self.chroma_client.get_or_create_collection(name=COLLECTION_NAME)
            self.logger.info("ChromaDB initialized successfully.")
        except Exception as e:
            self.logger.error(f"Error initializing ChromaDB: {e}")
            self.collection = None

        # Load CLIP model for embedding generation (optional)
        try:
            self.clip_model = CLIPModel.from_pretrained(model_name)
            self.clip_processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
            self.logger.info("CLIP model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading CLIP model: {e}")
            self.clip_model = None
            self.clip_processor = None

    def extract_faces(self, image):
        """
        Extract the first detected face from an image.
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

        self.logger.info(f"Face detected at (x1={x1}, y1={y1}, x2={x2}, y2={y2})")

        return image[y1:y2, x1:x2]

    def search_face(self, image, top_k=1):
        """
        Search for the closest matching face in ChromaDB.
        """
        self.logger.info("Starting face search...")

        # Extract the face
        face = self.extract_faces(image)
        if face is None:
            self.logger.warning("Face extraction failed.")
            return "No match found"

        # Generate embedding
        try:
            embedding = face_embedding.generate_embedding_from_image(face)
            self.logger.info("Face embedding generated successfully.")
        except Exception as e:
            self.logger.error(f"Error generating face embedding: {e}")
            return "No match found"

        # Query ChromaDB
        try:
            query_results = self.collection.query(
                query_embeddings=[embedding],
                n_results=top_k
            )
            self.logger.info("ChromaDB query executed successfully.")
        except Exception as e:
            self.logger.error(f"Error querying ChromaDB: {e}")
            return "No match found"

        if not query_results["ids"] or not query_results["ids"][0]:
            self.logger.warning("No match found in ChromaDB.")
            return "No match found"

        # Format results
        matches = []
        for i in range(len(query_results["ids"][0])):
            similarity = query_results["distances"][0][i]
            match_id = query_results["ids"][0][i] if similarity < 10 else "Unknown"

            match_info = {
                "matched_id": match_id,
                "similarity_score": similarity,
                "metadata": query_results["metadatas"][0][i] if match_id != "Unknown" else {
                    "Person Name": "Unknown",
                    "Person Age": "Unknown",
                    "Person Gender": "Unknown",
                    "Person Employee Code": "Unknown"
                }
            }
            matches.append(match_info)

        return matches


# if __name__ == "__main__":
#     # Example usage
#     TEST_IMAGE_PATH = r"Data\Reveived_Image_Data\SANGRAMA_26_MALE_YT470\SANGRAMA_26_MALE_YT470_2.jpg"
#     image = cv2.imread(TEST_IMAGE_PATH)

#     system = FaceRecognitionSystem()
#     if image is None:
#         print(f"Failed to load image from {TEST_IMAGE_PATH}")
#     else:
#         # Step 1: Detect and extract face
#         face = system.extract_faces(image)

#         if face is not None:
#             # Show the detected face for verification
#             cv2.imshow("Detected Face", face)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()

#             # Step 2: Search the detected face in ChromaDB
#             results = system.search_face(image, top_k=1)
#             print("Search Results:", results)

#             # Step 3: Print the matched person's details
#             if results != "No match found":
#                 print("Matched Metadata:", results[0]['metadata'])
#         else:
#             print("No face detected.")

