# ------------------- Standard Library ------------------------
import os
from pathlib import Path

# ------------------- Third-Party Libraries -------------------
import cv2
import chromadb
import torch
from PIL import UnidentifiedImageError
from transformers import CLIPProcessor, CLIPModel

# ------------------- Local Modules ---------------------------
from config import CHROMA_DB_PATH, FRAME_STORAGE_DIR
from logger import get_logger
from face_data_manager import FaceDataManager

# ------------------- Initialization --------------------------
logger = get_logger()
face_manager = FaceDataManager()


class FaceEmbeddingManager:
    def __init__(self, chroma_db_path=CHROMA_DB_PATH, collection_name="FaceEmbeddings"):
        """
        Initializes ChromaDB client, CLIP model, and sets up the embedding manager.
        """
        self.model_name = "openai/clip-vit-base-patch32"
        logger.info(f"Loading CLIP model: {self.model_name}")
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)

        logger.info("Initializing ChromaDB Client...")
        try:
            self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
            self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
            logger.info("ChromaDB initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.collection = None

    def generate_embedding_from_image(self, image):
        """
        Converts an image into an embedding using the CLIP model.
        """
        try:
            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                embedding = self.model.get_image_features(**inputs)
            embedding_list = embedding.squeeze().tolist()

            if not isinstance(embedding_list, list) or len(embedding_list) == 0:
                logger.error("Generated embedding is empty for image.")
                return None

            logger.info("Successfully generated embedding for image.")
            return embedding_list

        except UnidentifiedImageError:
            logger.error("Unable to identify image format.")
        except FileNotFoundError:
            logger.error("Image file not found.")
        except ValueError as ve:
            logger.error(f"Value Error: {ve}")
        except Exception as e:
            logger.error(f"Unexpected error in generate_embedding_from_image: {e}")

        return None

    def process_and_store_embeddings(self, folder_path=FRAME_STORAGE_DIR):
        """
        Processes images from subfolders and stores embeddings in ChromaDB.
        Each subfolder must be in the format: name_age_gender_employeeID
        """
        if not os.path.exists(folder_path):
            logger.error(f"Folder path '{folder_path}' does not exist.")
            return

        logger.info(f"Starting processing in folder: {folder_path}")

        for person_folder in os.listdir(folder_path):
            person_folder_path = os.path.join(folder_path, person_folder)

            if not os.path.isdir(person_folder_path):
                logger.warning(f"Skipping non-folder file: {person_folder}")
                continue

            logger.info(f"Processing folder: {person_folder}")

            # Extract metadata from folder name (Expected format: name_age_gender_empcode)
            parts = person_folder.split('_')
            if len(parts) < 4:
                logger.warning(f"Skipping folder '{person_folder}' due to incorrect naming format.")
                continue

            person_name, person_age, person_gender, person_employee_code = parts[:4]

            for image_file in os.listdir(person_folder_path):
                image_path = os.path.join(person_folder_path, image_file)

                if not image_file.lower().endswith(("png", "jpg", "jpeg")):
                    logger.warning(f"Skipping non-image file: {image_file}")
                    continue

                logger.info(f"Processing image: {image_file}")
                print(f"Processing image: {image_file}")

                image = cv2.imread(image_path)
                if image is None:
                    logger.error(f"Failed to load image {image_file}. Skipping.")
                    continue

                # Extract face from image
                face = face_manager.detect_and_crop_face(image)
                if face is None:
                    logger.error(f"Face extraction failed for {image_file}. Skipping.")
                    continue

                # Generate embedding
                embedding = self.generate_embedding_from_image(face)
                if embedding is None:
                    logger.error(f"Embedding generation failed for {image_file}. Skipping.")
                    continue

                # Store embedding in ChromaDB
                try:
                    self.collection.add(
                        ids=[image_file],
                        embeddings=[embedding],
                        metadatas=[{
                            "Person Name": person_name,
                            "Person Age": person_age,
                            "Person Gender": person_gender,
                            "Person Employee Code": person_employee_code,
                        }]
                    )
                    logger.info(f"Stored embedding for {image_file} in folder {person_folder}")
                except Exception as e:
                    logger.error(f"Failed to store embedding for {image_file}: {e}")

    def delete_embeddings_by_emp_code(self, employee_code):
        """
        Deletes all embeddings related to a given employee code.
        """
        try:
            results = self.collection.get()  # Fetch all stored embeddings
            ids_to_delete = [
                results["ids"][i]
                for i, metadata in enumerate(results["metadatas"])
                if metadata and metadata.get("Person Employee Code") == employee_code
            ]

            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} embeddings for emp_code {employee_code}.")
                return f"Deleted {len(ids_to_delete)} embeddings for emp_code {employee_code}."
            else:
                logger.warning(f"No embeddings found for emp_code {employee_code}.")
                return "No embeddings found for this emp_code."
        except Exception as e:
            logger.error(f"Error deleting embeddings for emp_code {employee_code}: {e}")
            return "Error deleting embeddings."


# Example Usage
if __name__ == "__main__":
    fem = FaceEmbeddingManager()
    fem.process_and_store_embeddings(FRAME_STORAGE_DIR)
    # response = fem.delete_embeddings_by_emp_code("YT470")
    # print(response)
