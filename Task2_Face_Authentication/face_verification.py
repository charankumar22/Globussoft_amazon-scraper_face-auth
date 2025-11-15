"""
Task 2 - Testing/Inference file
This file ONLY has:
  - A function to load the model
  - A function to predict/verify from user input images
"""
import numpy as np
from PIL import Image
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN

_embedder = None
_detector = None

def load_model():
    """
    Loads FaceNet model and MTCNN detector (lazy singletons).
    This is the 'testing' / inference side load function.
    """
    global _embedder, _detector
    if _embedder is None:
        _embedder = FaceNet()
    if _detector is None:
        _detector = MTCNN()
    return _embedder, _detector

def _extract_face(image: Image.Image, detector, required_size=(160, 160)):
    """
    Detects the largest face in the image, crops it, and resizes to required_size.
    Returns (face_image, bounding_boxes).
    """
    pixels = np.asarray(image.convert("RGB"))
    results = detector.detect_faces(pixels)
    if len(results) == 0:
        return None, []
    
    # Choose largest face
    results = sorted(results, key=lambda r: r["box"][2] * r["box"][3], reverse=True)
    
    face = results[0]
    x, y, w, h = face["box"]
    x, y = max(0, x), max(0, y)
    cropped = pixels[y:y+h, x:x+w]
    face_image = Image.fromarray(cropped).resize(required_size)
    
    bounding_box = [int(x), int(y), int(w), int(h)]
    return face_image, [bounding_box]

def _get_embedding(face_image: Image.Image, embedder):
    """
    Converts a cropped face image to a 512-d embedding using FaceNet.
    """
    face_array = np.asarray(face_image).astype("float32")
    mean, std = face_array.mean(), face_array.std()
    face_array = (face_array - mean) / (std + 1e-6)
    samples = np.expand_dims(face_array, axis=0)  # (1,160,160,3)
    embedding = embedder.embeddings(samples)
    return embedding[0]

def _cosine_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

def verify_faces(image1: Image.Image, image2: Image.Image, threshold: float = 0.5):
    """
    Given two PIL Images, returns:
      - verification_result: "same person" / "different person" / "face_not_detected"
      - similarity_score: cosine similarity between embeddings
      - bounding_boxes: dict with boxes for both images
    """
    embedder, detector = load_model()
    
    face1, boxes1 = _extract_face(image1, detector)
    face2, boxes2 = _extract_face(image2, detector)
    
    if face1 is None or face2 is None:
        return {
            "verification_result": "face_not_detected",
            "similarity_score": None,
            "bounding_boxes": {
                "image1": boxes1,
                "image2": boxes2
            }
        }
    
    emb1 = _get_embedding(face1, embedder)
    emb2 = _get_embedding(face2, embedder)
    
    sim = _cosine_similarity(emb1, emb2)
    result = "same person" if sim >= threshold else "different person"
    
    return {
        "verification_result": result,
        "similarity_score": sim,
        "bounding_boxes": {
            "image1": boxes1,
            "image2": boxes2
        }
    }
