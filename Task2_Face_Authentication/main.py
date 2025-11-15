from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io

from face_verification import verify_faces

app = FastAPI(title="Face Authentication Service with FaceNet")

@app.post("/verify-faces")
async def verify_faces_endpoint(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...)
):
    """
    Accepts two image files and returns:
      - verification_result
      - similarity_score
      - bounding_boxes
    """
    img_bytes1 = await file1.read()
    img_bytes2 = await file2.read()
    
    image1 = Image.open(io.BytesIO(img_bytes1))
    image2 = Image.open(io.BytesIO(img_bytes2))
    
    result = verify_faces(image1, image2)
    
    return JSONResponse(content=result)
