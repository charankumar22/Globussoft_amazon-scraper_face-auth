# ML/AI Tasks: Amazon Scraper & Face Authentication

This repository contains two machine learning tasks:
- **Task 1**: Amazon Laptop Web Scraper
- **Task 2**: Face Authentication System using FaceNet

## Project Structure

```
.
├── task1/
│   ├── amazon_scraper.py          # Web scraper for Amazon laptops
│   └── Task1_amazon_laptops.ipynb # Jupyter notebook
├── task2/
│   ├── train_facenet.py           # Model training script
│   ├── face_verification.py       # Inference module
│   └── main.py                    # FastAPI application
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## Task 1: Amazon Laptop Scraper

### Description
A web scraping tool that extracts laptop product data from Amazon India, including:
- Product titles
- Prices
- Ratings
- Images
- Ad vs Organic classification

### Features
- Scrapes multiple pages
- Exports data to CSV with timestamps
- Handles dynamic content
- User-agent rotation

### Usage
```python
python task1/amazon_scraper.py
```

## Task 2: Face Authentication System

### Description
A face verification system using FaceNet and MTCNN with a FastAPI backend.

### Features
- ✅ FaceNet embeddings (512-dimensional vectors)
- ✅ MTCNN face detection
- ✅ REST API with `/verify-faces` endpoint
- ✅ Returns verification result, similarity score, and bounding boxes

### Installation
```bash
pip install -r requirements.txt
```

### Running Task 2

#### 1. Prepare Model (Optional)
```bash
python task2/train_facenet.py
```

#### 2. Start FastAPI Server
```bash
uvicorn task2.main:app --reload
```

#### 3. Test API
```bash
curl -X POST "http://localhost:8000/verify-faces" \
  -F "file1=@image1.jpg" \
  -F "file2=@image2.jpg"
```

### API Response
```json
{
  "verification_result": "same person",
  "similarity_score": 0.9534,
  "bounding_boxes": {
    "image1": [[210, 187, 142, 206]],
    "image2": [[215, 190, 138, 202]]
  }
}
```

## Technologies Used

### Task 1
- Python 3.x
- BeautifulSoup4
- Requests
- Pandas

### Task 2
- FastAPI
- FaceNet (keras-facenet)
- MTCNN
- TensorFlow
- Uvicorn

## API Documentation

Once the server is running:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## License
MIT License

## Author
Created for ML/AI Assignment
