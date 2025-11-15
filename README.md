# Globussoft Data Science Task

## Task 1 - Amazon.in Laptop Scraper

- `task1_amazon_laptops.py` scrapes laptop results from Amazon.in.
- Extracts: image URL, title, rating, price, ad/organic.
- Saves to `data/laptops_<timestamp>.csv`.
To run the File(bash):
      python task1_amazon_laptops.py

# Task 2 - Face Authentication 
# Install dependencies
pip install fastapi uvicorn keras-facenet mtcnn tensorflow pillow python-multipart requests

# Run training
python train_facenet.py

# Start the API server
uvicorn main:app --reload

# Test the endpoint
curl -X POST "http://localhost:8000/verify-faces" \
  -F "file1=@image1.jpg" \
  -F "file2=@image2.jpg"
