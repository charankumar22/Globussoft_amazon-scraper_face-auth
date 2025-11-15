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

  # Output Task1- A csv file 
  <img width="588" height="661" alt="image" src="https://github.com/user-attachments/assets/39b1f5a3-dda0-4f1d-927a-97bd0d9880cf" />

  # Output Task2
  <img width="1022" height="610" alt="image" src="https://github.com/user-attachments/assets/5c45a7cc-7972-4e76-ba76-0c6ef09d9ec6" />


