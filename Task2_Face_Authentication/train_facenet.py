"""
Task 2 - Training file (model preparation)
Here we:
  - Load the pre-trained FaceNet model using keras-facenet.
  - This acts as the 'training' / preparation step.
"""
from keras_facenet import FaceNet

def prepare_model():
    print("Loading pre-trained FaceNet model...")
    embedder = FaceNet()
    # You could add any dataset-specific fine-tuning here if needed.
    print("Model is ready to use for embeddings.")
    return embedder

if __name__ == "__main__":
    # Just to verify everything works
    _ = prepare_model()
    print("Training/Preparation step completed successfully.")
