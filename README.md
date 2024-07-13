# image-classification

This project uses a pre-trained MobileNetV2 model to classify images into the top 3 categories from the ImageNet dataset. The model is wrapped in a Gradio interface, making it easy to use and interact with. Simply upload an image, and the model will return the top 3 predicted labels along with their confidence scores.

# Features

Pre-trained Model: Utilizes MobileNetV2, a state-of-the-art convolutional neural network pre-trained on the ImageNet dataset.
Gradio Interface: Provides a user-friendly web interface for image classification.
Real-time Predictions: Upload an image and get instant predictions with confidence scores.
Top 3 Predictions: Displays the top 3 categories the model predicts for the uploaded image.

# Screenshot

<img width="857" alt="image" src="https://github.com/user-attachments/assets/b6afb538-7759-4a2a-9148-2b572cb3033f">

# Demo

You can try out the application live here: https://huggingface.co/spaces/keerthi-balaji/image-classification

# Requirements

Python 3.6 or higher
TensorFlow
Gradio
Requests
NumPy

# Installation

1. Clone the repository: 

git clone https://github.com/your-username/image-classification-mobilenetv2.git

cd image-classification-mobilenetv2


2. Create and activate a virtual environment (optional): 

python -m venv venv

source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Install the required packages: 

pip install -r requirements.txt


# Usage

1. Run the application: 

python app.py

2. Open your web browser and go to the URL provided by Gradio (usually http://127.0.0.1:7860).

3. Upload an image to see the top 3 predicted labels along with their confidence scores.
