import tensorflow as tf
import requests
import gradio as gr
import numpy as np

# Load the MobileNetV2 model
inception_net = tf.keras.applications.MobileNetV2(weights="imagenet")

# Download human-readable labels for ImageNet
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

# Define the function to classify an image
def classify_image(image):
    # Preprocess the user-uploaded image
    image = tf.image.resize(image, [224, 224])
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    # Make predictions using the MobileNetV2 model
    prediction = inception_net.predict(image).flatten()

    # Get the top 3 predicted labels with their confidence scores
    top_indices = prediction.argsort()[-3:][::-1]
    top_classes = [labels[i] for i in top_indices]
    top_scores = [float(prediction[i]) for i in top_indices]

    return {top_classes[i]: top_scores[i] for i in range(3)}

# Create the Gradio interface
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=3),
    live=True,
    title="Image Classification",
    description="Upload an image, and the model will classify it into the top 3 categories.",
)

# Launch the Gradio interface
iface.launch()
