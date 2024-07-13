import tensorflow as tf
import requests
import gradio as gr

# Load the InceptionNet model
inception_net = tf.keras.applications.MobileNetV2()

# Download human-readable labels for ImageNet
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

# Define the function to classify an image
def classify_image(image):
    # Preprocess the user-uploaded image
    image = image.reshape((-1, 224, 224, 3))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    
    # Make predictions using the MobileNetV2 model
    prediction = inception_net.predict(image).flatten()
    
    # Get the top 3 predicted labels with their confidence scores
    top_classes = [labels[i] for i in prediction.argsort()[-3:][::-1]]
    top_scores = [float(prediction[i]) for i in prediction.argsort()[-3:][::-1]]
    
    return {top_classes[i]: top_scores[i] for i in range(3)}

# Create the Gradio interface
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(shape=(224, 224)),
    outputs=gr.Label(num_top_classes=3),
    live=True,
    capture_session=True,  # This captures the user's uploaded image
    title="Image Classification",
    description="Upload an image, and the model will classify it into the top 3 categories.",
)

# Launch the Gradio interface
iface.launch()
