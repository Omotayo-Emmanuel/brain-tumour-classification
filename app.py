# Import the necessary libraries for the web application and machine learning.
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the trained model. This model was saved in the 'train_model.py' script.
# The 'try-except' block is used to handle the case where the model file is not found.
try:
    model = load_model('my_model.h5')
except Exception as e:
    # If the model cannot be loaded, display an error message and provide guidance.
    st.error(f"Error loading model: {e}")
    st.info("Please ensure 'my_model.h5' is in the same directory and has been successfully created.")
    # Stop the script execution to prevent further errors.
    st.stop()

# Define the class names that correspond to the model's output.
# These names must match the subdirectories in your dataset.
class_names = ['glioma_tumour','meningioma_tumour','no_tumour','pituitary_tumour', 'unknown']

# Set up the Streamlit app's page configuration.
st.set_page_config(page_title="Brain Tumor Classification", page_icon="🧠")
# Display the main title of the application.
st.title('🧠 Brain Tumor Classification')
# Add a brief description of the app's purpose.
st.write("Upload a brain MRI scan image to classify it as Glioma brain tumour, Meningioma brain tumour, Pituitary brain tumour, No brain tumour or an external object.")
st.write("This app uses a VGG16-based transfer learning model.")

# Create a file uploader widget. This allows users to upload an image.
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

# Define a function to preprocess the image.
def preprocess_image(image):
    # Resize the image to 224x224 pixels, which is the input size expected by the model.
    image = image.resize((224, 224))
    # Convert the image to a NumPy array for numerical processing.
    image_array = np.array(image)
    # Add a new dimension to the array to represent the batch size (the model expects a batch of images, even if it's just one).
    image_array = np.expand_dims(image_array, axis=0)
    # Normalize the pixel values to a range of 0 to 1, as the model was trained on normalized data.
    image_array = image_array / 255.0
    # Return the processed image array.
    return image_array

# Check if a file has been uploaded.
if uploaded_file is not None:
    # Open the uploaded file as a PIL Image object.
    image = Image.open(uploaded_file)
    # Display the uploaded image in the Streamlit app, now with use_container_width.
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Create a button to trigger the classification process.
    if st.button('Classify Image'):
        # Preprocess the uploaded image, ensuring it's in RGB format.
        processed_image = preprocess_image(image.convert('RGB'))

        # Use the loaded model to make a prediction on the processed image.
        prediction = model.predict(processed_image)
        # Get the index of the class with the highest prediction probability.
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        # Use the index to get the corresponding class name.
        # This check prevents an IndexError if the model's output doesn't match the class names list.
        if predicted_class_index < len(class_names):
            predicted_class_name = class_names[predicted_class_index]
            # Get the confidence score for the predicted class.
            confidence = prediction[0][predicted_class_index] * 100

            # Display the prediction result to the user.
            st.write("### Prediction:")
            # Show a success message with the predicted class.
            st.success(f"The image is classified as: **{predicted_class_name}**")
            # Show an information message with the confidence score.
            st.info(f"Confidence: **{confidence:.2f}%**")
        else:
            st.error("Prediction index out of bounds. The model's output may not match the class names.")
        
        # Add a separator for better readability.
        st.write("---")

        # Display the probabilities for all classes.
        st.write("#### Probabilities:")
        # Check if the number of predictions matches the number of class names.
        if len(prediction[0]) != len(class_names):
            st.warning(f"Warning: The model's output has {len(prediction[0])} classes, but the app expects {len(class_names)} classes. Displaying available probabilities.")

        # Create a dictionary to map class names to their prediction probabilities, using the shortest list to avoid errors.
        probabilities = {name: float(pred) * 100 for name, pred in zip(class_names, prediction[0])}
        # Display the probabilities in a structured, readable format.
        st.json(probabilities)

# Add a button to reload the app, allowing the user to upload a new image.
if st.button('Upload Another Image'):
    st.rerun()

