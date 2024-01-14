import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load the pre-trained model
model = tf.keras.models.load_model('mnist_cnn_model.h5')


# Set page configuration
st.set_page_config(
    page_title="MNIST Digit Recognition App",
    page_icon="✏️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Customize the theme
st.write(
    """
    <style>
        body {
            background-color: #FAFAFC;
        }
        .stSidebar {
            background-color: #000000;
            color: #000000;
        }
        .stRadio .stRadioList label, .stCheckbox .stChecklist label {
            color: #000000;
        }
        .stButton {
            color: #FAFAFC;
        }
        .stButton:hover {
            background-color: #000000;
        }
        .stTextInput, .stTextArea, .stSelectbox, .stSlider, .stNumberInput {
            background-color: #C3190E;
            color: #C3190E;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Set the page title and description
st.title("MNIST Digit Recognition App")
st.write(
    "This app uses a trained CNN model to recognize handwritten digits from 0 to 9."
)

# Create a canvas for user drawing
canvas_result = st_canvas(
    fill_color="#000000",  # Initial color f the canvas
    stroke_width=18,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=300,
    width=300,
    drawing_mode="freedraw",
    key="canvas",
    update_streamlit=True
)

# Check if the user has drawn something
if st.button("Recognize Digit"):
    if canvas_result.image_data is not None:
        # Convert the canvas drawing to a grayscale image
        image_array = np.array(Image.fromarray(canvas_result.image_data).convert("L"))

        # Resize and preprocess the image for prediction
        image_array = np.array(Image.fromarray(image_array).resize((28, 28)))
        image_array = image_array.reshape((1, 28, 28, 1)) / 255.0  # Normalize pixel values

        # Make prediction
        prediction = model.predict(image_array)
        digit = np.argmax(prediction)

        # Display the drawing and prediction
        st.write("Prediction:")
        st.write(f"The model predicts the digit as: {digit}")