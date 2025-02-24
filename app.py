import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os

file_id = "1-B6plgolaHEdMd428_TnfciiU9CAnnjK"
url = 'https://drive.google.com/file/d/1-B6plgolaHEdMd428_TnfciiU9CAnnjK/view?usp=drive_link'
model_path = "trained_plant_disease_model.keras"

if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)

def model_prediction(test_image):
    model = tf.keras.models.load_model(model_path)
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Custom Clumsy CSS Styling
st.markdown(
    """
    <style>
        body {background-color: #fffae3;}
        h1, h2, h3 {font-family: 'Comic Sans MS', cursive; color: #d9534f; text-align: center; transform: rotate(-2deg);}
        .stButton>button {background-color: #f0ad4e; color: white; font-size: 20px; font-weight: bold; border-radius: 50px; transform: rotate(3deg); box-shadow: 5px 5px #888888;}
        .stButton>button:hover {background-color: #d9534f; transform: rotate(-3deg); transition: 0.2s;}
        .stImage {border: 5px dashed #5bc0de; border-radius: 20px; padding: 10px;}
        .stSidebar {background-color: grey; padding: 10px; border-radius: 20px;}
        .stFileUploader {border: 3px solid #5bc0de; padding: 5px; border-radius: 15px;}
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("🌱 Plant Disease Detection System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Import Image from pillow to open images
from PIL import Image
img = Image.open("disease.png")

# Display image using streamlit
st.image(img, caption="🌿 Look at these plant diseases!", use_column_width=True)

# Main Page
if app_mode == "HOME":
    st.markdown("<h1>Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)

# Prediction Page
elif app_mode == "DISEASE RECOGNITION":
    st.header("🌿 Let's Detect Some Plant Diseases!")
    test_image = st.file_uploader("📤 Choose an Image:")
    
    if st.button("Show Image 🖼️"):
        st.image(test_image, caption="Uploaded Leaf Image", use_column_width=True)
    
    # Predict button
    if st.button("Predict 🔍"):
        st.snow()
        st.write("🔮 Our Prediction:")
        result_index = model_prediction(test_image)
        
        # Reading Labels
        class_name = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
        st.success("🎯 Model is Predicting it's a {}".format(class_name[result_index]))
