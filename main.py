import streamlit as st
import cv2
import numpy as np
from keras.models import load_model

# Set page configuration
st.set_page_config(
    page_title="Crops Prediction", 
    page_icon=":seedling:", 
    layout="centered",
)

# Load the model
clf = load_model("best_model_cnn_project_pcd.h5")

# Title of the application
st.title('Agriculture-CROP-Prediction :seedling:')

# File uploader
uploaded_file = st.file_uploader("Choose a CROP image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Display the image
    st.image(img_rgb, channels="RGB", caption='Uploaded CROP image.', use_column_width=True)

    # Preprocess the image for prediction
    img_resized = cv2.resize(img, (150, 150))
    img_resized = img_resized / 255.0
    img_resized = np.array(img_resized).reshape((1, 150, 150, 3))

    # Add a button to predict
    if st.button('Predict'):
        # Predict the image
        Y_prediction = clf.predict(img_resized)
        y_pred = np.argmax(Y_prediction[0])
        class_labels = ["Corn", "Cotton", "Rice", "Sugarcane", "Wheat"]
        output_val = "Prediction: {0} with {1:.2f}% confidence".format(class_labels[y_pred], Y_prediction[0, y_pred] * 100)

        # Display the prediction
        st.markdown(f"<div style='text-align: center; font-size: 24px; color: green;'>{output_val}</div>", unsafe_allow_html=True)
