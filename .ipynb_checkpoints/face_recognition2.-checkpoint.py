import streamlit as st
import pickle 
import numpy as np
import cv2
import sklearn
model=pickle.load(open("face recognition.pkl","rb"))
label_map = {
0: "Shubham Maurya",
1: "Abhay Pratap Singh",
2: "Manish Kalendra Yadav",
3: "Shikar Srivastava",
4: "Nishant Pandey"
}
# Title
st.title("Face Recognition App 👤")
st.write("Upload an image and get the predicted friend")
# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Read and display image
    file_bytes = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_resized = cv2.resize(image, (256, 256))
    st.image(image_resized, channels="BGR", caption="Uploaded Image")
    if st.button("Predict"):
    # Flatten the image for prediction
        image_flat = image_resized.reshape(1, -1)
    # Predict
        prediction = model.predict(image_flat)[0]
        predicted_label = label_map.get(prediction, "Unknown")
        if prediction != 4:
            st.markdown(f"<h2 style='text-align: center; color: red;'>👤 This is your friend: <strong>{predicted_label}</strong></h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='text-align: center; color: red;'>👤 This is you: <strong>{predicted_label}</strong></h2>", unsafe_allow_html=True)

