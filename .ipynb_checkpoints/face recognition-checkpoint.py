import streamlit as st
import pickle 

model=pickle.load(open("face recognition.pkl","rb"))


    label_map = {
    0: "Shubham",
    1: "Abhay",
    2: "Manish",
    3: "Rishi",
    4: "Karan"
}

# Title
st.title("Face Recognition App 👤")
st.write("Upload an image and get the predicted friend")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_resized = cv2.resize(image, (256, 256))

    st.image(image_resized, channels="BGR", caption="Uploaded Image", use_column_width=True)

    # Flatten the image for prediction
    image_flat = image_resized.reshape(1, -1)

    # Predict
    prediction = model.predict(image_flat)[0]
    predicted_label = label_map.get(prediction, "Unknown")

    st.success(f"Predicted Person: **{predicted_label}**")