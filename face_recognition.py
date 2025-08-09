import streamlit as st
import pickle 
import numpy as np
import cv2
import sklearn
from PIL import Image
import time
import face_recognition

# Configure page
st.set_page_config(
    page_title="AI Face Recognition",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Fix the main app background */
    .main .block-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        padding-top: 0.5rem;
    }
    
    /* Override Streamlit's default dark background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Content container with white background */
    .content-container {
        background: white;
        padding: 1.5rem;
        border-radius: 20px;
        margin: 0.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2c3e50;
        margin-bottom: 0.5rem;
        margin-top: 0;
    }
    
    .sub-header {
        text-align: center;
        color: #495057;
        font-size: 1.2rem;
        margin-bottom: 1.5rem;
        margin-top: 0;
    }
    
    .upload-section {
        background: linear-gradient(135deg, #f8f9ff 0%, #e8f4f8 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.1);
        border: 2px solid rgba(102, 126, 234, 0.1);
    }
    
    .result-success {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);
        animation: fadeIn 0.8s ease-in;
    }
    
    .result-user {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        color: #2d3436;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 15px 35px rgba(255, 234, 167, 0.3);
        animation: fadeIn 0.8s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .friend-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    .sidebar-info {
        background: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(10px);
    }
    
    /* Upload area styling */
    .uploadedFile {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.75rem 2rem !important;
        font-weight: bold !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Upload area styling */
    .stFileUploader {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        border: 2px dashed #667eea;
    }
    
    /* Instructions styling */
    .upload-instructions {
        background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: #4a5568;
        border: 2px dashed #667eea;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        return pickle.load(open("face recognition.pkl", "rb"))
    except FileNotFoundError:
        st.error("Model file not found! Please ensure 'face recognition.pkl' is in the correct directory.")
        st.stop()

model = load_model()

# Label mapping
label_map = {
    0: "Shubham Maurya",
    1: "Abhay Pratap Singh", 
    2: "Manish Kalendra Yadav",
    3: "Shikar Srivastava",
    4: "Nishant Pandey"
}

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 About This App")
    st.markdown("""
    <div class="sidebar-info">
    This AI-powered face recognition system can identify your friends from uploaded images using advanced machine learning algorithms.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 👥 Recognized Friends")
    for idx, name in label_map.items():
        icon = "👤" if idx != 4 else "🫵"
        st.markdown(f"{icon} **{name}**")
    
    st.markdown("### 📊 Model Info")
    st.info("Model: Trained on facial features\nAccuracy: High precision recognition")

# Main content wrapped in a container
st.markdown('<div class="content-container">', unsafe_allow_html=True)
st.markdown('<h1 class="main-header">🔍 AI Face Recognition</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Discover who\'s in your photo with cutting-edge AI technology</p>', unsafe_allow_html=True)

# Create columns for layout
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload Your Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image containing a face for best results"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Process and display image
        file_bytes = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (256, 256))
        
        # Display image in a nice container
        st.markdown("### 🖼️ Uploaded Image")
        st.image(image_resized, caption="Image ready for recognition", width=300)
        
        # Predict button with custom styling
        if st.button("🔮 Analyze Face", type="primary", use_container_width=True):
            with st.spinner("🧠 AI is analyzing the image..."):
                time.sleep(1)  # Add a small delay for better UX
                
                # Flatten the image for prediction (convert back to BGR for model)
                image_bgr = cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR)
                image_flat = image_bgr.reshape(1, -1)
                
                # Predict
                prediction = model.predict(image_flat)[0]
                predicted_label = label_map.get(prediction, "Unknown")
                
                # Display results with animation
                if prediction != 4:
                    st.markdown(f"""
                    <div class="result-success">
                        <div class="friend-icon">👥</div>
                        <h2>Friend Identified!</h2>
                        <h3>This is your friend:</h3>
                        <h1 style="font-size: 2.5rem; margin: 1rem 0;">{predicted_label}</h1>
                        <p>Confidence: High ✨</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add celebration
                    st.balloons()
                    
                else:
                    st.markdown(f"""
                    <div class="result-user">
                        <div class="friend-icon">🫵</div>
                        <h2>Hey, that's you!</h2>
                        <h3>Detected user:</h3>
                        <h1 style="font-size: 2.5rem; margin: 1rem 0;">{predicted_label}</h1>
                        <p>Looking good! 😊</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add snow effect for user
                    st.snow()
                    
    else:
        # Instructions when no file is uploaded
        st.markdown("""
        <div class="upload-instructions">
            <h3>🎯 Ready to Recognize Faces?</h3>
            <p style="font-size: 1.1rem; margin: 1rem 0;">Upload an image above to get started</p>
            <p><strong>📋 Supported formats:</strong> JPG, JPEG, PNG</p>
            <p><strong>💡 Pro tip:</strong> Use clear, well-lit photos for best results</p>
            <div style="margin-top: 1.5rem; font-size: 2rem;">📸 ➡️ 🤖 ➡️ 👤</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Close the content container
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; color: white; padding: 1.5rem; margin-top: 1rem;">
    <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 15px; backdrop-filter: blur(10px);">
        <p style="font-size: 1.1rem; margin: 0;">🤖 Powered by Machine Learning | Built with ❤️ using Streamlit</p>
    </div>
    <!-- Bottom right corner name -->
    <div style="position: fixed; bottom: 10px; right: 15px; color: rgba(255, 255, 255, 0.7); font-size: 1rem; z-index: 1000;">
        Rhythm forever ❤️
    </div>
</div>
""", unsafe_allow_html=True)