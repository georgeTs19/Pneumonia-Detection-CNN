import streamlit as st # type: ignore
import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.applications.imagenet_utils import preprocess_input # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure page
st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .normal-prediction {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .pneumonia-prediction {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'base_model' not in st.session_state:
    st.session_state.base_model = None
if 'transfer_model' not in st.session_state:
    st.session_state.transfer_model = None

import os  # ⬅️ Add this at the top if not already present

@st.cache_resource
def load_models():
    """Load only the transfer model (used directly and in ensemble)"""
    try:
        ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        transfer_model_path = os.path.join(ROOT_DIR, "outputs", "best_transfer_model.keras")
        base_model_path = os.path.join(ROOT_DIR, "outputs", "best_model.keras")  

        # If you still want ensemble, load both
        base_model = tf.keras.models.load_model(base_model_path)
        transfer_model = tf.keras.models.load_model(transfer_model_path)

        return base_model, transfer_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def ensemble_predictions(base_model, transfer_model, img_array):
    """Average predictions from base and transfer model"""
    try:
        base_prob = base_model.predict(img_array)[0][0]
        transfer_prob = transfer_model.predict(img_array)[0][0]
        avg_prob = (base_prob + transfer_prob) / 2.0
        prediction = "PNEUMONIA" if avg_prob > 0.5 else "NORMAL"
        confidence = avg_prob if avg_prob > 0.5 else 1 - avg_prob
        return prediction, confidence, avg_prob
    except Exception as e:
        st.error(f"Error in ensemble prediction: {str(e)}")
        return None, None, None

def preprocess_image(uploaded_file, target_size=(224, 224)):
    """Preprocess uploaded image for prediction"""
    try:
        # Convert uploaded file to PIL Image
        img = Image.open(uploaded_file)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image
        img = img.resize(target_size)
        
        # Convert to array
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess based on model requirements
        img_array = preprocess_input(img_array)
        
        return img_array, img
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None, None

def make_prediction(model, img_array, model_name):
    """Make prediction using the specified model"""
    try:
        prediction = model.predict(img_array)
        confidence = float(prediction[0][0])
        
        # Assuming binary classification: 0=Normal, 1=Pneumonia
        if confidence > 0.5:
            result = "PNEUMONIA"
            probability = confidence
        else:
            result = "NORMAL"
            probability = 1 - confidence
            
        return result, probability, confidence
    except Exception as e:
        st.error(f"Error making prediction with {model_name}: {str(e)}")
        return None, None, None

def generate_gradcam(model, img_array, layer_name="out_relu"):
    """Generate GradCAM heatmap"""
    try:
        # Create a model that maps the input image to the activations of the last conv layer
        grad_model = tf.keras.models.Model(
            [model.inputs], 
            [model.get_layer(layer_name).output, model.output]
        )
        
        # Compute the gradient of the top predicted class for our input image
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]
        
        # Extract the gradients of the top predicted class
        grads = tape.gradient(loss, conv_outputs)
        
        # Pool the gradients over all the axes leaving out the channel dimension
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiply each channel by its importance
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize the heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    except Exception as e:
        st.error(f"Error generating GradCAM: {str(e)}")
        return None

def display_gradcam(original_img, heatmap, alpha=0.4):
    """Display GradCAM overlay on original image"""
    try:
        # Resize heatmap to match original image
        heatmap_resized = cv2.resize(heatmap, (original_img.size[0], original_img.size[1]))
        
        # Convert PIL to numpy array
        img_array = np.array(original_img)
        
        # Create heatmap overlay
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay heatmap on original image
        superimposed_img = heatmap_colored * alpha + img_array * (1 - alpha)
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        
        return superimposed_img
    except Exception as e:
        st.error(f"Error displaying GradCAM: {str(e)}")
        return None

def create_model_comparison_chart(base_result, transfer_result, base_conf, transfer_conf):
    """Create comparison chart between models"""
    models = ['Ensemple Model', 'Transfer Learning']
    predictions = [base_result, transfer_result]
    confidences = [base_conf, transfer_conf]
    
    fig = go.Figure()
    
    # Add bars for confidence scores
    fig.add_trace(go.Bar(
        x=models,
        y=confidences,
        text=[f'{pred}<br>{conf:.2%}' for pred, conf in zip(predictions, confidences)],
        textposition='auto',
        marker_color=['#ff7f0e' if pred == 'PNEUMONIA' else '#2ca02c' for pred in predictions],
        name='Confidence Score'
    ))
    
    fig.update_layout(
        title='Model Predictions Comparison',
        xaxis_title='Model',
        yaxis_title='Confidence Score',
        yaxis=dict(range=[0, 1]),
        showlegend=False,
        height=400
    )
    
    return fig

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🩺 Pneumonia Detection AI</h1>', unsafe_allow_html=True)
    st.markdown("### Upload a chest X-ray image to detect pneumonia using AI")
    
    # Sidebar
    st.sidebar.header("🛠️ Model Settings")
    
    # Model loading
    if not st.session_state.models_loaded:
        with st.spinner("Loading AI models..."):
            base_model, transfer_model = load_models()
            if base_model is not None and transfer_model is not None:
                st.session_state.base_model = base_model
                st.session_state.transfer_model = transfer_model
                st.session_state.models_loaded = True
                st.success("✅ Models loaded successfully!")
            else:
                st.error("❌ Failed to load models. Please check model paths.")
                return
    
    # Sidebar options
    show_gradcam = st.sidebar.checkbox("Show GradCAM Visualization", value=True)
    compare_models = st.sidebar.checkbox("Compare Both Models", value=True)
    gradcam_layer = st.sidebar.selectbox(
        "GradCAM Layer",
        ["out_relu", "block_2_depthwise", "block_2_expand_relu"],
        index=0
    )
    
    # File upload
    st.header("📤 Upload Chest X-ray Image")
    uploaded_file = st.file_uploader(
        "Choose an X-ray image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a chest X-ray image in PNG, JPG, or JPEG format"
    )
    
    if uploaded_file is not None:
        # Create columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Original Image")
            original_image = Image.open(uploaded_file)
            st.image(original_image, caption="Uploaded X-ray", use_column_width=True)
        
        # Preprocess image
        img_array, processed_img = preprocess_image(uploaded_file)
        
        if img_array is not None:
            # Make predictions
            with st.spinner("🔍 Analyzing X-ray..."):
                # Transfer learning model prediction
                transfer_result, transfer_prob, transfer_conf = make_prediction(
                    st.session_state.transfer_model, img_array, "Transfer Learning"
                )
                
                # Base model prediction (if comparison is enabled)
                if compare_models:
                    base_result, base_prob, base_conf = ensemble_predictions(
                    st.session_state.base_model,
                    st.session_state.transfer_model,
                    img_array
                )

            
            # Display results
            with col2:
                st.subheader("🎯 Prediction Results")
                
                # Transfer learning result
                if transfer_result:
                    prediction_class = "pneumonia-prediction" if transfer_result == "PNEUMONIA" else "normal-prediction"
                    st.markdown(f"""
                    <div class="prediction-box {prediction_class}">
                        <strong>Transfer Learning Model</strong><br>
                        Prediction: {transfer_result}<br>
                        Confidence: {transfer_prob:.2%}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Base model result (if enabled)
                if compare_models and 'base_result' in locals():
                    prediction_class = "pneumonia-prediction" if base_result == "PNEUMONIA" else "normal-prediction"
                    st.markdown(f"""
                    <div class="prediction-box {prediction_class}">
                        <strong>Ensemble Model</strong><br>
                        Prediction: {base_result}<br>
                        Confidence: {base_prob:.2%}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Model comparison chart
            if compare_models and 'base_result' in locals():
                st.subheader("📊 Model Comparison")
                comparison_chart = create_model_comparison_chart(
                    base_result, transfer_result, base_prob, transfer_prob
                )
                st.plotly_chart(comparison_chart, use_container_width=True)
            
            # GradCAM visualization
            if show_gradcam:
                st.subheader("🔥 GradCAM Visualization")
                st.info("GradCAM shows which parts of the X-ray the AI focused on for its decision")
                
                with st.spinner("Generating GradCAM..."):
                    heatmap = generate_gradcam(
                        st.session_state.transfer_model, 
                        img_array, 
                        layer_name=gradcam_layer
                    )
                
                if heatmap is not None:
                    gradcam_img = display_gradcam(processed_img, heatmap)
                    if gradcam_img is not None:
                        col3, col4 = st.columns(2)
                        with col3:
                            st.image(processed_img, caption="Original", use_column_width=True)
                        with col4:
                            st.image(gradcam_img, caption="GradCAM Overlay", use_column_width=True)
    
    # Information section
    st.header("ℹ️ About This AI System")
    
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🧠 AI Architecture</h3>
            <p>Transfer Learning with ResNet50/VGG16</p>
            <p>Custom classification head</p>
        </div>
        """, unsafe_allow_html=True)
    
    with info_col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Dataset</h3>
            <p>5,863 chest X-ray images</p>
            <p>Normal vs Pneumonia classification</p>
        </div>
        """, unsafe_allow_html=True)
    
    with info_col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Features</h3>
            <p>GradCAM visualization</p>
            <p>Model comparison</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Disclaimer
    st.warning("⚠️ **Medical Disclaimer**: This AI system is for educational and research purposes only. Always consult qualified healthcare professionals for medical diagnosis and treatment.")

if __name__ == "__main__":
    main()