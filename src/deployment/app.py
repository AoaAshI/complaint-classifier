"""
Railway Complaint Classification Web App

A Flask service for classifying railway complaint images using a trained Keras model.
"""

from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# --- Configuration ---

app = Flask(__name__)

MODEL_PATH = 'models/saved_models/railway_complaint_classifier.h5'
model = None

# --- Model Loading ---

def load_model(model_path: str):
    """Attempt to load a local Keras model. Returns the model object or None."""
    if os.path.exists(model_path):
        try:
            model_obj = tf.keras.models.load_model(model_path)
            print("Model loaded successfully.")
            print(f"Model input shape: {model_obj.input_shape}")
            print(f"Model output shape: {model_obj.output_shape}")
            return model_obj
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Model file not found at: {model_path}")
    print("Please train the model first using: python src/training/train_model.py")
    return None

model = load_model(MODEL_PATH)

# --- Categories ---

class_names = [
    'ac', 'charging_port', 'fans', 'light',
    'pest', 'toilet', 'train_coach_exterior',
    'train_coach_interior', 'washbasin'
]

parent_category_map = {
    'ac': 'Electrical Equipment',
    'charging_port': 'Electrical Equipment',
    'fans': 'Electrical Equipment',
    'light': 'Electrical Equipment',
    'pest': 'Cleanliness',
    'toilet': 'Cleanliness',
    'washbasin': 'Cleanliness',
    'train_coach_exterior': 'Train Exterior',
    'train_coach_interior': 'Train Interior'
}

# --- Image Preprocessing ---

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert bytes to preprocessed numpy array for model inference."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")

# --- Inference Helper ---

def format_prediction_results(predictions: np.ndarray, class_names: list[str]):
    """
    Map model output to label, confidence, and all class probabilities.
    Returns (predicted_class, confidence, sorted_probs_dict)
    """
    try:
        preds = predictions.reshape(1, -1) if len(predictions.shape) == 1 else predictions
        if preds.shape[1] != len(class_names):
            raise ValueError(f"Model output mismatch: expected {len(class_names)} classes, got {preds.shape[1]}")
        pred_idx = int(np.argmax(preds[0]))
        pred_class = class_names[pred_idx]
        confidence = float(preds[0][pred_idx])
        prob_dict = {k: float(preds[0][i]) for i, k in enumerate(class_names)}
        return pred_class, confidence, dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
    except Exception as e:
        raise RuntimeError(f"Prediction formatting failed: {str(e)}")

# --- Routes ---

@app.route('/', methods=['GET', 'POST'])
def predict():
    """Main form handler. GET shows form; POST uploads file and returns result."""
    if request.method == 'GET':
        return render_template('index.html')

    if model is None:
        # Emoji here only, for very clear user feedback
        return render_template('index.html', error='❌ Model not loaded. Please train the model first.')

    file = request.files.get('file')
    if not file or file.filename == '':
        return render_template('index.html', error='No file selected.')

    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
        return render_template('index.html', error='Invalid file type. Please upload PNG, JPG, JPEG, GIF, or BMP images.')

    try:
        img_array = preprocess_image(file.read())
        predictions = model.predict(img_array, verbose=0)
        pred_class, confidence, all_predictions = format_prediction_results(predictions, class_names)
        parent_category = parent_category_map.get(pred_class, "Other")
        return render_template(
            'index.html',
            prediction=pred_class,
            parent_category=parent_category,
            confidence=confidence,
            all_predictions=all_predictions,
            filename=file.filename,
            success=True
        )
    except Exception as e:
        return render_template('index.html', error=f"Prediction failed: {str(e)}")

@app.route('/health')
def health_check():
    """Healthcheck API for model and app status."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH,
        'classes': len(class_names),
        'categories': class_names
    })

@app.route('/about')
def about():
    """System info for API consumers."""
    return jsonify({
        'system': 'Railway Complaint Classification System',
        'model': 'EfficientNetB0 trained from scratch',
        'classes': len(class_names),
        'categories': class_names,
        'description': (
            'AI-powered system to automatically categorize railway '
            'passenger complaints into 9 infrastructure categories.'
        )
    })

if __name__ == '__main__':
    print("Starting Railway Complaint Classifier Web Interface.")
    print("Access at: http://127.0.0.1:5001")
    print("Health check: http://127.0.0.1:5001/health")
    print("System info: http://127.0.0.1:5001/about")
    if model is not None:
        print(f"Model ready with {len(class_names)} complaint categories.")
        print("Categories: " + ", ".join([c.replace('_', ' ').title() for c in class_names]))
    else:
        print("Model not loaded - train first.")
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)
