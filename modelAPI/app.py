import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from PIL import Image
import io
import json
import logging
import os
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model and class names
model = None
cat_to_name = None
idx_to_class = None

# Load class names from JSON
def load_class_names():
    global cat_to_name, idx_to_class
    try:
        logger.info("Loading class names...")
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
            
        # Create a mapping from index to class
        idx_to_class = {idx: class_id for idx, (class_id, _) in enumerate(sorted(cat_to_name.items()))}
        logger.info(f"Loaded {len(cat_to_name)} classes")
        logger.info(f"First few mappings: {dict(list(idx_to_class.items())[:5])}")
    except Exception as e:
        logger.error(f"Error loading class names: {str(e)}")
        raise

# Load model
def load_model():
    global model
    try:
        logger.info("Loading model...")
        # Use ResNet50 to match training
        model = models.resnet50(pretrained=False)
        # Freeze parameters
        for param in model.parameters():
            param.requires_grad = False
        # Replace final layer
        num_classes = len(cat_to_name) if cat_to_name else 102
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        # Load the trained weights
        model.load_state_dict(torch.load('flower_recognition_model.pth',
        map_location=torch.device('cpu')))
        model.eval()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# Initialize transform - match the validation/test transform from training
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Preprocess the input image
def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return transform(image).unsqueeze(0)
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

@app.before_first_request
def initialize():
    load_class_names()
    load_model()

@app.route('/predict', methods=['POST'])
def predict():
    logger.info("Received request")
    
    if 'image' not in request.files:
        logger.error("No 'image' in request.files")
        return jsonify({"error": "No file part"}), 400

    file = request.files['image']
    if file.filename == '':
        logger.error("Empty filename")
        return jsonify({"error": "No selected file"}), 400

    try:
        # Process image
        image_bytes = file.read()
        image_tensor = preprocess_image(image_bytes)

        # Get prediction
        with torch.no_grad():
            logger.info("Getting prediction...")
            output = model(image_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            
            # Get top 3 predictions
            top_probs, top_indices = torch.topk(probs, 3)
            
            predictions = []
            for i in range(3):
                idx = top_indices[i].item()
                class_id = idx_to_class[idx]  # Get the actual class ID from the mapping
                confidence = float(top_probs[i].item())
                class_name = cat_to_name.get(class_id, "Unknown flower")
                logger.info(f"Prediction {i+1}: idx={idx}, class_id={class_id}, name={class_name}, conf={confidence}")
                predictions.append({
                    'class_id': int(class_id),
                    'class_name': class_name,
                    'confidence': confidence
                })

        # Clear some memory
        del image_tensor, output, probs
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return jsonify(predictions)

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
