from flask import Flask, request, jsonify, send_from_directory
import numpy as np
from keras.models import load_model
from PIL import Image
import base64
import io
import os
from flask_cors import CORS


# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Load your Keras model
model = load_model('C:/Users/PMLS/Desktop/pythonProject1/Model/keras_model.h5')

# Preprocessing function
def preprocess_image(image, target_size):
    image = image.resize(target_size)  # Resize the image to the model's input size
    image = np.array(image)  # Convert to NumPy array
    image = image / 255.0  # Normalize pixel values (0-1 range)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/')
def index():
    print("Serving index.html from:", os.path.join(app.root_path, 'static'))
    return send_from_directory(os.path.join(app.root_path, 'static'), 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the image data from the request
    data = request.json
    image_data = data.get('image')

    if not image_data:
        return jsonify({"error": "No image data provided"}), 400

    try:
        # Decode base64 image
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    except Exception as e:
        return jsonify({"error": "Invalid image data"}), 400

    # Preprocess the image
    target_size = (64, 64)  # Update this to match your model's input size
    processed_image = preprocess_image(image, target_size)

    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=-1)  # Get the class index
    character = chr(predicted_class[0] + 65)  # Convert to a letter (assuming 'A' = class 0)

    return jsonify({"character": character})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)

