from flask import Flask, request, jsonify
import numpy as np
import cv2
import base64
from keras.models import load_model

app = Flask(__name__)

# Load the Keras model
model = load_model('C:/Users/PMLS/Desktop/pythonProject1/converted_keras (1)/keras_model.h5')  # Replace with your actual model path


def preprocess_image(image_base64):
    # Decode the base64 image
    img_data = base64.b64decode(image_base64)
    np_img = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Preprocess the image for your model (resize, normalize, etc.)
    img = cv2.resize(img, (64, 64))  # Example size, adjust to your model's input size
    img = img / 255.0  # Normalize if needed
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    return img


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_base64 = data['image']

    # Preprocess the image and get the prediction
    processed_img = preprocess_image(image_base64)
    prediction = model.predict(processed_img)

    # Assuming the output is a softmax with probabilities
    predicted_character = chr(np.argmax(prediction) + 65)  # Assuming A-Z prediction

    return jsonify({'character': predicted_character})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
