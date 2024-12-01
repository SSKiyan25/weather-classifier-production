import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)

# Load your pre-trained model
model = load_model('weather-model.keras')

# Function to preprocess the image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(127, 127))  # target_size as per the model
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension
    return img_array

# Function to make a prediction
def predict_weather(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    weather_conditions = ['Cloudy', 'Rainy', 'Sunny']
    predicted_label = weather_conditions[predicted_class]
    confidence = np.max(predictions) * 100  # Confidence as a percentage
    return predicted_label, confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'imageFile' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        imageFile = request.files['imageFile']
        imagePath = os.path.join('./images', imageFile.filename)
        imageFile.save(imagePath)
        
        # Make prediction
        predicted_weather, confidence = predict_weather(imagePath)

        # Pass the prediction result and percentage to the template
        return redirect(url_for('index', prediction=predicted_weather, confidence=str(confidence)))
    
    prediction = request.args.get('prediction')
    confidence = request.args.get('confidence')
    
    # Convert confidence back to float if it exists
    if confidence:
        confidence = float(confidence)
    
    return render_template('index.html', prediction=prediction, confidence=confidence)

if __name__ == '__main__':
    app.run()