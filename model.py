import os
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)
CORS(app)

Classnames= ['Aloevera', 'Amla', 'Amruthaballi', 'Arali', 'Astma_weed', 'Badipala', 'Balloon_Vine', 'Bamboo', 'Beans', 'Betel', 'Bhrami', 'Bringaraja', 'Caricature', 'Castor', 'Catharanthus', 'Chakte', 'Chilly', 'Citron lime (herelikai)', 'Common rue(naagdalli)', 'Coriender', 'Curry', 'Doddpathre', 'Drumstick', 'Ekka', 'Eucalyptus', 'Gasagase', 'Ginger', 'Globe Amarnath', 'Guava', 'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jackfruit', 'Jasmine', 'Kambajala', 'Kasambruga', 'Lantana', 'Lemon', 'Lemongrass', 'Malabar_Nut', 'Malabar_Spinach', 'Mango', 'Marigold', 'Mint', 'Neem', 'Nelavembu', 'Padri', 'Palak(Spinach)', 'Papaya', 'Parijatha', 'Pea', 'Pepper', 'Pomoegranate', 'Pumpkin', 'Raddish', 'Rose', 'Sampige', 'Sapota', 'Seethaashoka', 'Seethapala', 'Spinach1', 'Tamarind', 'Taro', 'Tecoma', 'Thumbe', 'Tomato', 'Tulsi', 'Turmeric', 'ashoka', 'camphor', 'laptop', 'testtest']

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/')
def home():
    return "Hello, this is the home page!"

# Function to preprocess the image to fit the model
def preprocess_image(image_path, input_size):
    try:
        # Open the image file
        img = Image.open(image_path)

        # Ensure the image is RGB (3 channels)
        img = img.convert('RGB')

        # Resize the image to 256x256 (as expected by the model)
        img = img.resize((input_size, input_size))

        # Convert the image to numpy array and normalize to 0-1
        img = np.array(img).astype('float32') / 255.0

        # Expand dimensions to fit the model input (batch size, 256, 256, 3)
        img = np.expand_dims(img, axis=0)

        print(f"Image preprocessed with shape: {img.shape}")
        return img
    except Exception as e:
        print(f"Error during image preprocessing: {str(e)}")
        return None

# Function to classify the image using TFLite model
def classify_image(image_path):
    input_size = input_details[0]['shape'][1]  # Expected input size (256)
    print(f"Expected input shape: {input_details[0]['shape']}")  # Debugging step
    input_data = preprocess_image(image_path, input_size)

    if input_data is None:
        return "Error during preprocessing"

    # Ensure the input shape matches the model's expected shape
    try:
        interpreter.set_tensor(input_details[0]['index'], input_data)
    except ValueError as e:
        print(f"Error during image classification: {str(e)}")
        return "Error during classification"

    # Run inference
    interpreter.invoke()

    # Get the results
    output_data = interpreter.get_tensor(output_details[0]['index'])
    # Get the index of the class with the highest probability
    predicted_class = np.argmax(output_data)

    # Get the confidence level of the predicted class
    confidence = output_data[0][predicted_class]

    return {"predicted_class": predicted_class, "confidence": confidence}

# Define an endpoint to receive images and return classification result
@app.route("/classify", methods=["POST"])
def classify_endpoint():
    print("Request to classify found")
    if 'file' not in request.files:
        print("No file part in the request")
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    print("Got the file")

    if file.filename == '':
        print("No file selected")
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            # Ensure the uploads directory exists
            upload_dir = "uploads"
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)
            
            # Save the file to a temporary location
            file_path = os.path.join(upload_dir, file.filename)
            file.save(file_path)
            print(f"File saved at {file_path}")

            # Run classification
            result = classify_image(file_path)

            if "error" in result:
                print("Error during classification")
                return jsonify(result), 500

            predicted_class = result['predicted_class']
            confidence = result['confidence']

            print(f"Class predicted: {predicted_class}, Confidence: {confidence}")

            # Remove the temporary file after inference
            os.remove(file_path)
            print(f"Temporary file removed: {file_path}")

            return jsonify({
                "predicted_class": Classnames[int(predicted_class)], 
                "confidence": float(confidence)  # Ensure the confidence is in a JSON serializable format
            })
        except Exception as e:
            print(f"An error occurred during classification: {str(e)}")
            return jsonify({"error": "Internal Server Error"}), 500

# Run the Flask app
if __name__ == "__main__":
    # Run the app, ensuring the uploads directory is created before startup
    app.run(host="0.0.0.0", port=5000)
