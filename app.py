from flask import Flask, render_template, request
import tensorflow as tf
from keras.models import load_model
import numpy as np
import pickle as pk
import os
import sys
import io
from PIL import Image
import base64

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load labels
with open('my_object.pkl', 'rb') as file:
    loaded_object = pk.load(file)

labels = loaded_object


# Load the Keras model
#model = load_model(r"E:\data sciences\intel-image-classification\inter.keras")
# Load tflite model
conv = tf.lite.Interpreter(model_path=r"E:\data sciences\intel-image-classification\intel.tflite")
conv.allocate_tensors()
input_details = conv.get_input_details()
output_details = conv.get_output_details()

# Initialize Flask app
app = Flask(__name__, template_folder='template')

# Define the home route
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Define the route for predicting the image
@app.route('/Predict_Image', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        # Get the uploaded image file
        image_file = request.files['imagefile']

        # Save the uploaded image to a consistent path
        image_path = os.path.join('./saved_images', image_file.filename)
        image_file.save(image_path)

        # Open and encode the image to base64
        img = Image.open(image_path)
        data = io.BytesIO()
        img.save(data, "JPEG")
        encode_img_data = base64.b64encode(data.getvalue()).decode('utf-8')

        # Process the image
        def preprocess(image_path):
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.image.resize(image, size=[224, 224])
            image = tf.expand_dims(image, 0)  
            return image

        # Preprocess the uploaded image
        image = preprocess(image_path)

        # Predict the class of the image
        conv.set_tensor(input_details[0]['index'],image)
        conv.invoke()
        pred = conv.get_tensor(output_details[0]['index'])
        label = labels[np.argmax(pred)]

        # Render the prediction result and image in the HTML template
        return render_template('index.html', prediction=label, filename=encode_img_data)

    return render_template('index.html')


if __name__ == "__main__":
    # Ensure the directory for saving images exists
    os.makedirs('./saved_images', exist_ok=True)

    # Run the Flask app
    app.run(debug=True, port=5000)
