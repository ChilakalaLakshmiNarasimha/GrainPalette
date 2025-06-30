import tensorflow as tf
import tensorflow_hub as hub
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import cv2

# Load the model
model = tf.keras.models.load_model(
    filepath='rice_model.h5',
    custom_objects={'KerasLayer': hub.KerasLayer}
)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/details')
def pred():
    return render_template('details.html')

@app.route('/result', methods=['POST'])
def predict():
    if request.method == 'POST':
        f = request.files['image']
        filename = secure_filename(f.filename)

        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        f.save(upload_path)

        # Image preprocessing
        img = cv2.imread(upload_path)
        img = cv2.resize(img, (223, 223))  # ✅ Correct
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Prediction
        pred = model.predict(img)
        pred_class = pred.argmax()

        df_labels = {
            0: 'arborio',
            1: 'basmati',
            2: 'ipsala',
            3: 'jasmine',
            4: 'karacadag'
        }

        prediction = df_labels.get(pred_class, "Unknown")
        return render_template('results.html', prediction_text=prediction, image_path='images/' + filename)

    return render_template('index.html')

# Main entry point
if __name__ == "__main__":
    app.run(debug=True)
