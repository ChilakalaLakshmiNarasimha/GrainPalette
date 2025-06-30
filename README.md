
# 🍚 Rice Type Detection - Deep Learning Classifier

Rice Type Detection is a deep learning-powered web application that identifies the type of rice from an uploaded image. Built using TensorFlow and Flask, this app predicts the rice grain category from a trained CNN model. The tool is ideal for researchers, agricultural stakeholders, and food industry experts.

---

## 🚀 Features

- 📷 Upload rice grain images in JPG, JPEG, or PNG format
- 🤖 Real-time rice type prediction using a CNN (MobileNetV2) model
- 📊 Displays predicted rice class from trained categories
- 💻 Clean and simple HTML user interface (3 pages: Index, Details, Results)
- 🔒 Local Flask-based deployment
- 🧠 Supports multiple rice types (customizable based on dataset)

---

## 🏷 Supported Rice Types

This project can support any number of rice types depending on your dataset. Example types might include:

- Basmati  
- Jasmine  
- Arborio  
- Others...

---

## 🏗 Tech Stack

| Component       | Technology       |
|----------------|------------------|
| Frontend       | HTML5, CSS       |
| Model          | TensorFlow / Keras (MobileNetV2) |
| Backend        | Python + Flask   |
| Image Handling | Pillow, Keras Preprocessing |
| Visualization  | Matplotlib       |

---

## 🧪 Model Training

The model is trained using:
- MobileNetV2 as a base CNN (pre-trained on ImageNet)
- Global Average Pooling and Dense layers for classification
- Categorical cross-entropy loss
- `ImageDataGenerator` for real-time image preprocessing
- Model saved as `rice_model.h5`

---

## 🗂 Folder Structure

```
RiceTypeDetection/
├── dataset/                # Image folders per rice class
│   ├── Basmati/
│   ├── Jasmine/
│   └── ...
├── static/
│   └── uploads/            # Uploaded images go here
├── templates/
│   ├── index.html
│   ├── details.html
│   └── results.html
├── train_model.py          # CNN training script
├── app.py                  # Flask server
├── rice_model.h5           # Trained model
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/rice-type-detection.git
cd rice-type-detection
```

### 2. Install Required Libraries
```bash
pip install tensorflow keras flask matplotlib pillow
```

### 3. Prepare Dataset
Ensure your dataset is in the `dataset/` folder, structured like:
```
dataset/
  Basmati/
  Jasmine/
  Arborio/
  ...
```

---

## 🏋️‍♀️ Train the Model

Run the training script:
```bash
python train_model.py
```

This will:
- Load and preprocess image data
- Train a CNN model based on MobileNetV2
- Save the trained model as `rice_model.h5`

---

## ▶️ Run the Application

Start the Flask web server:
```bash
python app.py
```

Then open your browser and visit:
```
http://127.0.0.1:5000/
```

### User Flow:
1. Upload an image of rice
2. Confirm the uploaded image
3. View the predicted rice type


---

## 📃 License

This project is open-source under the MIT License. You are free to use, modify, and distribute it.

---

## 👨‍💻 Author
Project: *Rice Type Detection with Deep Learning*
