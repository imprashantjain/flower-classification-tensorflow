# 🌸 Flower Classification using TensorFlow & MobileNetV2

An end-to-end image classification project built using TensorFlow, Keras, and MobileNetV2 Transfer Learning.

##  Project Overview

This project classifies flower images into 5 categories:

- Daisy
- Dandelion
- Roses
- Sunflowers
- Tulips

The project started with a custom CNN baseline and was later improved using MobileNetV2 transfer learning.

---

## Results

### CNN Baseline

- Training Accuracy: 85%
- Validation Accuracy: 64%

### MobileNetV2 Transfer Learning

- Training Accuracy: 95%
- Validation Accuracy: 87.6%

---

## Tech Stack

- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy

---

## Project Structure

```text
flower/
│
├── src/
│   ├── load_data.py
│   ├── model.py
│   ├── train.py
│   ├── train_mobilenet.py
│   └── predict.py
│
├── flower_model.keras
├── flower_mobilenet.keras
│
├── README.md
├── requirements.txt
└── .gitignore
```

##  Concepts Learned

- Deep Learning
- CNN Architecture
- Conv2D
- MaxPooling
- Overfitting
- Transfer Learning
- MobileNetV2
- Model Inference

---

##  Future Improvements

- Streamlit Web App
- Docker Deployment
- Fine-tuning MobileNetV2
- Cloud Deployment

---

##  Author

Prashant Jain