# Traffic Sign Recognition System using CNN

## Overview

This project is a Traffic Sign Recognition System developed using TensorFlow/Keras and Streamlit. It uses a Convolutional Neural Network (CNN) to classify traffic signs from uploaded images. The model is trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

---

## Features

- Traffic sign classification using CNN
- Image upload and prediction
- Streamlit-based user interface
- Trained on the GTSRB dataset
- Easy-to-use application

---

## Technologies Used

- Python
- TensorFlow
- Keras
- Streamlit
- NumPy
- Pandas
- OpenCV
- Matplotlib

---

## Project Structure

```
Traffic-Sign-Recognition-System/
│
├── app.py
├── train_model.py
├── predict.py
├── traffic_sign_model.keras
│
├── Train/
├── Test/
├── Meta/
├── Train.csv
├── Test.csv
├── Meta.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset

This project uses the German Traffic Sign Recognition Benchmark (GTSRB) dataset, which contains images belonging to 43 different traffic sign classes.

---

## How to Run

1. Clone the repository

```bash
git clone https://github.com/harpreetkaur34600/Traffic-Sign-Recognition-System-CNN.git
```

2. Install the required libraries

```bash
pip install -r requirements.txt
```

3. Run the Streamlit application

```bash
streamlit run app.py
```

---

## Future Improvements

- Real-time traffic sign detection using a webcam
- Mobile application support
- Driver assistance features
- Improved model accuracy and performance

---
## Application Preview

### Main Interface

![Main Interface](images/main_interface.png)

### Another Prediction

![Prediction](images/another_prediction.png)
 ---

## Author

Harpreet Kaur

B.Tech Computer Science Engineering Student