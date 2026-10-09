# ASL Fingerspelling Recognition and Text-to-Speech System

## Project Overview
This project implements a deep learning-based system designed to recognize American Sign Language (ASL) fingerspelling and convert it into readable text and audible speech. Utilizing a Convolutional Neural Network (CNN), the system processes hand gesture images, classifies them into alphabetical characters (A-Z) and a space character, and subsequently generates an audio output using Text-to-Speech (TTS) technologies.

**Author:** Eng. Jaafar Saleh  
**Role:** Sole Developer, Systems Engineer, and Machine Learning Practitioner  
**Institution:** Tartous University, Faculty of Information and Communication Engineering (2023-2024)

---

## Key Features
- **Image Preprocessing Pipeline:** Automated background removal, grayscale conversion, Gaussian blurring, and adaptive thresholding (Otsu's method) to isolate hand features.
- **Deep Learning Classification:** A custom-trained CNN model achieving **96.73% accuracy** across 27 classes (A-Z and Space).
- **Text-to-Speech Integration:** Converts the recognized sequence of letters into an audible `.mp3` file using the `gTTS` (Google Text-to-Speech) library.
- **Modular Architecture:** Separated workflows for model training (via Google Colab) and local inference/application execution.

---

## Technical Stack
- **Programming Language:** Python 3.11
- **Machine Learning Frameworks:** TensorFlow, Keras
- **Computer Vision:** OpenCV, NumPy
- **Audio Processing:** gTTS, playsound
- **Environment:** Jupyter Notebook, Google Colab (for training)

---

## Model Architecture
The CNN model (`asl_classifier.h5`) is structured as follows:
1. **Input Layer:** 128x128 grayscale images.
2. **Convolutional Block 1:** `Conv2D` (32 filters, 3x3, ReLU) $\rightarrow$ `MaxPooling2D` (2x2).
3. **Convolutional Block 2:** `Conv2D` (32 filters, 3x3, ReLU) $\rightarrow$ `MaxPooling2D` (2x2).
4. **Flatten Layer:** Converts 2D feature maps to a 1D vector.
5. **Dense Block 1:** `Dense` (128 units, ReLU) $\rightarrow$ `Dropout` (40%).
6. **Dense Block 2:** `Dense` (96 units, ReLU) $\rightarrow$ `Dropout` (40%).
7. **Dense Block 3:** `Dense` (64 units, ReLU).
8. **Output Layer:** `Dense` (27 units, Softmax) representing the 27 classes.

---

## Directory Structure

├── Final_App.ipynb          # Main inference application (Image reading & TTS)
├── Training_Explained.ipynb # Data preprocessing and model training pipeline
├── asl_classifier.h5        # The trained CNN model weights
├── test/                    # Dataset directory (27 folders for classes 0-26)
├── LeJaafar/                # Processed images ready for inference
├── leRead/                  # Raw captured images
├── old exps/                # Archived previous model iterations
└── Helpful files/           # Reference materials and auxiliary scripts


---

## Installation and Usage

### Prerequisites
Ensure you have Python installed along with the required libraries:
```bash
pip install tensorflow opencv-python numpy gTTS playsound matplotlib scikit-learn
```

### 1. Training the Model
Open `Training_Explained.ipynb`. This notebook handles:
- Loading the dataset from the `test/` directory.
- Applying the OpenCV preprocessing pipeline.
- Training the CNN model and evaluating its accuracy.
- Saving the final model as `asl_classifier.h5`.

### 2. Running the Inference Application
Open `Final_App.ipynb`. This notebook handles:
- Loading the trained `asl_classifier.h5` model.
- Reading preprocessed images from the `LeJaafar/` directory.
- Predicting the characters and concatenating them into a string.
- Generating the speech audio file (`welcome2121.mp3`).

---

## Engineering Challenges and Limitations
During the development lifecycle, several technical challenges were identified and documented:
1. **Real-Time Camera Noise:** Direct real-time webcam inference yielded lower accuracy due to excessive background details. *Workaround:* The system was optimized to process steady, pre-captured images rather than a raw live video stream.
2. **Resolution Constraints:** Google Colab's memory limitations required resizing images to 128x128, which restricted the model's ability to capture micro-features in real-time scenarios.
3. **TTS Dependencies:** The `gTTS` library requires an active internet connection and cannot be natively compiled for offline hardware simulation (e.g., Raspberry Pi via Proteus).

---

## Future Work
- Expanding the dataset using high-shutter-speed cameras to capture dynamic motion gestures.
- Transitioning to an offline TTS engine (e.g., `pyttsx3`) to enable deployment on edge devices like the Raspberry Pi.
- Upgrading the hardware infrastructure to support higher resolution inputs (e.g., 256x256 or 512x512) without memory bottlenecks.

---

## License
This project is developed for academic and research purposes.
```
