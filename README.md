# 🎤 Speech Emotion Recognition Web App

This project is a deep learning-based **speech emotion recognition** system. It takes `.wav` audio files as input and predicts the speaker's emotional state (like *angry*, *happy*, *calm*, etc.) using a trained CNN model.

---

## 🚀 Live Demo

You can try out the app instantly via **Hugging Face Spaces**:

👉 [🌐 Launch App](https://huggingface.co/spaces/Aadi75240/Speech_Emotion_Recognition)

---

## 🔍 Project Overview

- **Model:** Convolutional Neural Network (CNN)
- **Framework:** TensorFlow / Keras
- **Input:** `.wav` audio files
- **Output:** One of 8 emotions

Emotions detected: angry, calm, disgust, fearful, happy, neutral, sad, surprised


---

## 🧠 Dataset

- **RAVDESS** – Ryerson Audio-Visual Database of Emotional Speech and Song  
  [More Info](https://zenodo.org/record/1188976)

---

## 🛠️ Tech Stack

- Python 3.10
- TensorFlow 2.15
- Librosa for audio processing
- Gradio for the UI
- Hugging Face Spaces for deployment

---

## 🖥️ How to Run Locally

```bash
git clone https://github.com/Aadi7524/speech-emotion-recognition.git
cd speech-emotion-recognition
pip install -r requirements.txt
python app.py

