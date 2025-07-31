import gradio as gr
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Load the .h5 model
model = load_model("human_model.h5", compile=False)

# Label map
label_map = {
    0: 'angry',
    1: 'calm',
    2: 'disgust',
    3: 'fearful',
    4: 'happy',
    5: 'neutral',
    6: 'sad',
    7: 'surprised'
}

# Function to extract mel spectrogram
def extract_mel_spectrogram(file_path, sr=22050, n_mels=128, fmax=8000):
    y, sr = librosa.load(file_path, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
    log_mel = librosa.power_to_db(mel)
    mel_spec = librosa.util.fix_length(log_mel, size=130, axis=1)
    return mel_spec

# Emotion prediction function
def predict_emotion(audio):
    mel = extract_mel_spectrogram(audio)
    input_data = mel[np.newaxis, ..., np.newaxis]  # Add batch and channel dims
    pred = model.predict(input_data)
    predicted_emotion = label_map[np.argmax(pred)]
    return f"Predicted Emotion: {predicted_emotion}"

# Gradio interface
gr.Interface(
    fn=predict_emotion,
    inputs=gr.Audio(type="filepath", label="Upload WAV audio"),
    outputs="text",
    title="🎤 Speech Emotion Recognition",
    description="Upload a .wav file to detect the emotion using a CNN model trained on RAVDESS."
).launch()
