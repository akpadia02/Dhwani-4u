from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import librosa
import numpy as np
import os
import traceback
from pydub import AudioSegment
import speech_recognition as sr

app = Flask(__name__)
CORS(app)

# Load trained models and encoders
gender_model = joblib.load("models/clf_gender.pkl")
age_model = joblib.load("models/clf_age.pkl")
gender_encoder = joblib.load("models/gender_encoder.pkl")
age_encoder = joblib.load("models/age_encoder.pkl")

# --- Audio Preprocessing ---
def trim_silence(y):
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    return y_trimmed

def normalize_audio(y):
    peak = np.max(np.abs(y))
    return y / peak if peak > 0 else y

def resample_audio(y, orig_sr, target_sr=22050):
    return (librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr), target_sr) if orig_sr != target_sr else (y, orig_sr)

# --- Feature Extraction (56 Features) ---
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    y = trim_silence(y)
    y = normalize_audio(y)
    y, sr = resample_audio(y, sr, target_sr=22050)

    # MFCC (40)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Chroma (12)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Spectral features
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))

    # Combine into 56-length vector
    features = np.hstack([
        mfcc_mean,
        chroma_mean,
        spectral_centroid,
        spectral_bandwidth,
        spectral_rolloff,
        zcr
    ])

    return features

def transcribe_audio(file_path):
    recognizer = sr.Recognizer()

    # Convert audio to PCM WAV (required format)
    audio_wav_path = "converted_audio.wav"
    audio = AudioSegment.from_file(file_path)
    audio.export(audio_wav_path, format="wav")

    with sr.AudioFile(audio_wav_path) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return "Speech not clear"
        except sr.RequestError:
            return "Google API unavailable"

# --- Predict Endpoint ---
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio = request.files["audio"]
        audio_path = "temp_audio.wav"
        audio.save(audio_path)
        text = transcribe_audio(audio_path)

        features = extract_audio_features(audio_path)
        if features is None or features.shape[0] != 56:
            return jsonify({"error": "Invalid feature vector"}), 500

        features = features.reshape(1, -1)

        gender_pred = gender_model.predict(features)
        age_pred = age_model.predict(features)

        gender = gender_encoder.inverse_transform(gender_pred)[0]
        age = age_encoder.inverse_transform(age_pred)[0]

        # Clean up temp file
        os.remove(audio_path)

        return jsonify({
            "gender": gender,
            "age": age,
            "text": text 
        })

    except Exception as e:
        print("Error in /predict:", str(e))
        traceback.print_exc()
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(debug=True)
