from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import librosa
import os

app = Flask(__name__)

# Load the trained emotion model
model_path = "models/emotion_model.h5"
model = tf.keras.models.load_model(model_path)
print(f"✅ Loaded emotion model from {model_path}")

# Define emotion labels (update these according to your model)
label_encoder = {0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad'}

def extract_audio_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        y, _ = librosa.effects.trim(y)

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)

        features = np.hstack([
            np.mean(mfccs, axis=1),
            np.mean(spec_cent),
            np.mean(spec_bw),
            np.mean(rolloff),
            np.mean(zcr),
            np.mean(chroma, axis=1)
        ])
        return features
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

@app.route('/predict-emotion', methods=['POST'])
def predict_emotion():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    file_path = "temp.wav"
    file.save(file_path)

    # Extract features
    features = extract_audio_features(file_path)
    if features is None:
        return jsonify({'error': 'Feature extraction failed'}), 500

    # Reshape for model input
    features = features.reshape(1, -1)

    # Predict emotion
    prediction = model.predict(features)
    predicted_label_index = np.argmax(prediction)
    predicted_emotion = label_encoder[predicted_label_index]

    return jsonify({'emotion': predicted_emotion})


if __name__ == '__main__':
    app.run(debug=True)
