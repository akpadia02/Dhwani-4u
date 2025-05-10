from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import librosa
import numpy as np
import os
import traceback
from pydub import AudioSegment
import speech_recognition as sr
import warnings
import time
import google.generativeai as genai
import json
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler


# Load environment variables from .env file
load_dotenv()
scaler = StandardScaler()

app = Flask(__name__)
# Configure CORS
CORS(app, resources={r"/*": {"origins": "*"}})

# Suppress scikit-learn version warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize Gemini API
try:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if not GEMINI_API_KEY:
        print("WARNING: GEMINI_API_KEY not found in environment variables")
    else:
        genai.configure(api_key=GEMINI_API_KEY)
        print("Gemini API configured successfully")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    traceback.print_exc()

print("Loading ML models...")
# Load trained models and encoders with error handling
try:
    gender_model = joblib.load("models/clf_gender.pkl")
    # age_model = joblib.load("models/clf_age.pkl")
    gender_encoder = joblib.load("models/gender_encoder.pkl")
    # age_encoder = joblib.load("models/age_encoder.pkl")
    scaler = joblib.load("scaler1.pkl")             # Scaler used during training
    age_encoder = joblib.load("label_encoder1.pkl") # LabelEncoder used for age
    age_model = load_model("mlp_age_classifier.h5")    # Trained MLP model


    print("All ML models loaded successfully!")
except Exception as e:
    print(f"Error loading ML models: {e}")
    traceback.print_exc()

# --- Audio Preprocessing Functions ---
def trim_silence(y):
    try:
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        return y_trimmed
    except Exception as e:
        print(f"Error trimming silence: {e}")
        return y  # Return original if trimming fails

def normalize_audio(y):
    try:
        peak = np.max(np.abs(y))
        return y / peak if peak > 0 else y
    except Exception as e:
        print(f"Error normalizing audio: {e}")
        return y  # Return original if normalization fails

def resample_audio(y, orig_sr, target_sr=22050):
    try:
        if orig_sr != target_sr:
            return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr), target_sr
        return y, orig_sr
    except Exception as e:
        print(f"Error resampling audio: {e}")
        return y, orig_sr  # Return original if resampling fails

# --- Feature Extraction ---
def extract_audio_features(file_path):
    try:
        print(f"Starting feature extraction for: {file_path}")
        start_time = time.time()
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return None
            
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        print(f"Audio loaded: {len(y)} samples, {sr} Hz")
        
        # Process audio
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
        
        elapsed_time = time.time() - start_time
        print(f"Feature extraction complete. Features shape: {features.shape}, Time: {elapsed_time:.2f}s")
        
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        traceback.print_exc()
        return None

def transcribe_audio(file_path):
    audio_wav_path = "converted_audio.wav"
    try:
        print(f"Starting transcription for: {file_path}")
        recognizer = sr.Recognizer()

        # Convert audio to PCM WAV (required format)
        audio = AudioSegment.from_file(file_path)
        audio.export(audio_wav_path, format="wav")
        print("Audio converted to WAV format")

        with sr.AudioFile(audio_wav_path) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)
                print(f"Transcription result: {text}")
                return text
            except sr.UnknownValueError:
                print("Speech not clear enough to transcribe")
                return "Speech not clear"
            except sr.RequestError as e:
                print(f"Google API error: {e}")
                return "Google API unavailable"
    except Exception as e:
        print(f"Error in transcription: {e}")
        traceback.print_exc()
        return "Error transcribing audio"
    finally:
        # Clean up
        if os.path.exists(audio_wav_path):
            try:
                os.remove(audio_wav_path)
                print(f"Temporary WAV file removed: {audio_wav_path}")
            except Exception as e:
                print(f"Error removing temporary file: {e}")

def analyze_intent_with_gemini(text):
    """
    Use Gemini API to analyze the intent of the transcribed text
    """
    try:
        if not GEMINI_API_KEY:
            print("Gemini API key not available")
            return "Intent analysis unavailable (API key not configured)"
            
        if text in ["Speech not clear", "Google API unavailable", "Error transcribing audio"]:
            return "Intent analysis not possible due to transcription issues"
            
        print(f"Analyzing intent for text: {text}")
        
        # Configure Gemini model
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Prompt for intent analysis
        prompt = f"""
        Analyze the following text and determine the primary intent/purpose of the speaker.
        Common intents include: Question, Request, Information Sharing, Command, Complaint, 
        Greeting, Farewell, Confirmation, Clarification, Opinion, Suggestion.
        
        Please respond with ONLY the single most appropriate intent category from the list above
        or a similar short category if more appropriate. Return just the intent word or phrase, 
        with no additional explanation or punctuation.
        
        Text to analyze: "{text}"
        """
        
        # Generate response
        response = model.generate_content(prompt)
        intent = response.text.strip()
        print(f"Gemini intent analysis result: {intent}")
        
        return intent
    except Exception as e:
        print(f"Error analyzing intent with Gemini: {e}")
        traceback.print_exc()
        return "Intent analysis error"

def analyze_emotion_with_gemini(text):
    """
    Use Gemini API to analyze the emotional tone of the transcribed text
    """
    try:
        if not GEMINI_API_KEY:
            print("Gemini API key not available")
            return "Emotion analysis unavailable (API key not configured)"
            
        if text in ["Speech not clear", "Google API unavailable", "Error transcribing audio"]:
            return "Emotion analysis not possible due to transcription issues"
            
        print(f"Analyzing emotion for text: {text}")
        
        # Configure Gemini model
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Prompt for emotion analysis
        prompt = f"""
        Analyze the following text and determine the primary emotional tone of the speaker.
        Common emotions include: Happy, Sad, Angry, Surprised, Confused, Excited, Neutral, 
        Frustrated, Worried, Curious, Amused, or Concerned.
        
        Please respond with ONLY the single most appropriate emotion category from the list above
        or a similar short emotion if more appropriate. Return just the emotion word, with no 
        additional explanation or punctuation.
        
        Text to analyze: "{text}"
        """
        
        # Generate response
        response = model.generate_content(prompt)
        emotion = response.text.strip()
        print(f"Gemini emotion analysis result: {emotion}")
        
        return emotion
    except Exception as e:
        print(f"Error analyzing emotion with Gemini: {e}")
        traceback.print_exc()
        return "Emotion analysis error"

# --- API Endpoints ---
@app.route("/health", methods=["GET"])
def health_check():
    """Simple endpoint to check if the server is running"""
    return jsonify({
        "status": "healthy",
        "message": "Server is running",
        "gemini_api": "configured" if GEMINI_API_KEY else "not configured"
    })

@app.route("/predict", methods=["POST"])
def predict():
    temp_path = None
    try:
        print("Received prediction request")
        
        # Check if audio file is in the request
        if "audio" not in request.files:
            print("No audio file in request")
            return jsonify({"error": "No audio file provided"}), 400

        audio = request.files["audio"]
        if audio.filename == '':
            print("Empty filename")
            return jsonify({"error": "Empty audio file provided"}), 400
            
        print(f"Processing audio file: {audio.filename}")
        
        # Save the file temporarily
        temp_path = "temp_audio" + os.path.splitext(audio.filename)[1]
        audio.save(temp_path)
        print(f"Audio saved to: {temp_path}")
        
        # Extract features for ML prediction
        # features = extract_audio_features(temp_path).reshape(1,56)
        # input_features_scaled = scaler.transform(features)
        # Feature extraction
        # Print debug messages to confirm feature extraction
        print("Starting feature extraction...")

        # Feature extraction
        features = extract_audio_features(temp_path)

        # Check if features are extracted properly
        print(f"Extracted features: {features.shape}")

        if features is None:
            raise ValueError("Feature extraction failed. Features are None.")

        # Reshape and scale features
        features_reshaped = features.reshape(1, -1)
        print(f"Features reshaped: {features_reshaped.shape}")

        # Apply scaling
        input_features_scaled = scaler.transform(features_reshaped)


        # Gender prediction
        gender_pred = gender_model.predict(features_reshaped)

        # Age prediction
        pred_probs = age_model.predict(input_features_scaled)
        predicted_class = np.argmax(pred_probs, axis=1)
        age = age_encoder.inverse_transform(predicted_class)[0]

        # Gender prediction
        gender = gender_encoder.inverse_transform(gender_pred)[0]

        # Output results
        print(f"Prediction results - Gender: {gender}, Age: {age}")

       


        print(f"Extracted features shape: {features.shape}")
        print(f"Expected model input shape: {age_model.input_shape}")
        print(f"Scaler expected shape: {scaler.mean_.shape}")


        
        if features is None:
            print("Feature extraction failed")
            return jsonify({"error": "Feature extraction failed"}), 500
            
        if features.shape[0] != 56:
            print(f"Feature shape mismatch: {features.shape}")
            return jsonify({"error": f"Feature shape mismatch: {features.shape}"}), 500

        print("Feature extraction successful")

        # Transcribe audio
        text = "Transcription unavailable"
        try:
            text = transcribe_audio(temp_path)
        except Exception as e:
            print(f"Transcription error: {e}")
            traceback.print_exc()

        # Make gender and age predictions
        features = features.reshape(1, -1)
        gender_pred = gender_model.predict(features)
        # age_pred = age_model.predict(features)

        pred_probs = age_model.predict(input_features_scaled)
        predicted_class = np.argmax(pred_probs, axis=1)
        age = age_encoder.inverse_transform(predicted_class)[0]


        gender = gender_encoder.inverse_transform(gender_pred)[0]
        # age = age_encoder.inverse_transform(age_pred)[0]
        
        print(f"Prediction results - Gender: {gender}, Age: {age}")
        
        # Analyze intent and emotion with Gemini (if text is available)
        intent = "Unknown"
        emotion = "Unknown"
        
        if text not in ["Transcription unavailable", "Speech not clear", "Google API unavailable", "Error transcribing audio"]:
            intent = analyze_intent_with_gemini(text)
            emotion = analyze_emotion_with_gemini(text)

        result = {
            "gender": gender,
            "age": age,
            "text": text,
            "intent": intent,
            "emotion": emotion
        }
        
        return jsonify(result)

    except Exception as e:
        print("Error in /predict:", str(e))
        traceback.print_exc()
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"Temporary file removed: {temp_path}")
            except Exception as e:
                print(f"Error removing temp file: {e}")

if __name__ == "__main__":
    print("Starting Flask server on http://localhost:5000")
    # Set threaded=True for better handling of multiple requests
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)


































# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import joblib
# import librosa
# import numpy as np
# import os
# import traceback
# from pydub import AudioSegment
# import speech_recognition as sr
# import google.generativeai as genai

# app = Flask(__name__)
# CORS(app)

# genai.configure(api_key="AIzaSyB3WHAbXHed5E-9DB4qOhF6WTnxseTyGdw")  # Replace with actual API key
# model = genai.GenerativeModel("gemini-1.5-pro-latest")

# # Load trained models and encoders
# gender_model = joblib.load("models/clf_gender.pkl")
# age_model = joblib.load("models/clf_age.pkl")
# gender_encoder = joblib.load("models/gender_encoder.pkl")
# age_encoder = joblib.load("models/age_encoder.pkl")


# # --- Audio Preprocessing ---
# def trim_silence(y):
#     y_trimmed, _ = librosa.effects.trim(y, top_db=20)
#     return y_trimmed

# def normalize_audio(y):
#     peak = np.max(np.abs(y))
#     return y / peak if peak > 0 else y

# def resample_audio(y, orig_sr, target_sr=22050):
#     return (librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr), target_sr) if orig_sr != target_sr else (y, orig_sr)

# # --- Feature Extraction (56 Features) ---
# def extract_audio_features(file_path):
#     y, sr = librosa.load(file_path, sr=None)
#     y = trim_silence(y)
#     y = normalize_audio(y)
#     y, sr = resample_audio(y, sr, target_sr=22050)

#     # MFCC (40)
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
#     mfcc_mean = np.mean(mfcc, axis=1)

#     # Chroma (12)
#     chroma = librosa.feature.chroma_stft(y=y, sr=sr)
#     chroma_mean = np.mean(chroma, axis=1)

#     # Spectral features
#     spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
#     spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
#     spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
#     zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))

#     # Combine into 56-length vector
#     features = np.hstack([
#         mfcc_mean,
#         chroma_mean,
#         spectral_centroid,
#         spectral_bandwidth,
#         spectral_rolloff,
#         zcr
#     ])

#     return features

# def transcribe_audio(file_path):
#     recognizer = sr.Recognizer()

#     # Convert audio to PCM WAV (required format)
#     audio_wav_path = "converted_audio.wav"
#     audio = AudioSegment.from_file(file_path)
#     audio.export(audio_wav_path, format="wav")

#     with sr.AudioFile(audio_wav_path) as source:
#         audio_data = recognizer.record(source)
#         try:
#             return recognizer.recognize_google(audio_data)
#         except sr.UnknownValueError:
#             return "Speech not clear"
#         except sr.RequestError:
#             return "Google API unavailable"

# def intent(text):
#     prompt = f"Analyze the following sentence and return only the speaker's intent in 1-2 words:\n\n{text}"
#     try:
#         response = model.generate_content(prompt)
#         return response.text.strip() if response and response.text else "No response."
#     except Exception as e:
#         return f"Error: {str(e)}"
        
# # --- Predict Endpoint ---
# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         if "audio" not in request.files:
#             return jsonify({"error": "No audio file provided"}), 400

#         audio = request.files["audio"]
#         audio_path = "temp_audio.wav"
#         audio.save(audio_path)
#         text = transcribe_audio(audio_path)
#         detected_intent = intent(text)

#         features = extract_audio_features(audio_path)
#         if features is None or features.shape[0] != 56:
#             return jsonify({"error": "Invalid feature vector"}), 500

#         features = features.reshape(1, -1)

#         gender_pred = gender_model.predict(features)
#         age_pred = age_model.predict(features)

#         gender = gender_encoder.inverse_transform(gender_pred)[0]
#         age = age_encoder.inverse_transform(age_pred)[0]

#         # Clean up temp file
#         os.remove(audio_path)

#         return jsonify({
#             "gender": gender,
#             "age": age,
#             "text": text,
#             "intent": detected_intent
#         })

#     except Exception as e:
#         print("Error in /predict:", str(e))
#         traceback.print_exc()
#         return jsonify({"error": "Internal Server Error"}), 500

# if __name__ == "__main__":
#     app.run(debug=True)


# /////////
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import joblib
# import librosa
# import numpy as np
# import os
# import traceback
# from pydub import AudioSegment
# import speech_recognition as sr

# app = Flask(__name__)
# CORS(app)

# # Load trained models and encoders
# gender_model = joblib.load("models/clf_gender.pkl")
# age_model = joblib.load("models/clf_age.pkl")
# gender_encoder = joblib.load("models/gender_encoder.pkl")
# age_encoder = joblib.load("models/age_encoder.pkl")

# # --- Audio Preprocessing ---
# def trim_silence(y):
#     y_trimmed, _ = librosa.effects.trim(y, top_db=20)
#     return y_trimmed

# def normalize_audio(y):
#     peak = np.max(np.abs(y))
#     return y / peak if peak > 0 else y

# def resample_audio(y, orig_sr, target_sr=22050):
#     return (librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr), target_sr) if orig_sr != target_sr else (y, orig_sr)

# # --- Feature Extraction (56 Features) ---
# def extract_audio_features(file_path):
#     y, sr = librosa.load(file_path, sr=None)
#     y = trim_silence(y)
#     y = normalize_audio(y)
#     y, sr = resample_audio(y, sr, target_sr=22050)

#     # MFCC (40)
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
#     mfcc_mean = np.mean(mfcc, axis=1)

#     # Chroma (12)
#     chroma = librosa.feature.chroma_stft(y=y, sr=sr)
#     chroma_mean = np.mean(chroma, axis=1)

#     # Spectral features
#     spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
#     spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
#     spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
#     zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))

#     # Combine into 56-length vector
#     features = np.hstack([
#         mfcc_mean,
#         chroma_mean,
#         spectral_centroid,
#         spectral_bandwidth,
#         spectral_rolloff,
#         zcr
#     ])

#     return features

# def transcribe_audio(file_path):
#     recognizer = sr.Recognizer()

#     # Convert audio to PCM WAV (required format)
#     audio_wav_path = "converted_audio.wav"
#     audio = AudioSegment.from_file(file_path)
#     audio.export(audio_wav_path, format="wav")

#     with sr.AudioFile(audio_wav_path) as source:
#         audio_data = recognizer.record(source)
#         try:
#             return recognizer.recognize_google(audio_data)
#         except sr.UnknownValueError:
#             return "Speech not clear"
#         except sr.RequestError:
#             return "Google API unavailable"

# # --- Predict Endpoint ---
# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         if "audio" not in request.files:
#             return jsonify({"error": "No audio file provided"}), 400

#         audio = request.files["audio"]
#         audio_path = "temp_audio.wav"
#         audio.save(audio_path)
#         text = transcribe_audio(audio_path)

#         features = extract_audio_features(audio_path)
#         if features is None or features.shape[0] != 56:
#             return jsonify({"error": "Invalid feature vector"}), 500

#         features = features.reshape(1, -1)

#         gender_pred = gender_model.predict(features)
#         age_pred = age_model.predict(features)

#         gender = gender_encoder.inverse_transform(gender_pred)[0]
#         age = age_encoder.inverse_transform(age_pred)[0]

#         # Clean up temp file
#         os.remove(audio_path)

#         return jsonify({
#             "gender": gender,
#             "age": age,
#             "text": text 
#         })

#     except Exception as e:
#         print("Error in /predict:", str(e))
#         traceback.print_exc()
#         return jsonify({"error": "Internal Server Error"}), 500

# if __name__ == "__main__":
#     app.run(debug=True)