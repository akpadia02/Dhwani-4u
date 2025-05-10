import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers, optimizers, callbacks
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import joblib

# Load the dataset
df = pd.read_csv("audio_features.csv")
df = pd.get_dummies(df, columns=['gender'])

# Prepare features and labels
X = df.drop(columns=['filename', 'age', 'gender_female', 'gender_male'])
y = df['age']

# Encode class labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)

# Build the MLP model
def build_model(input_shape, output_units):
    model = models.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(output_units, activation='softmax')
    ])
    return model

# Compile and train the model
model = build_model(X_train.shape[1], y_categorical.shape[1])
optimizer = optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4)
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# Callbacks
early_stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
lr_reduce = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

# Training
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2, callbacks=[early_stop, lr_reduce], verbose=2)

# Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")

# Save model and scalers
joblib.dump(scaler, "scaler1.pkl")
joblib.dump(label_encoder, "label_encoder1.pkl")
model.save("mlp_age_classifier.h5")
