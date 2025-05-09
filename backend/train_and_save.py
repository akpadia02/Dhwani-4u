import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("features_with_labels.csv")

# ✅ Optional: Keep only these age groups
df = df[df['age'].isin(['teens', 'twenties', 'thirties'])]

# ✅ Encode target labels
gender_encoder = LabelEncoder()
df['gender_encoded'] = gender_encoder.fit_transform(df['gender'])

age_encoder = LabelEncoder()
df['age_encoded'] = age_encoder.fit_transform(df['age'])

# ✅ Drop non-feature columns (filename, path, labels)
X = df.drop(columns=['filename', 'path', 'gender', 'age', 'gender_encoded', 'age_encoded'], errors='ignore')
y_gender = df['gender_encoded']
y_age = df['age_encoded']

# ✅ Split for training
X_train, X_test, y_gender_train, y_gender_test = train_test_split(X, y_gender, test_size=0.2, random_state=42)
_, _, y_age_train, y_age_test = train_test_split(X, y_age, test_size=0.2, random_state=42)

# ✅ Train classifiers
clf_gender = RandomForestClassifier()
clf_gender.fit(X_train, y_gender_train)

clf_age = RandomForestClassifier()
clf_age.fit(X_train, y_age_train)

# ✅ Evaluation (optional)
gender_preds = clf_gender.predict(X_test)
print("🎯 Gender Accuracy:", accuracy_score(y_gender_test, gender_preds))
print(classification_report(y_gender_test, gender_preds, target_names=gender_encoder.classes_))

age_preds = clf_age.predict(X_test)
print("🎯 Age Accuracy:", accuracy_score(y_age_test, age_preds))
print(classification_report(y_age_test, age_preds, target_names=age_encoder.classes_))

# ✅ Save models and encoders
os.makedirs("models", exist_ok=True)
joblib.dump(gender_encoder, "models/gender_encoder.pkl")
joblib.dump(age_encoder, "models/age_encoder.pkl")
joblib.dump(clf_gender, "models/clf_gender.pkl")
joblib.dump(clf_age, "models/clf_age.pkl")

print("✅ Models and encoders saved successfully.")
