# EDA dan Training Model Prediksi Diabetes (All-in-One Script)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ========== 1. Load Dataset ==========
DATA_PATH = 'data/diabetes_data.csv'
df = pd.read_csv(DATA_PATH)

print("Sample Data:\n", df.head(), "\n")
print("Data Info:\n")
print(df.info(), "\n")

# ========== 2. Cek Missing Values ==========
print("Missing Values:\n", df.isnull().sum(), "\n")

# ========== 3. Visualisasi Distribusi Target ==========
sns.countplot(data=df, x='diabetes')
plt.title("Distribusi Label Diabetes")
plt.savefig("output_label_distribution.png")
plt.clf()

# ========== 4. Korelasi Fitur Numerik ==========
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Korelasi antar Fitur Numerik")
plt.savefig("output_correlation.png")
plt.clf()

# ========== 5. Siapkan Data ==========
X = df.drop(columns='diabetes')
y = df['diabetes']

numeric_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
categorical_features = ['gender', 'smoking_history']
binary_features = ['hypertension', 'heart_disease']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(), categorical_features)
], remainder='passthrough')  # binary: pass-through

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== 6. Pipeline dan Training ==========
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)

# ========== 7. Evaluasi ==========
y_pred = pipeline.predict(X_test)

print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.title("Confusion Matrix")
plt.savefig("output_confusion_matrix.png")
plt.clf()

# ========== 8. Simpan Model ==========
os.makedirs("models", exist_ok=True)
MODEL_PATH = "models/diabetes_model.pkl"
joblib.dump(pipeline, MODEL_PATH)
print(f"\n✅ Model berhasil disimpan di {MODEL_PATH}")
