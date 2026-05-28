import streamlit as st
import pandas as pd
import joblib

# Load model dengan mekanisme fallback otomatis jika terjadi ketidakcocokan versi pickle
try:
    model = joblib.load('models/diabetes_model.pkl')
except Exception as e:
    import os
    st.info("🔄 Menyiapkan model untuk pertama kali di lingkungan ini, harap tunggu sekitar 5 detik...")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    
    # Memuat dataset
    df = pd.read_csv('data/diabetes_data.csv')
    
    # Fitur dan target
    X = df.drop(columns='diabetes')
    y = df['diabetes']
    
    # Preprocessing
    numeric_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    categorical_features = ['gender', 'smoking_history']
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ], remainder='passthrough')
    
    # Build pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Fit model
    model.fit(X, y)
    
    # Pastikan direktori models ada
    os.makedirs('models', exist_ok=True)
    
    # Simpan model agar loading berikutnya instan
    joblib.dump(model, 'models/diabetes_model.pkl')
    st.success("✅ Model berhasil disiapkan dan dikonfigurasi!")

# CSS untuk memperindah tampilan
st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        .stTitle {
            font-size: 3em !important;
        }
    </style>
""", unsafe_allow_html=True)

# Judul aplikasi
st.title("🩺 Prediksi Diabetes")
st.markdown("Masukkan data pasien di bawah ini untuk memprediksi apakah pasien menderita **diabetes**.")

# Form input
with st.form("prediction_form"):
    st.subheader("📋 Data Pasien")

    # Inputan biasa (angka, text)
    age = st.number_input("Usia", min_value=0, max_value=120, value=30)
    bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=22.0)
    HbA1c_level = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5)
    blood_glucose_level = st.number_input("Kadar Glukosa Darah", min_value=50, max_value=500, value=120)

    # Radio button untuk pilihan biner
    gender = st.radio("Jenis Kelamin", ["Male", "Female", "Other"])
    hypertension = st.radio("Hipertensi", ["Tidak", "Ya"])
    heart_disease = st.radio("Penyakit Jantung", ["Tidak", "Ya"])
    smoking_history = st.radio("Riwayat Merokok", ["never", "current", "former", "not current", "ever", "No Info"])

    submitted = st.form_submit_button("🔍 Prediksi")

    if submitted:
        # Encode nilai biner
        hypertension_bin = 1 if hypertension == "Ya" else 0
        heart_disease_bin = 1 if heart_disease == "Ya" else 0

        # Buat DataFrame input
        input_data = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "bmi": bmi,
            "hypertension": hypertension_bin,
            "heart_disease": heart_disease_bin,
            "smoking_history": smoking_history,
            "HbA1c_level": HbA1c_level,
            "blood_glucose_level": blood_glucose_level
        }])

        # Prediksi
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]
        proba_percent = proba * 100

        # Hasil prediksi
        st.subheader("📊 Hasil Prediksi:")
        if prediction == 1:
            st.error(f"❌ Pasien **berpotensi menderita diabetes**")
            st.markdown(f"**Probabilitas: {proba_percent:.2f}%**")
        else:
            st.success(f"✅ Pasien **tidak menderita diabetes**")
            st.markdown(f"**Probabilitas: {proba_percent:.2f}%**")

        # Visualisasi pie chart (opsional)
        st.markdown("### 🔬 Probabilitas Visualisasi")
        st.pyplot(
            pd.Series(
                [proba_percent, 100 - proba_percent],
                index=["Diabetes", "Tidak Diabetes"]
            ).plot.pie(
                autopct='%1.1f%%', 
                colors=['#ff6b6b', '#1dd1a1'],
                figsize=(4, 4),
                ylabel=''
            ).figure
        )
