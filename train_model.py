import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Load dataset
df = pd.read_csv('data/diabetes_data.csv')

# Features and target
X = df.drop(columns='diabetes')
y = df['diabetes']

# Preprocessing
numeric_features = ['age', 'bmi', 'hba1c_level', 'blood_glucose_level']
categorical_features = ['gender', 'smoking_history']
binary_features = ['hypertension', 'heart_disease']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(), categorical_features)
], remainder='passthrough')  # binary features pass through

# Build pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, 'models/diabetes_model.pkl')
