import streamlit as st
import pandas as pd
import json
import pickle
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Function definitions (copied from the first cell)
def compute_information_gain(df, target_column):
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include=['object']).columns:
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])
    X = df_encoded.drop(columns=[target_column])
    y = df_encoded[target_column]
    ig = mutual_info_classif(X, y, discrete_features='auto')
    return pd.Series(ig, index=X.columns).sort_values(ascending=False)

def generate_questions(df):
    questions = {}
    for col in df.columns:
        unique_vals = df[col].dropna().unique().tolist()[:4]
        questions[col] = [{
            "question": f"What best describes your preference for {col}?",
            "options": unique_vals
        }]
    return questions

def encode_features(df, features, target):
    encoders = {}
    X = df[features].copy()
    y = df[target]

    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)

    return X, y, encoders, target_encoder

def train_model(df, features, target):
    X, y, encoders, target_encoder = encode_features(df, features, target)
    model = DecisionTreeClassifier()
    model.fit(X, y)
    return model, encoders, target_encoder

# Streamlit app starts here
st.set_page_config(page_title="Career Predictor", layout="wide")
st.title("Career Path Predictor using Decision Tree and Information Gain")

uploaded_file = st.file_uploader("Upload your dataset (CSV/XLSX)", type=["csv", "xlsx"])

if uploaded_file:
    file_ext = uploaded_file.name.split('.')[-1]
    df = pd.read_excel(uploaded_file) if file_ext == "xlsx" else pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    target = st.selectbox("Select target column", df.columns)
    if target:
        ig_scores = compute_information_gain(df, target)
        st.subheader("Information Gain Scores")
        st.write(ig_scores)

        threshold = st.slider("Select IG threshold", 0.0, 1.0, 0.01, 0.01)
        selected_features = ig_scores[ig_scores > threshold].index.tolist()
        st.write(f"Selected Features ({len(selected_features)}):", selected_features)

        if st.button("Train and Generate Questions"):
            questions = generate_questions(df[selected_features])
            with open("questions.json", "w") as f:
                json.dump(questions, f, indent=2)

            model, encoders, target_encoder = train_model(df, selected_features, target)
            with open("model.pkl", "wb") as f:
                pickle.dump((model, encoders, target_encoder), f)

            st.success("Model trained and questions saved!")

if os.path.exists("questions.json") and os.path.exists("model.pkl"):
    st.header("Career Prediction Questionnaire")
    with open("questions.json") as f:
        questions = json.load(f)
    with open("model.pkl", "rb") as f:
        model, encoders, target_encoder = pickle.load(f)

    responses = {}
    for feature, qlist in questions.items():
        q = qlist[0]
        response = st.radio(q["question"], q["options"], key=feature)
        responses[feature] = response

    if st.button("Predict Career Field"):
        input_data = []
        for feature in responses:
            # Ensure the feature is in the encoders dictionary before attempting to transform
            val = responses[feature]
            if feature in encoders:
                 val = encoders[feature].transform([val])[0]
            # If the feature is not in encoders, it might be a numerical feature, so keep the value as is
            input_data.append(val)

        # Ensure input_data has the same number of features as the model was trained on
        # This might require careful handling of missing features or different ordering
        # For simplicity, assuming the order in `responses` matches the order in `selected_features` during training
        try:
            pred = model.predict([input_data])[0]
            result = target_encoder.inverse_transform([pred])[0]
            st.success(f"Predicted Career Field: **{result}**")
        except ValueError as e:
             st.error(f"Error predicting: {e}. Make sure the number and order of features in the input match the trained model.")
             st.write("Input data:", input_data)
             st.write("Expected features (based on encoders):", list(encoders.keys()))