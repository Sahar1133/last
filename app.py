
import streamlit as st
import pandas as pd
import json
import pickle
import random
from utils import load_data, encode_data, generate_questions, train_model, calculate_info_gain

st.set_page_config(page_title="Career Path Predictor", layout="wide")
st.title("Career Path Predictor")

if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.encoders = None
    st.session_state.target_encoder = None
    st.session_state.questions = None
    st.session_state.selected_features = None

uploaded_file = st.file_uploader("Upload your dataset (Excel file)", type=["xlsx"])
if uploaded_file:
    df = load_data(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Summary")
    st.write(df.describe(include='all'))

    target_column = st.selectbox("Select the target column (career field)", df.columns)

    if st.button("Calculate Information Gain and Train Model"):
        info_gain = calculate_info_gain(df, target_column)
        threshold = st.slider("Select Information Gain Threshold", 0.0, 1.0, 0.01)
        selected_features = [feature for feature, gain in info_gain.items() if gain >= threshold]
        st.session_state.selected_features = selected_features

        st.write("Selected Features:", selected_features)
        X, y, encoders, target_encoder = encode_data(df, selected_features, target_column)
        model = train_model(X, y)

        questions = generate_questions(df, selected_features)
        with open("questions.json", "w") as f:
            json.dump(questions, f, indent=4)

        with open("model.pkl", "wb") as f:
            pickle.dump((model, encoders, target_encoder), f)

        st.session_state.model = model
        st.session_state.encoders = encoders
        st.session_state.target_encoder = target_encoder
        st.session_state.questions = questions

if st.session_state.model and st.session_state.questions:
    st.subheader("Answer the following questions")
    responses = {}
    for feature in st.session_state.selected_features:
        q_list = st.session_state.questions.get(feature, [])
        if q_list:
            q = random.choice(q_list)
            response = st.radio(q["question"], q["options"], key=feature)
            responses[feature] = response

    if st.button("Predict Career Field"):
        input_data = []
        for feature in st.session_state.selected_features:
            val = responses[feature]
            if feature in st.session_state.encoders:
                val = st.session_state.encoders[feature].transform([val])[0]
            input_data.append(val)

        pred = st.session_state.model.predict([input_data])[0]
        career = st.session_state.target_encoder.inverse_transform([pred])[0]
        st.success(f"Your predicted career field is: **{career}**")
        st.info("This career was predicted based on features like: " +
                ", ".join(st.session_state.selected_features))
