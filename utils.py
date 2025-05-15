
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif

def load_data(uploaded_file):
    return pd.read_excel(uploaded_file)

def encode_data(df, features, target_column):
    encoders = {}
    df_enc = df.copy()
    for feature in features:
        if df_enc[feature].dtype == "object":
            enc = LabelEncoder()
            df_enc[feature] = enc.fit_transform(df_enc[feature].astype(str))
            encoders[feature] = enc

    target_encoder = LabelEncoder()
    df_enc[target_column] = target_encoder.fit_transform(df_enc[target_column].astype(str))

    X = df_enc[features]
    y = df_enc[target_column]
    return X, y, encoders, target_encoder

def calculate_info_gain(df, target_column):
    df_clean = df.dropna()
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category").cat.codes
    if y.dtype == "object":
        y = pd.Series(LabelEncoder().fit_transform(y))
    info_gain = mutual_info_classif(X, y, discrete_features=True)
    return dict(zip(X.columns, info_gain))

def train_model(X, y):
    model = DecisionTreeClassifier()
    model.fit(X, y)
    return model

def generate_questions(df, features):
    questions = {}
    for feature in features:
        if feature.lower() in ["gpa", "years_of_experience", "certifications_count", "courses_completed", "github_repos"]:
            questions[feature] = [{
                "question": f"What is your {feature.replace('_', ' ')}?",
                "options": sorted(df[feature].dropna().unique().astype(str).tolist())
            }]
        elif feature.lower() == "work_style":
            questions[feature] = [
                {"question": "How do you prefer to structure your day?",
                 "options": ["Independent", "Flexible", "Collaborative"]},
                {"question": "When working on a group project, what role do you naturally take?",
                 "options": ["Independent", "Flexible", "Collaborative"]},
                {"question": "How do you feel when plans change last minute?",
                 "options": ["Independent", "Flexible", "Collaborative"]},
                {"question": "What kind of work environment brings out your best?",
                 "options": ["Independent", "Flexible", "Collaborative"]},
                {"question": "How do you prefer to receive assignments?",
                 "options": ["Independent", "Flexible", "Collaborative"]}
            ]
        else:
            unique_vals = df[feature].dropna().unique().astype(str).tolist()
            if len(unique_vals) > 1:
                questions[feature] = [{
                    "question": f"What is your {feature.replace('_', ' ')}?",
                    "options": unique_vals
                }]
    return questions
