import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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