# app.py (Streamlit inference)
import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

ARTIFACT_DIR = "/app/model_artifacts"  # when deploying, let this path point to folder with joblib + json
# For local testing on Kaggle, set ARTIFACT_DIR = "/model_artifacts"

CLF_PATH = ARTIFACT_DIR + "/clf.joblib"
SCALER_PATH = ARTIFACT_DIR + "/scaler.joblib"
FEATURES_JSON = ARTIFACT_DIR + "/feature_columns.json"

@st.cache_resource
def load_models(artifact_dir):
    clf = joblib.load(artifact_dir + "/clf.joblib")
    scaler = joblib.load(artifact_dir + "/scaler.joblib")
    with open(artifact_dir + "/feature_columns.json","r") as f:
        feature_cols = json.load(f)
    # load SBERT from HF (Streamlit cloud will download once)
    sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return clf, scaler, feature_cols, sbert

def clean_text(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|httpsS+', ' URL ', text, flags=re.MULTILINE)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'\\([^\\]+)\\*', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

COMMERCIAL_KEYWORDS = [...]
LEGAL_KEYWORDS = [...]

def extract_advanced_features(text):
    # same implementation as training (copy-paste)
    if pd.isna(text):
        text = ""
    text = str(text)
    features = {}
    features['length'] = len(text)
    features['word_count'] = len(text.split())
    features['avg_word_length'] = np.mean([len(w) for w in text.split()]) if text.split() else 0
    features['has_url'] = 1 if re.search(r'http\S+|www\S+', text) else 0
    features['url_count'] = len(re.findall(r'http\S+|www\S+', text))
    features['has_shortened_url'] = 1 if re.search(r'bit\.ly|goo\.gl|tinyurl|t\.co', text.lower()) else 0
    features['special_char_count'] = len(re.findall(r'[!@#$%^&*()_+=\[\]{};:\'",.<>?/\\|`~]', text))
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['upper_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
    features['has_all_caps_word'] = 1 if re.search(r'\b[A-Z]{3,}\b', text) else 0
    features['has_email'] = 1 if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text) else 0
    features['has_phone'] = 1 if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text) else 0
    text_lower = text.lower()
    features['commercial_keyword_count'] = sum(1 for kw in COMMERCIAL_KEYWORDS if kw in text_lower)
    features['has_price'] = 1 if re.search(r'\$\d+|\d+\s*(dollar|usd|euro)', text_lower) else 0
    features['legal_keyword_count'] = sum(1 for kw in LEGAL_KEYWORDS if kw in text_lower)
    features['has_question_about_law'] = 1 if re.search(r'(should i|can i|is it legal|is this legal)', text_lower) else 0
    features['sentence_count'] = len(re.findall(r'[.!?]+', text))
    features['has_question'] = 1 if '?' in text else 0
    words = text_lower.split()
    features['word_diversity'] = len(set(words)) / len(words) if len(words) > 0 else 0
    return features

def build_feature_row(body_text, sbert):
    # create a single-row DataFrame with same pipeline as training
    body_clean = clean_text(body_text)
    manual = extract_advanced_features(body_clean)
    # For inference, we don't have rule or positive/negative examples.
    # So we set examples to empty string -> embeddings zeros (we will encode empty string)
    b_emb = sbert.encode([body_clean])[0]
    empty_emb = sbert.encode([""])[0]
    # compute similarity features vs empty examples -> zeros
    sim_pos1 = float(cosine_similarity([b_emb], [empty_emb])[0][0])
    sim_pos2 = sim_pos1
    sim_neg1 = sim_pos1
    sim_neg2 = sim_pos1
    sim_pos_mean = (sim_pos1 + sim_pos2)/2.0
    sim_neg_mean = (sim_neg1 + sim_neg2)/2.0
    sim_rule = float(cosine_similarity([b_emb], [empty_emb])[0][0])
    sim_feats = {
        'sim_pos_max': max(sim_pos1, sim_pos2),
        'sim_pos_min': min(sim_pos1, sim_pos2),
        'sim_pos_mean': sim_pos_mean,
        'sim_neg_max': max(sim_neg1, sim_neg2),
        'sim_neg_min': min(sim_neg1, sim_neg2),
        'sim_neg_mean': sim_neg_mean,
        'sim_rule': sim_rule,
        'sim_diff': sim_pos_mean - sim_neg_mean,
        'sim_ratio': sim_pos_mean / (sim_neg_mean + 1e-6)
    }
    # combine
    feat = {}
    feat.update(sim_feats)
    feat.update(manual)
    feat.update({'is_advertising_rule': 0, 'is_legal_rule': 0})
    # insert dummy row_id (Streamlit doesn't need it but training had row_id first)
    feat_row = pd.DataFrame([feat])
    feat_row.insert(0, 'row_id', 0)
    return feat_row

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Jigsaw Rule Classifier — SBERT + LogisticRegression")
st.write("Masukkan 1 teks (komentar Reddit). Model akan memprediksi `rule` yang relevan.")

# Load
artifact_dir = st.text_input("Artifact folder path (set to '/model_artifacts' for local Kaggle)", value="/model_artifacts")
clf, scaler, feature_cols, sbert = load_models(artifact_dir)

user_text = st.text_area("Masukkan teks di sini:", height=200)

if st.button("Prediksi"):
    if not user_text.strip():
        st.warning("Teks kosong.")
    else:
        st.info("Membuat fitur & menghitung embedding (SBERT)... tunggu sebentar.")
        feat_row = build_feature_row(user_text, sbert)
        # Reindex columns to match scaler / training order
        # During training we saved feature columns list: feature_cols
        # Now we need to arrange feat_row columns into that exact order
        feat_row_reindexed = feat_row.reindex(columns=feature_cols, fill_value=0)
        # drop row_id for scaler.transform (training scaler was fitted on columns including row_id)
        X_to_scale = feat_row_reindexed.values
        X_scaled = scaler.transform(X_to_scale)
        import numpy as np
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
        # drop row_id then predict
        X_for_pred = X_scaled_df.drop('row_id', axis=1).values
        pred = clf.predict(X_for_pred)[0]
        proba = None
        try:
            proba = clf.predict_proba(X_for_pred).max()
        except Exception:
            proba = None
        st.success(f"Prediksi rule: **{pred}**")
        if proba is not None:
            st.write(f"Confidence: {proba:.3f}")
