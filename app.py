import streamlit as st
import joblib, os, numpy as np
st.set_page_config(page_title="Breast Cancer Prediction (All Features)", layout="wide")
st.title("Breast Cancer Prediction — All Features (Matches your model)")
MODEL_PATH = "model.pkl"
FEATURES_PATH = "feature_names.txt"
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("model.pkl not found. Please upload model.pkl to the repo root.")
        st.stop()
    return joblib.load(MODEL_PATH)
def load_feature_names():
    if not os.path.exists(FEATURES_PATH):
        st.error("feature_names.txt not found. Please upload feature_names.txt (one feature per line).")
        st.stop()
    with open(FEATURES_PATH) as f:
        names = [ln.strip() for ln in f if ln.strip()]
    return names
model = load_model()
feature_names = load_feature_names()
st.sidebar.header("Model info")
st.sidebar.write("Model file: model.pkl")
st.header("Enter feature values (all features)")
cols = st.columns(3)
inputs = []
for i, fname in enumerate(feature_names):
    with cols[i % 3]:
        val = st.number_input(label=fname, format="%.6f", key=fname, value=0.0)
        inputs.append(val)
if st.button("Predict"):
    try:
        arr = np.array(inputs).reshape(1, -1)
        if arr.shape[1] != getattr(model, "n_features_in_", arr.shape[1]):
            st.error(f"Model expects {getattr(model,'n_features_in_','unknown')} features but got {arr.shape[1]}. Make sure feature_names.txt and model.pkl match.")
        else:
            pred = model.predict(arr)
            proba = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(arr)[0].max()
            if pred[0] == 1 or (isinstance(pred[0], str) and str(pred[0]).lower().startswith('m')):
                st.error(f"⚠️ Prediction: Malignant. (prob={proba})")
            else:
                st.success(f"✅ Prediction: Benign. (prob={proba})")
    except Exception as e:
        st.exception(e)
