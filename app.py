import streamlit as st
import joblib, os, numpy as np

st.set_page_config(page_title="Breast Cancer Prediction (Top 10)", layout="centered")
st.title("Breast Cancer Prediction — Top 10 Features")

MODEL_PATH = "model.pkl"
FEATURES_PATH = "feature_names.txt"

def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("model.pkl not found in repo root.")
        st.stop()
    return joblib.load(MODEL_PATH)

def load_feature_names():
    if not os.path.exists(FEATURES_PATH):
        st.error("feature_names.txt not found in repo root.")
        st.stop()
    with open(FEATURES_PATH) as f:
        names = [ln.strip() for ln in f if ln.strip()]
    return names

model = load_model()
feature_names = load_feature_names()

st.sidebar.header("Model info")
st.sidebar.write(f"Model type: **{type(model).__name__}**")
st.sidebar.write(f"Model expects: **{getattr(model, 'n_features_in_', 'unknown')}** features (Top 10).")

st.write("Enter values for the 10 features below (use numbers).")

cols = st.columns(2)
inputs = []
for i, fname in enumerate(feature_names):
    with cols[i % 2]:
        val = st.number_input(label=fname, format="%.6f", key=fname, value=0.0)
        inputs.append(val)

if st.button("Predict"):
    try:
        arr = np.array(inputs).reshape(1, -1)
        if arr.shape[1] != getattr(model, "n_features_in_", arr.shape[1]):
            st.error(f"Model expects {getattr(model,'n_features_in_','unknown')} features but got {arr.shape[1]}. Ensure feature_names.txt and model.pkl match.")
        else:
            pred = model.predict(arr)
            proba = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(arr)[0].max()

            label = pred[0]
            if isinstance(label, (int, float)):
                is_malignant = int(label) == 1
            else:
                is_malignant = str(label).lower().startswith('m') or str(label).lower().startswith('1')

            if is_malignant:
                st.error(f"⚠️ Prediction: Malignant. (prob={proba})")
            else:
                st.success(f"✅ Prediction: Benign. (prob={proba})")
    except Exception as e:
        st.exception(e)
