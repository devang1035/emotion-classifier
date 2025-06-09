import streamlit as st
import joblib
import numpy as np

# Load model components
model = joblib.load("model/emotion_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
encoder = joblib.load("model/label_encoder.pkl")

# Emoji mapping
emoji_map = {
    "joy": "😄",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲"
}

# Page config
st.set_page_config(page_title="Emotion Classifier", page_icon="🧠", layout="centered")

# Sidebar
with st.sidebar:
    st.title("🧠 Emotion Classifier")
    st.markdown("Predict emotions from text using ML.\n\nMade with ❤️ by Devang Patel")
    st.markdown("📊 Model: Logistic Regression + TF-IDF")
    st.markdown("🧪 Trained on: dair-ai/emotion")
    st.markdown("---")
    st.caption("Try inputs like:")
    st.caption("- I am so happy today!")
    st.caption("- I'm feeling scared.")
    st.caption("- That was surprising!")

# Main UI
st.markdown("## 🎯 Enter a sentence to detect its emotion:")
text = st.text_input("Your sentence here", placeholder="e.g., I'm feeling fantastic today!")

if "history" not in st.session_state:
    st.session_state.history = []

if text and text.strip():
    vec = vectorizer.transform([text])
    pred_num = model.predict(vec)[0]
    pred_proba = model.predict_proba(vec)[0]

    label = encoder.inverse_transform([pred_num])[0]
    label_lower = label.lower()
    emoji = emoji_map.get(label_lower, "🧠")
    confidence = np.max(pred_proba) * 100

    st.markdown(f"""
    ### ✅ Prediction:  
    <div style="font-size: 32px;">{emoji} <strong>{label.capitalize()}</strong> ({confidence:.1f}%)</div>
    """, unsafe_allow_html=True)

    # Save to history
    st.session_state.history.append((text, label.capitalize(), emoji, confidence))

    # Show history
    with st.expander("🕓 Prediction History"):
        for idx, (sent, lbl, emo, conf) in enumerate(reversed(st.session_state.history), 1):
            st.markdown(f"**{idx}.** _{sent}_ → {emo} **{lbl}** ({conf:.1f}%)")

else:
    st.info("Enter a sentence above to get started!")
