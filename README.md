# emotion-classifier
The Emotion Classifier is an intelligent NLP-based application that detects human emotions from text inputs using machine learning. It helps determine whether a given sentence expresses joy, sadness, anger, fear, love, or surprise, and displays an appropriate emoji for better visual feedback.

# Emotion Classifier

## Dataset Used
This project uses the **`dair-ai/emotion`** dataset from Hugging Face, which contains text samples labeled with six emotions: `anger`, `fear`, `joy`, `love`, `sadness`, and `surprise`.

## Approach Summary
- Text data is vectorized using **TF-IDF** to capture important word features.
- A **Logistic Regression** model is trained on the encoded labels for multi-class emotion classification.
- The model achieves around **85% accuracy** on the test set.
- The classifier is deployed with a **Streamlit** web app, providing real-time emotion prediction and emoji visualization.

## Dependencies
- Python 3.x
- pandas
- scikit-learn
- datasets
- joblib
- streamlit
- seaborn
- matplotlib

