import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os
from pathlib import Path
import spacy
from spacy.cli import download
from spacy.util import load_model_from_path
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"))

faq_df = pd.read_csv("faq_data.csv")
faq_df["question"] = faq_df["question"].str.lower()

MODEL_PATH = Path("local_models/en_core_web_sm")
if MODEL_PATH.exists():
    nlp = load_model_from_path(MODEL_PATH)
else:
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(MODEL_PATH)
    nlp = load_model_from_path(MODEL_PATH)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(faq_df["question"])
y = faq_df["intent"]
clf = LogisticRegression().fit(X, y)

def preprocess(text):
    return text.lower().strip()

def classify_intent(user_input):
    user_input = preprocess(user_input)
    X_input = vectorizer.transform([user_input])
    return clf.predict(X_input)[0]

def extract_entities(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents]

def get_response(intent, user_input):
    processed_input = preprocess(user_input)
    for _, row in faq_df.iterrows():
        if row["intent"] == intent and row["question"] in processed_input:
            return row["answer"]
    return get_fallback_gpt(user_input)

def get_fallback_gpt(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for a college chatbot."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ GPT fallback failed: {str(e)}"