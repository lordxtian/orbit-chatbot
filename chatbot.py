import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import openai
import spacy
import os

# Load OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY", "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")

# Load FAQ dataset
faq_df = pd.read_csv("faq_data.csv")

# Use spaCy blank English model (no download needed)
nlp = spacy.blank("en")

# Train a simple classifier for intent detection
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(faq_df["question"])
y = faq_df["intent"]
clf = LogisticRegression().fit(X, y)

def classify_intent(user_input):
    X_input = vectorizer.transform([user_input])
    return clf.predict(X_input)[0]

def extract_entities(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents]

def get_response(intent, user_input):
    for _, row in faq_df.iterrows():
        if row["intent"] == intent and row["question"].lower() in user_input.lower():
            return row["answer"]
    return get_fallback_gpt(user_input)

def get_fallback_gpt(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an educational assistant helping students learn about IT, CS, and EMC programs."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception:
        return "Sorry, I couldn't find an answer at the moment."
