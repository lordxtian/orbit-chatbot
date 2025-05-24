import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import spacy
import openai
import os

# Load data
faq_df = pd.read_csv("faq_data.csv")
nlp = spacy.load("en_core_web_sm")

# Load OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")

# Train intent classifier
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
            messages=[{"role": "system", "content": "You are an educational chatbot that answers questions about IT, CS, and EMC programs."},
                      {"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception:
        return "I'm sorry, I couldn't find an answer right now."