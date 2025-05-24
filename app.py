import streamlit as st
import spacy
import os

# Download model at runtime if not already present
try:
    nlp = spacy.load("en_core_web_sm")
except:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    
from chatbot import classify_intent, extract_entities, get_response



st.title("🤖 orBIT Chatbot: Ask About IT, CS, or EMC!")
st.markdown("Welcome! Type your question below:")

user_input = st.text_input("💬 You:")

if user_input:
    intent = classify_intent(user_input)
    entities = extract_entities(user_input)
    response = get_response(intent, user_input)

    st.markdown(f"**🧠 Intent:** {intent}")
    if entities:
        st.markdown(f"**🔍 Entities:** {', '.join(entities)}")
    st.markdown(f"**🤖 orBIT:** {response}")
